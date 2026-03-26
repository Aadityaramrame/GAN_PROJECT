# =============================================================================
# WGAN-GP: Wasserstein GAN with Gradient Penalty
# University GAN Lab — PyTorch Implementation
#
# Architecture:
#   Generator  : Latent vector (100) → 3x128x128 image (DCGAN-style)
#   Critic     : 3x128x128 image → scalar score (Spectral Norm, no BatchNorm)
#
# Loss:
#   Critic  : E[D(fake)] - E[D(real)] + λ * GradientPenalty
#   Generator: -E[D(fake)]
#
# Reference: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
# =============================================================================

import os
import random
import warnings

# Suppress PIL palette image warnings (cosmetic only, not an error)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =============================================================================
# 1. DEVICE
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 2. HYPERPARAMETERS
# =============================================================================

Z_DIM        = 100          # Latent vector dimension
IMG_SIZE     = 128          # Spatial resolution (128x128)
IMG_CHANNELS = 3            # RGB
FEATURES_G   = 64           # Base feature map count for Generator
FEATURES_D   = 64           # Base feature map count for Critic

LR_G         = 1e-4         # Generator learning rate
LR_D         = 4e-5         # Critic learning rate (TTUR — slightly slower)
BETAS        = (0.0, 0.9)   # Adam betas — β1=0 recommended for WGAN-GP

BATCH_SIZE   = 64
NUM_EPOCHS   = 200          # More epochs → better separation score
CRITIC_ITERS = 7            # Critic updates per generator update (increased from 3)
LAMBDA_GP    = 10           # Gradient penalty coefficient
SAVE_EVERY   = 5            # Save generated samples every N epochs
NUM_WORKERS  = 6

# Evaluation split: hold out 10% of real images for clean eval
EVAL_SPLIT   = 0.10

# =============================================================================
# 3. DATASET
#
# Folder structure:
#   GAN/
#     Real_9/
#       aloo_gobi/    ← subfolders required by ImageFolder
#       aloo_methi/
#       ...
#
# Labels from ImageFolder are IGNORED — GAN only uses raw pixel values.
#
# Images are converted to RGB to handle any palette/transparency edge cases.
# =============================================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Fix palette/RGBA warnings
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # Normalise to [-1, 1]
        std=[0.5, 0.5, 0.5]
    )
])

full_dataset = datasets.ImageFolder(root="./Real_9", transform=transform)

# Split into train and eval subsets for clean evaluation
eval_size  = int(len(full_dataset) * EVAL_SPLIT)
train_size = len(full_dataset) - eval_size
train_dataset, eval_dataset = random_split(
    full_dataset,
    [train_size, eval_size],
    generator=torch.Generator().manual_seed(SEED)
)

loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True  # Full batches only — required for stable gradient penalty
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

print(f"Total images    : {len(full_dataset)}")
print(f"Training images : {train_size}")
print(f"Eval images     : {eval_size}")
print(f"Classes         : {full_dataset.classes}")

# =============================================================================
# 4. GENERATOR
#
# z (Z_DIM x 1 x 1) → ConvTranspose2d blocks → 3 x 128 x 128
#
# Spatial progression: 1 → 4 → 8 → 16 → 32 → 64 → 128
# =============================================================================

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, img_channels=IMG_CHANNELS, features_g=FEATURES_G):
        super(Generator, self).__init__()

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.net = nn.Sequential(
            # Stem: z_dim x 1 x 1 → features_g*16 x 4 x 4
            nn.ConvTranspose2d(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(inplace=True),

            up_block(features_g * 16, features_g * 8),  # → 8 x 8
            up_block(features_g * 8,  features_g * 4),  # → 16 x 16
            up_block(features_g * 4,  features_g * 2),  # → 32 x 32
            up_block(features_g * 2,  features_g),      # → 64 x 64

            # Output layer: no BatchNorm, Tanh activation
            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1] to match normalised real images
        )

        self._init_weights()

    def _init_weights(self):
        """DCGAN-style weight init: N(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

# =============================================================================
# 5. CRITIC (DISCRIMINATOR)
#
# Input : 3 x 128 x 128
# Output: scalar Wasserstein score — NO Sigmoid
#
# Design:
#   - Spectral Normalisation on ALL Conv layers (Lipschitz constraint)
#   - NO BatchNorm (incompatible with gradient penalty)
#   - LeakyReLU(0.2) throughout
#   - Deeper than generator for stronger discriminative power
#
# Spatial progression: 128 → 64 → 32 → 16 → 8 → 4 → 1
# =============================================================================

class Critic(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, features_d=FEATURES_D):
        super(Critic, self).__init__()

        def sn_conv(in_c, out_c, kernel=4, stride=2, pad=1):
            """Strided Conv with Spectral Norm + LeakyReLU"""
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            # 3 x 128 x 128 → fd x 64 x 64
            sn_conv(img_channels,   features_d),

            # fd x 64 → fd*2 x 32
            sn_conv(features_d,     features_d * 2),

            # fd*2 x 32 → fd*4 x 16
            sn_conv(features_d * 2, features_d * 4),

            # fd*4 x 16 → fd*8 x 8
            sn_conv(features_d * 4, features_d * 8),

            # fd*8 x 8 → fd*16 x 4  (deepest downsampling)
            sn_conv(features_d * 8, features_d * 16),

            # Extra conv at 4x4 — increases depth without changing spatial size
            nn.utils.spectral_norm(
                nn.Conv2d(features_d * 16, features_d * 16, kernel_size=3, stride=1, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Another extra conv — stronger feature extraction
            nn.utils.spectral_norm(
                nn.Conv2d(features_d * 16, features_d * 8, kernel_size=3, stride=1, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Final: fd*8 x 4 x 4 → 1 x 1 x 1
            nn.utils.spectral_norm(
                nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
            )
            # NO Sigmoid — raw scalar score for Wasserstein loss
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), 1)  # Shape: (batch_size, 1)

# =============================================================================
# 6. GRADIENT PENALTY
#
# Creates interpolated samples between real and fake, computes critic score,
# then penalises gradients that deviate from unit norm.
#
# GP = E[ (||∇D(x̂)||₂ - 1)² ]   where  x̂ = ε·real + (1-ε)·fake
# =============================================================================

def gradient_penalty(critic, real, fake):
    batch_size = real.size(0)

    # ε ∈ [0,1] sampled per image in the batch
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)

    # Interpolated images between real and fake manifolds
    interpolated = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)

    # Critic score at interpolated points
    mixed_scores = critic(interpolated)

    # Gradients of critic output w.r.t. interpolated inputs
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,   # Must be True — GP itself needs to be differentiated
        retain_graph=True
    )[0]

    # L2 norm per sample, then penalise deviation from 1
    gradients  = gradients.view(batch_size, -1)
    grad_norm  = gradients.norm(2, dim=1)
    gp         = torch.mean((grad_norm - 1.0) ** 2)
    return gp

# =============================================================================
# 7. MODEL + OPTIMISER SETUP
# =============================================================================

gen    = Generator().to(device)
critic = Critic().to(device)

opt_gen    = optim.Adam(gen.parameters(),    lr=LR_G, betas=BETAS)
opt_critic = optim.Adam(critic.parameters(), lr=LR_D, betas=BETAS)

n_gen    = sum(p.numel() for p in gen.parameters()    if p.requires_grad)
n_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
print(f"\nGenerator params : {n_gen:,}")
print(f"Critic params    : {n_critic:,}")

os.makedirs("generated", exist_ok=True)

# Fixed noise vector — same 16 seeds every save for consistent visual comparison
fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=device)

# =============================================================================
# 8. TRAINING LOOP
#
# Each iteration:
#   1. Train critic CRITIC_ITERS times with gradient penalty
#   2. Train generator once
#   3. Log losses every epoch
#   4. Save sample grid every SAVE_EVERY epochs
# =============================================================================

print("\nStarting training...\n")

for epoch in range(1, NUM_EPOCHS + 1):

    gen.train()
    critic.train()

    for batch_idx, (real, _) in enumerate(loader):
        real      = real.to(device)
        cur_batch = real.size(0)

        # ------------------------------------------------------------------
        # Step 1: Train Critic
        #
        # Critic loss = E[D(fake)] - E[D(real)] + λ·GP
        # Minimising this → maximises Wasserstein distance estimate
        # ------------------------------------------------------------------
        for _ in range(CRITIC_ITERS):
            noise = torch.randn(cur_batch, Z_DIM, 1, 1, device=device)
            fake  = gen(noise).detach()  # Detach: skip generator graph

            score_real = critic(real)
            score_fake = critic(fake)
            gp         = gradient_penalty(critic, real, fake)

            loss_critic = torch.mean(score_fake) - torch.mean(score_real) + LAMBDA_GP * gp

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # ------------------------------------------------------------------
        # Step 2: Train Generator
        #
        # Generator loss = -E[D(G(z))]
        # Generator wants the critic to score its outputs as HIGH as possible
        # ------------------------------------------------------------------
        noise        = torch.randn(cur_batch, Z_DIM, 1, 1, device=device)
        fake         = gen(noise)
        score_fake_g = critic(fake)
        loss_gen     = -torch.mean(score_fake_g)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # ----------------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------------
    print(
        f"Epoch [{epoch:>3}/{NUM_EPOCHS}] | "
        f"Critic Loss: {loss_critic.item():>9.4f} | "
        f"Gen Loss: {loss_gen.item():>9.4f}"
    )

    # ----------------------------------------------------------------------
    # Save generated image grid using fixed noise
    # ----------------------------------------------------------------------
    if epoch % SAVE_EVERY == 0:
        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_noise)
        save_image(
            samples,
            f"generated/epoch_{epoch:03d}.png",
            normalize=True,  # Rescale [-1,1] → [0,1] for saving
            nrow=4
        )
        gen.train()
        print(f"  → Saved generated/epoch_{epoch:03d}.png")

print("\nTraining Finished ✓\n")

# =============================================================================
# 9. EVALUATION
#
# Uses the HELD-OUT eval split (not seen during training) for a clean test.
#
# Metrics:
#   Mean real score  — should be positive / higher
#   Mean fake score  — should be negative / lower
#   Separation       — larger gap = stronger critic
#
# Interpretation guide:
#   Separation > 50   → Strong critic
#   Separation 20–50  → Moderate
#   Separation < 20   → Weak (more training needed)
# =============================================================================

print("=" * 60)
print("EVALUATION: Critic Discriminative Ability")
print("(Using held-out eval split — not seen during training)")
print("=" * 60)

critic.eval()
gen.eval()

all_real_scores = []
all_fake_scores = []

with torch.no_grad():
    for real_batch, _ in eval_loader:
        real_batch = real_batch.to(device)
        cur        = real_batch.size(0)

        # Real scores
        r_scores = critic(real_batch)
        all_real_scores.append(r_scores)

        # Fake scores
        noise      = torch.randn(cur, Z_DIM, 1, 1, device=device)
        fake_batch = gen(noise)
        f_scores   = critic(fake_batch)
        all_fake_scores.append(f_scores)

mean_real = torch.cat(all_real_scores).mean().item()
mean_fake = torch.cat(all_fake_scores).mean().item()
sep       = mean_real - mean_fake

print(f"\nMean Critic Score — Real images : {mean_real:+.4f}")
print(f"Mean Critic Score — Fake images : {mean_fake:+.4f}")
print(f"Separation (real - fake)        : {sep:+.4f}")
print()

if sep > 50:
    verdict = "Strong ✓ — Critic clearly distinguishes real from fake."
elif sep > 20:
    verdict = "Moderate — Consider training longer or increasing CRITIC_ITERS."
else:
    verdict = "Weak — Try more epochs, larger FEATURES_D, or higher CRITIC_ITERS."

print(f"Assessment: {verdict}")
print("=" * 60)
