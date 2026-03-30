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
warnings.filterwarnings("ignore", category=UserWarning)

import csv
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

Z_DIM        = 100
IMG_SIZE     = 128
IMG_CHANNELS = 3
FEATURES_G   = 64
FEATURES_D   = 64       # Keep at 64 — matches the run that gave separation 12.7

LR_G         = 1e-4
LR_D         = 4e-5     # Slightly slower critic LR (TTUR) — same as 12.7 run
BETAS        = (0.0, 0.9)

BATCH_SIZE   = 64
NUM_EPOCHS   = 200      # 200 was enough for 12.7 — no need for 300
CRITIC_ITERS = 7        # 7 gave better separation than 5
LAMBDA_GP    = 10
SAVE_EVERY   = 5
NUM_WORKERS  = 6
EVAL_SPLIT   = 0.10

# =============================================================================
# 3. DATASET
# =============================================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Only use the 5 selected classes — create Real_5 folder with just these
# OR use the subset approach below which filters from Real_9 directly
SELECTED_CLASSES = ["aloo_gobi", "aloo_methi", "aloo_mutter", "palak_paneer", "poha"]

full_dataset = datasets.ImageFolder(root="./Real_9", transform=transform)

# Filter dataset to only selected 5 classes
selected_indices = [
    i for i, (_, label) in enumerate(full_dataset.samples)
    if full_dataset.classes[label] in SELECTED_CLASSES
]
full_dataset = torch.utils.data.Subset(full_dataset, selected_indices)

# Patch .classes onto Subset for printing
full_dataset.classes = SELECTED_CLASSES

eval_size  = int(len(full_dataset) * EVAL_SPLIT)
train_size = len(full_dataset) - eval_size
train_dataset, eval_dataset = random_split(
    full_dataset,
    [train_size, eval_size],
    generator=torch.Generator().manual_seed(SEED)
)

loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
eval_loader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)

print(f"Total images    : {len(full_dataset)}")
print(f"Training images : {train_size}")
print(f"Eval images     : {eval_size}")
print(f"Classes         : {SELECTED_CLASSES}")

# =============================================================================
# 4. GENERATOR
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
            nn.ConvTranspose2d(z_dim, features_g*16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(inplace=True),

            up_block(features_g*16, features_g*8),
            up_block(features_g*8,  features_g*4),
            up_block(features_g*4,  features_g*2),
            up_block(features_g*2,  features_g),

            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

# =============================================================================
# 5. CRITIC
# =============================================================================

class Critic(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, features_d=FEATURES_D):
        super(Critic, self).__init__()

        def sn_conv(in_c, out_c, kernel=4, stride=2, pad=1):
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            sn_conv(img_channels,   features_d),
            sn_conv(features_d,     features_d*2),
            sn_conv(features_d*2,   features_d*4),
            sn_conv(features_d*4,   features_d*8),
            sn_conv(features_d*8,   features_d*16),

            nn.utils.spectral_norm(
                nn.Conv2d(features_d*16, features_d*16, kernel_size=3, stride=1, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(
                nn.Conv2d(features_d*16, features_d*8, kernel_size=3, stride=1, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(
                nn.Conv2d(features_d*8, 1, kernel_size=4, stride=1, padding=0, bias=False)
            )
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        return self.net(x).view(x.size(0), 1)

# =============================================================================
# 6. GRADIENT PENALTY
# =============================================================================

def gradient_penalty(critic, real, fake):
    batch_size   = real.size(0)
    epsilon      = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)
    mixed_scores = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return torch.mean((grad_norm - 1.0) ** 2)

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

os.makedirs("generated",    exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Initialise loss log CSV — so results are never lost even if terminal closes
LOSS_LOG = "loss_log.csv"
with open(LOSS_LOG, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "critic_loss", "gen_loss"])

fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=device)

# =============================================================================
# 8. TRAINING LOOP
# =============================================================================

print("\nStarting training...\n")

for epoch in range(1, NUM_EPOCHS + 1):

    gen.train()
    critic.train()

    for batch_idx, (real, _) in enumerate(loader):
        real      = real.to(device)
        cur_batch = real.size(0)

        # ---- Train Critic ----
        for _ in range(CRITIC_ITERS):
            noise      = torch.randn(cur_batch, Z_DIM, 1, 1, device=device)
            fake       = gen(noise).detach()
            score_real = critic(real)
            score_fake = critic(fake)
            gp         = gradient_penalty(critic, real, fake)
            loss_critic = torch.mean(score_fake) - torch.mean(score_real) + LAMBDA_GP * gp

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # ---- Train Generator ----
        noise        = torch.randn(cur_batch, Z_DIM, 1, 1, device=device)
        fake         = gen(noise)
        score_fake_g = critic(fake)
        loss_gen     = -torch.mean(score_fake_g)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    print(
        f"Epoch [{epoch:>3}/{NUM_EPOCHS}] | "
        f"Critic Loss: {loss_critic.item():>9.4f} | "
        f"Gen Loss: {loss_gen.item():>9.4f}"
    )

    # Save loss to CSV every epoch — use this for visualizations later
    with open(LOSS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([epoch, round(loss_critic.item(), 6), round(loss_gen.item(), 6)])

    # Save image grid
    if epoch % SAVE_EVERY == 0:
        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_noise)
        save_image(samples, f"generated/epoch_{epoch:03d}.png", normalize=True, nrow=4)
        gen.train()
        print(f"  → Saved generated/epoch_{epoch:03d}.png")

    # Save checkpoint every 25 epochs
    if epoch % 25 == 0:
        torch.save({
            "epoch"              : epoch,
            "gen_state_dict"     : gen.state_dict(),
            "critic_state_dict"  : critic.state_dict(),
            "opt_gen"            : opt_gen.state_dict(),
            "opt_critic"         : opt_critic.state_dict(),
            "z_dim"              : Z_DIM,
            "img_size"           : IMG_SIZE,
            "img_channels"       : IMG_CHANNELS,
            "features_g"         : FEATURES_G,
            "features_d"         : FEATURES_D,
        }, f"saved_models/checkpoint_epoch_{epoch:03d}.pt")
        print(f"  → Checkpoint saved: saved_models/checkpoint_epoch_{epoch:03d}.pt")

print("\nTraining Finished ✓\n")

# =============================================================================
# 9. SAVE FINAL MODEL
#    Saves everything needed to reload and run inference later.
#    mean_real and mean_fake are placeholders — updated after evaluation below.
# =============================================================================

def save_final_model(mean_real=None, mean_fake=None):
    torch.save({
        "gen_state_dict"    : gen.state_dict(),
        "critic_state_dict" : critic.state_dict(),
        "z_dim"             : Z_DIM,
        "img_size"          : IMG_SIZE,
        "img_channels"      : IMG_CHANNELS,
        "features_g"        : FEATURES_G,
        "features_d"        : FEATURES_D,
        "num_epochs"        : NUM_EPOCHS,
        # Threshold info — used by predict.py to classify images
        "mean_real_score"   : mean_real,
        "mean_fake_score"   : mean_fake,
        "threshold"         : ((mean_real + mean_fake) / 2) if (mean_real and mean_fake) else None,
    }, "saved_models/Trial4_GAN_Model_5class.pt")
    print("✓ Final model saved → saved_models/Trial4_GAN_Model_5class.pt")

# Save once now (without threshold — before eval)
save_final_model()

# =============================================================================
# 10. EVALUATION
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

        r_scores = critic(real_batch)
        all_real_scores.append(r_scores)

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

if sep > 50:
    verdict = "Strong ✓ — Critic clearly distinguishes real from fake."
elif sep > 20:
    verdict = "Moderate — Consider training longer or increasing CRITIC_ITERS."
else:
    verdict = "Weak — Try more epochs, larger FEATURES_D, or higher CRITIC_ITERS."

print(f"\nAssessment: {verdict}")
print("=" * 60)

# =============================================================================
# 11. RESAVE FINAL MODEL WITH THRESHOLD
#     Now that we have real eval scores, resave with the correct threshold
#     baked in — predict.py will use this automatically.
# =============================================================================

save_final_model(mean_real=mean_real, mean_fake=mean_fake)
print(f"\nThreshold saved in model: {(mean_real + mean_fake) / 2:.4f}")
print("Model: saved_models/Trial4_GAN_Model_5class.pt")
print("Run predict.py to classify new images using this model.\n")
