# =============================================================================
# DCGAN: Deep Convolutional Generative Adversarial Network
# University GAN Lab — PyTorch Implementation
#
# Architecture:
#   Generator     : z (100) → 3x128x128  (ConvTranspose2d, BatchNorm, ReLU, Tanh)
#   Discriminator : 3x128x128 → scalar   (Conv2d, BatchNorm, LeakyReLU, Sigmoid)
#
# Loss: Binary Cross Entropy (BCE)
#   Discriminator : -[log(D(real)) + log(1 - D(fake))]
#   Generator     : -log(D(G(z)))   (non-saturating form)
#
# Reference: Radford et al., "Unsupervised Representation Learning with
#            Deep Convolutional Generative Adversarial Networks" (2016)
#
# ALL results auto-saved to results/ for easy visualization later.
# =============================================================================

import os
import csv
import json
import random
import warnings
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

Z_DIM        = 100       # Latent vector dimension
IMG_SIZE     = 128       # 128x128 output
IMG_CHANNELS = 3         # RGB
FEATURES_G   = 64        # Generator base feature maps
FEATURES_D   = 64        # Discriminator base feature maps

LR           = 1e-4      # Learning rate — matched to WGAN-GP generator LR
BETAS        = (0.0, 0.9)   # Adam betas — matched to WGAN-GP

BATCH_SIZE   = 64
NUM_EPOCHS   = 200          # Matched to WGAN-GP
SAVE_EVERY   = 5         # Save image grid every N epochs
NUM_WORKERS  = 4
EVAL_SPLIT   = 0.10      # 10% held out for evaluation

# Label smoothing — improves discriminator stability
REAL_LABEL   = 0.9       # Soft label for real (instead of 1.0)
FAKE_LABEL   = 0.0       # Label for fake

# =============================================================================
# 3. RESULTS DIRECTORY SETUP
#
# Everything is saved here for easy visualization later — no re-running needed.
#
# results/
#   images/          ← generated image grids every SAVE_EVERY epochs
#   loss_log.csv     ← epoch, D_loss, G_loss, D_real, D_fake per epoch
#   eval_log.csv     ← epoch, mean_real_score, mean_fake_score per eval
#   config.json      ← all hyperparameters saved for reference
#   checkpoints/     ← model weights saved every 25 epochs
# =============================================================================

RESULTS_DIR      = "results_dcgan"
IMG_DIR          = os.path.join(RESULTS_DIR, "images")
CKPT_DIR         = os.path.join(RESULTS_DIR, "checkpoints")
LOSS_LOG_PATH    = os.path.join(RESULTS_DIR, "loss_log.csv")
EVAL_LOG_PATH    = os.path.join(RESULTS_DIR, "eval_log.csv")
CONFIG_PATH      = os.path.join(RESULTS_DIR, "config.json")

for d in [RESULTS_DIR, IMG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# Save config
config = {
    "model"       : "DCGAN",
    "z_dim"       : Z_DIM,
    "img_size"    : IMG_SIZE,
    "features_g"  : FEATURES_G,
    "features_d"  : FEATURES_D,
    "lr"          : LR,
    "betas"       : list(BETAS),
    "batch_size"  : BATCH_SIZE,
    "num_epochs"  : NUM_EPOCHS,
    "real_label"  : REAL_LABEL,
    "fake_label"  : FAKE_LABEL,
    "seed"        : SEED,
    "device"      : str(device),
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

# Initialise CSV files with headers
with open(LOSS_LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "d_loss", "g_loss", "d_real_mean", "d_fake_mean"])

with open(EVAL_LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "mean_real_score", "mean_fake_score", "separation"])

print(f"Results will be saved to: {RESULTS_DIR}/")

# =============================================================================
# 4. DATASET
#
# Folder structure:
#   GAN/
#     Real_9/
#       aloo_gobi/
#       aloo_methi/
#       ...  (9 class folders, each with .jpg images)
#
# Class labels are IGNORED — GAN only uses raw pixel values.
# =============================================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Fix palette/RGBA images
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)              # Normalise to [-1, 1]
])

full_dataset = datasets.ImageFolder(root="./Real_9", transform=transform)

eval_size    = int(len(full_dataset) * EVAL_SPLIT)
train_size   = len(full_dataset) - eval_size
train_dataset, eval_dataset = random_split(
    full_dataset, [train_size, eval_size],
    generator=torch.Generator().manual_seed(SEED)
)

loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True, drop_last=True
)
eval_loader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=True, drop_last=True
)

print(f"Total images    : {len(full_dataset)}")
print(f"Training images : {train_size}")
print(f"Eval images     : {eval_size}")
print(f"Classes         : {full_dataset.classes}")

# =============================================================================
# 5. GENERATOR
#
# z (Z_DIM x 1 x 1) → 3 x 128 x 128
#
# Spatial progression: 1 → 4 → 8 → 16 → 32 → 64 → 128
# Each block: ConvTranspose2d → BatchNorm → ReLU
# Final layer: ConvTranspose2d → Tanh (no BatchNorm on output)
# =============================================================================

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, img_channels=IMG_CHANNELS, features_g=FEATURES_G):
        super(Generator, self).__init__()

        def up_block(in_c, out_c):
            """Upsample block: doubles spatial size"""
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.net = nn.Sequential(
            # Stem: z x 1x1 → fg*16 x 4x4
            nn.ConvTranspose2d(z_dim, features_g*16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(inplace=True),

            up_block(features_g*16, features_g*8),   # → fg*8  x 8x8
            up_block(features_g*8,  features_g*4),   # → fg*4  x 16x16
            up_block(features_g*4,  features_g*2),   # → fg*2  x 32x32
            up_block(features_g*2,  features_g),     # → fg    x 64x64

            # Output: fg x 64x64 → 3 x 128x128
            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        """DCGAN paper weight init: N(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, z):
        return self.net(z)

# =============================================================================
# 6. DISCRIMINATOR
#
# 3 x 128 x 128 → scalar probability [0, 1]
#
# Spatial progression: 128 → 64 → 32 → 16 → 8 → 4 → 1
# Each block: Conv2d → BatchNorm → LeakyReLU(0.2)
# Note: NO BatchNorm on first layer (input is raw pixels)
# Final layer: Conv2d → Sigmoid
# =============================================================================

class Discriminator(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, features_d=FEATURES_D):
        super(Discriminator, self).__init__()

        def down_block(in_c, out_c, use_bn=True):
            """Downsample block: halves spatial size"""
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            # First layer: no BatchNorm on raw input
            down_block(img_channels,   features_d,     use_bn=False),  # → fd   x 64x64
            down_block(features_d,     features_d*2),                  # → fd*2 x 32x32
            down_block(features_d*2,   features_d*4),                  # → fd*4 x 16x16
            down_block(features_d*4,   features_d*8),                  # → fd*8 x 8x8
            down_block(features_d*8,   features_d*16),                 # → fd*16 x 4x4

            # Output: fd*16 x 4x4 → 1 x 1x1
            nn.Conv2d(features_d*16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()   # Output: probability that input is real
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), 1)   # (batch, 1)

# =============================================================================
# 7. MODEL + LOSS + OPTIMISER
# =============================================================================

gen   = Generator().to(device)
disc  = Discriminator().to(device)

criterion = nn.BCELoss()

opt_gen  = optim.Adam(gen.parameters(),  lr=LR, betas=BETAS)
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=BETAS)

n_gen  = sum(p.numel() for p in gen.parameters()  if p.requires_grad)
n_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
print(f"\nGenerator params     : {n_gen:,}")
print(f"Discriminator params : {n_disc:,}")

# Fixed noise — same 16 seeds every save for consistent visual comparison
fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=device)

# =============================================================================
# 8. TRAINING LOOP
#
# Each iteration:
#   Step 1 — Train Discriminator:
#     a. Real batch  → D should output ~1  → loss_real = BCE(D(real), 1)
#     b. Fake batch  → D should output ~0  → loss_fake = BCE(D(fake), 0)
#     c. Total D loss = loss_real + loss_fake
#
#   Step 2 — Train Generator:
#     a. Generate fake batch
#     b. Pass through (frozen) D → want D(fake) ≈ 1  → loss_G = BCE(D(fake), 1)
#
# ALL per-epoch metrics saved to CSV automatically.
# =============================================================================

print("\nStarting training...\n")

for epoch in range(1, NUM_EPOCHS + 1):

    gen.train()
    disc.train()

    epoch_d_loss   = 0.0
    epoch_g_loss   = 0.0
    epoch_d_real   = 0.0
    epoch_d_fake   = 0.0
    num_batches    = 0

    for batch_idx, (real, _) in enumerate(loader):
        real      = real.to(device)
        cur_batch = real.size(0)

        # ------------------------------------------------------------------
        # Step 1: Train Discriminator
        # ------------------------------------------------------------------
        opt_disc.zero_grad()

        # 1a. Real images — label = REAL_LABEL (soft label 0.9)
        real_labels  = torch.full((cur_batch, 1), REAL_LABEL, device=device)
        d_real_out   = disc(real)
        loss_d_real  = criterion(d_real_out, real_labels)

        # 1b. Fake images — label = FAKE_LABEL (0.0)
        noise        = torch.randn(cur_batch, Z_DIM, 1, 1, device=device)
        fake         = gen(noise).detach()   # Detach: skip generator graph
        fake_labels  = torch.full((cur_batch, 1), FAKE_LABEL, device=device)
        d_fake_out   = disc(fake)
        loss_d_fake  = criterion(d_fake_out, fake_labels)

        # 1c. Combined discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_disc.step()

        # ------------------------------------------------------------------
        # Step 2: Train Generator
        # ------------------------------------------------------------------
        opt_gen.zero_grad()

        # Generator wants discriminator to believe fakes are real
        noise        = torch.randn(cur_batch, Z_DIM, 1, 1, device=device)
        fake         = gen(noise)
        gen_labels   = torch.full((cur_batch, 1), REAL_LABEL, device=device)  # Generator target = real
        d_fake_for_g = disc(fake)
        loss_g       = criterion(d_fake_for_g, gen_labels)

        loss_g.backward()
        opt_gen.step()

        # Accumulate batch metrics
        epoch_d_loss += loss_d.item()
        epoch_g_loss += loss_g.item()
        epoch_d_real += d_real_out.mean().item()
        epoch_d_fake += d_fake_out.mean().item()
        num_batches  += 1

    # ------------------------------------------------------------------
    # Per-epoch averages
    # ------------------------------------------------------------------
    avg_d_loss = epoch_d_loss / num_batches
    avg_g_loss = epoch_g_loss / num_batches
    avg_d_real = epoch_d_real / num_batches
    avg_d_fake = epoch_d_fake / num_batches

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print(
        f"Epoch [{epoch:>3}/{NUM_EPOCHS}] | "
        f"D Loss: {avg_d_loss:.4f} | "
        f"G Loss: {avg_g_loss:.4f} | "
        f"D(real): {avg_d_real:.4f} | "
        f"D(fake): {avg_d_fake:.4f}"
    )

    # ------------------------------------------------------------------
    # Save loss log to CSV  ← key for visualization
    # ------------------------------------------------------------------
    with open(LOSS_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch, avg_d_loss, avg_g_loss, avg_d_real, avg_d_fake])

    # ------------------------------------------------------------------
    # Save generated image grid
    # ------------------------------------------------------------------
    if epoch % SAVE_EVERY == 0:
        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_noise)
        save_image(
            samples,
            os.path.join(IMG_DIR, f"epoch_{epoch:03d}.png"),
            normalize=True,
            nrow=4
        )
        gen.train()
        print(f"  → Saved image grid: epoch_{epoch:03d}.png")

    # ------------------------------------------------------------------
    # Save model checkpoint every 25 epochs
    # ------------------------------------------------------------------
    if epoch % 25 == 0:
        torch.save({
            "epoch"          : epoch,
            "gen_state"      : gen.state_dict(),
            "disc_state"     : disc.state_dict(),
            "opt_gen_state"  : opt_gen.state_dict(),
            "opt_disc_state" : opt_disc.state_dict(),
        }, os.path.join(CKPT_DIR, f"checkpoint_epoch_{epoch:03d}.pt"))
        print(f"  → Saved checkpoint: epoch_{epoch:03d}.pt")

print("\nTraining Finished ✓\n")

# =============================================================================
# 9. EVALUATION
#
# Uses held-out eval split — never seen during training.
# Saves results to eval_log.csv for every evaluated epoch.
#
# D(real) → should be close to 1.0  (discriminator confident on real)
# D(fake) → should be close to 0.0  (discriminator confident on fake)
# Separation = D(real) - D(fake)    (larger = stronger discriminator)
# =============================================================================

print("=" * 65)
print("EVALUATION: Discriminator Ability on Held-out Split")
print("=" * 65)

disc.eval()
gen.eval()

all_real_scores = []
all_fake_scores = []

with torch.no_grad():
    for real_batch, _ in eval_loader:
        real_batch = real_batch.to(device)
        cur        = real_batch.size(0)

        real_scores = disc(real_batch)
        all_real_scores.append(real_scores)

        noise      = torch.randn(cur, Z_DIM, 1, 1, device=device)
        fake_batch = gen(noise)
        fake_scores = disc(fake_batch)
        all_fake_scores.append(fake_scores)

mean_real = torch.cat(all_real_scores).mean().item()
mean_fake = torch.cat(all_fake_scores).mean().item()
sep       = mean_real - mean_fake

# Save final eval result
with open(EVAL_LOG_PATH, "a", newline="") as f:
    csv.writer(f).writerow([NUM_EPOCHS, mean_real, mean_fake, sep])

print(f"\nMean D Score — Real images : {mean_real:.4f}  (ideal: ~0.9)")
print(f"Mean D Score — Fake images : {mean_fake:.4f}  (ideal: ~0.0)")
print(f"Separation (real - fake)   : {sep:.4f}")
print()

if sep > 0.6:
    verdict = "Strong ✓  — Discriminator clearly separates real from fake."
elif sep > 0.3:
    verdict = "Moderate  — Discriminator has learned but is not fully confident."
else:
    verdict = "Weak      — Try more epochs or increase FEATURES_D."

print(f"Assessment: {verdict}")
print("=" * 65)

print(f"""
All results saved to ./{RESULTS_DIR}/

  images/                     ← Generated grids every {SAVE_EVERY} epochs
  loss_log.csv                ← epoch, d_loss, g_loss, d_real, d_fake
  eval_log.csv                ← mean_real_score, mean_fake_score, separation
  config.json                 ← hyperparameters used
  checkpoints/                ← model weights every 25 epochs
""")
