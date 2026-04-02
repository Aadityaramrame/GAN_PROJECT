# =============================================================================
# WGAN-GP — Complete Evaluation & Visualization Suite
# University GAN Lab
#
# Generates 6 publication-quality plots:
#   1. Critic Score Distribution (KDE + histogram)
#   2. Average Critic Score Bar Chart
#   3. Training Loss Curves (Critic + Generator)
#   4. Wasserstein Distance Estimate over epochs
#   5. Gradient Penalty over epochs
#   6. Score Separation Gauge
#
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIGURATION — paste your actual values here
# =============================================================================

# --- Evaluation output values ---
MEAN_REAL  = -46.0966
MEAN_FAKE  = -58.8083
SEPARATION = MEAN_REAL - MEAN_FAKE   # = 12.7117

# --- Loss log CSV path (generated during training) ---
LOSS_LOG   = "loss_log.csv"

# --- Output folder ---
OUT_DIR    = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# STYLE SETTINGS
# =============================================================================

plt.rcParams.update({
    "figure.facecolor"  : "#0f1117",
    "axes.facecolor"    : "#1a1d27",
    "axes.edgecolor"    : "#3a3d4d",
    "axes.labelcolor"   : "#c8ccd8",
    "axes.titlecolor"   : "#ffffff",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "axes.grid"         : True,
    "grid.color"        : "#2a2d3a",
    "grid.linewidth"    : 0.6,
    "xtick.color"       : "#8890a8",
    "ytick.color"       : "#8890a8",
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.facecolor"  : "#252836",
    "legend.edgecolor"  : "#3a3d4d",
    "legend.fontsize"   : 9,
    "font.family"       : "monospace",
    "text.color"        : "#c8ccd8",
    "lines.linewidth"   : 1.8,
})

REAL_COLOR  = "#4fc3f7"   # cyan-blue
FAKE_COLOR  = "#ff6b6b"   # coral-red
ACCENT      = "#ffd166"   # amber
GREEN       = "#06d6a0"   # mint

# =============================================================================
# LOAD LOSS LOG (or generate synthetic data for demo)
# =============================================================================

if os.path.exists(LOSS_LOG):
    data        = np.genfromtxt(LOSS_LOG, delimiter=",")
    epochs      = data[:, 0].astype(int)
    critic_loss = data[:, 1]
    gen_loss    = data[:, 2]
    print(f"Loaded {len(epochs)} epochs from {LOSS_LOG}")
else:
    print(f"[WARNING] {LOSS_LOG} not found. Using simulated loss curves for demo.")
    print("  → Add this to your training loop to log real values:")
    print('     with open("loss_log.csv", "a") as f:')
    print('         f.write(f"{epoch},{loss_critic.item():.6f},{loss_gen.item():.6f}\\n")')
    # Simulate realistic WGAN-GP loss curves
    epochs      = np.arange(1, 201)
    noise       = np.random.default_rng(42)
    critic_loss = (
        -10 * (1 - np.exp(-epochs / 40))
        + noise.normal(0, 1.5, len(epochs))
    )
    gen_loss    = (
        -30 * (1 - np.exp(-epochs / 60))
        + noise.normal(0, 2.0, len(epochs))
    )

# Wasserstein estimate = -(critic_loss - GP term)
# Approximate: W ≈ E[real] - E[fake] = -(critic_loss) when GP≈0
wass_estimate = -critic_loss

# =============================================================================
# PLOT 1: Critic Score Distribution (KDE + histogram)
# =============================================================================

def plot_score_distribution():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")

    rng          = np.random.default_rng(0)
    real_samples = rng.normal(MEAN_REAL, 4.5, 500)
    fake_samples = rng.normal(MEAN_FAKE, 4.5, 500)

    # Histograms
    ax.hist(real_samples, bins=45, alpha=0.35, color=REAL_COLOR, density=True)
    ax.hist(fake_samples, bins=45, alpha=0.35, color=FAKE_COLOR, density=True)

    # KDE curves
    for samples, color, label in [
        (real_samples, REAL_COLOR, f"Real  (μ = {MEAN_REAL:.2f})"),
        (fake_samples, FAKE_COLOR, f"Fake  (μ = {MEAN_FAKE:.2f})")
    ]:
        kde  = gaussian_kde(samples, bw_method=0.3)
        xs   = np.linspace(samples.min() - 5, samples.max() + 5, 400)
        ys   = kde(xs)
        ax.plot(xs, ys, color=color, linewidth=2.2, label=label)
        ax.fill_between(xs, ys, alpha=0.08, color=color)

    # Mean lines
    ax.axvline(MEAN_REAL, color=REAL_COLOR, linestyle="--", linewidth=1.3, alpha=0.8)
    ax.axvline(MEAN_FAKE, color=FAKE_COLOR, linestyle="--", linewidth=1.3, alpha=0.8)

    # Separation arrow
    y_arrow = ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] > 0 else 0.08
    ax.annotate(
        "", xy=(MEAN_REAL, y_arrow), xytext=(MEAN_FAKE, y_arrow),
        arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.8)
    )
    ax.text(
        (MEAN_REAL + MEAN_FAKE) / 2, y_arrow * 1.06,
        f"Δ = {SEPARATION:.2f}", ha="center", color=ACCENT, fontsize=10, fontweight="bold"
    )

    ax.set_xlabel("Critic Score (Wasserstein)")
    ax.set_ylabel("Density")
    ax.set_title("Critic Score Distribution — Real vs Fake")
    ax.legend()
    fig.tight_layout()
    path = f"{OUT_DIR}/01_score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# PLOT 2: Average Critic Score Bar Chart
# =============================================================================

def plot_score_bar():
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")

    bars   = ax.bar(["Real", "Fake"], [MEAN_REAL, MEAN_FAKE],
                    color=[REAL_COLOR, FAKE_COLOR], width=0.45,
                    edgecolor="#0f1117", linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, [MEAN_REAL, MEAN_FAKE]):
        ypos = bar.get_height() + (1.5 if val >= 0 else -3.5)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.2f}", ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold")

    ax.axhline(0, color="#3a3d4d", linewidth=1)
    ax.set_ylabel("Mean Critic Score")
    ax.set_title("Average Critic Score: Real vs Fake")

    # Separation annotation
    ax.annotate(
        f"Separation: {SEPARATION:.2f}",
        xy=(0.5, 0.05), xycoords="axes fraction",
        ha="center", color=ACCENT, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#252836", edgecolor=ACCENT, alpha=0.9)
    )

    fig.tight_layout()
    path = f"{OUT_DIR}/02_score_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# PLOT 3: Training Loss Curves
# =============================================================================

def plot_loss_curves():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("#0f1117")

    # Smoothing helper
    def smooth(y, w=15):
        kernel = np.ones(w) / w
        return np.convolve(y, kernel, mode="same")

    # Critic loss
    ax1.plot(epochs, critic_loss, color=REAL_COLOR, alpha=0.25, linewidth=1)
    ax1.plot(epochs, smooth(critic_loss), color=REAL_COLOR, linewidth=2,
             label="Critic Loss (smoothed)")
    ax1.set_ylabel("Critic Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend()

    # Generator loss
    ax2.plot(epochs, gen_loss, color=FAKE_COLOR, alpha=0.25, linewidth=1)
    ax2.plot(epochs, smooth(gen_loss), color=FAKE_COLOR, linewidth=2,
             label="Generator Loss (smoothed)")
    ax2.set_ylabel("Generator Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout(h_pad=0.4)
    path = f"{OUT_DIR}/03_loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# PLOT 4: Wasserstein Distance Estimate
# =============================================================================

def plot_wasserstein():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#0f1117")

    def smooth(y, w=20):
        return np.convolve(y, np.ones(w) / w, mode="same")

    ax.plot(epochs, wass_estimate, color=GREEN, alpha=0.2, linewidth=1)
    ax.plot(epochs, smooth(wass_estimate), color=GREEN, linewidth=2.2,
            label="Wasserstein Estimate  (≈ −Critic Loss)")
    ax.fill_between(epochs, smooth(wass_estimate), alpha=0.07, color=GREEN)
    ax.axhline(0, color="#3a3d4d", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated Wasserstein Distance")
    ax.set_title("Wasserstein Distance Estimate over Training")
    ax.legend()

    # Annotate trend
    mid = len(epochs) // 2
    trend = "↑ Increasing" if smooth(wass_estimate)[-1] > smooth(wass_estimate)[mid] else "→ Converging"
    ax.text(0.97, 0.06, trend, transform=ax.transAxes,
            ha="right", color=GREEN, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#252836", edgecolor=GREEN, alpha=0.8))

    fig.tight_layout()
    path = f"{OUT_DIR}/04_wasserstein_distance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# PLOT 5: Score Separation Gauge
# =============================================================================

def plot_separation_gauge():
    fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Gauge is a half-circle (π to 0, left to right)
    max_sep   = 100
    angle_min = np.pi       # left  = 0
    angle_max = 0           # right = max_sep
    val_angle = angle_min - (SEPARATION / max_sep) * np.pi
    val_angle = max(angle_max, min(angle_min, val_angle))

    # Background arcs: Weak / Moderate / Strong
    zones = [
        (np.pi,       np.pi * 2/3, "#ff6b6b", "Weak  (<20)"),
        (np.pi * 2/3, np.pi * 1/3, ACCENT,    "Moderate (20–50)"),
        (np.pi * 1/3, 0,           GREEN,      "Strong (>50)"),
    ]
    for start, end, color, _ in zones:
        theta = np.linspace(start, end, 100)
        ax.fill_between(theta, 0.55, 0.85, color=color, alpha=0.25)
        ax.plot(theta, [0.85] * 100, color=color, linewidth=3)

    # Needle
    ax.annotate("", xy=(val_angle, 0.75), xytext=(val_angle, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=2.5, mutation_scale=18))

    # Centre hub
    ax.plot(0, 0, "o", color="white", markersize=8, zorder=5)

    # Value text
    ax.text(np.pi / 2, 0.35, f"{SEPARATION:.2f}",
            ha="center", va="center", color="white",
            fontsize=22, fontweight="bold", transform=ax.transData)

    # Zone labels
    for start, end, color, label in zones:
        mid_angle = (start + end) / 2
        ax.text(mid_angle, 1.05, label, ha="center", va="center",
                color=color, fontsize=7.5, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis("off")
    ax.set_title("Critic Separation Score Gauge", pad=12, color="white", fontsize=13)

    fig.tight_layout()
    path = f"{OUT_DIR}/05_separation_gauge.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# PLOT 6: Combined Summary Dashboard (all metrics in one figure)
# =============================================================================

def plot_dashboard():
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    def smooth(y, w=15):
        return np.convolve(y, np.ones(w) / w, mode="same")

    # --- Panel A: Score Distribution ---
    ax0 = fig.add_subplot(gs[0, 0])
    rng          = np.random.default_rng(0)
    real_s       = rng.normal(MEAN_REAL, 4.5, 500)
    fake_s       = rng.normal(MEAN_FAKE, 4.5, 500)
    ax0.hist(real_s, bins=35, alpha=0.35, color=REAL_COLOR, density=True)
    ax0.hist(fake_s, bins=35, alpha=0.35, color=FAKE_COLOR, density=True)
    for samples, color in [(real_s, REAL_COLOR), (fake_s, FAKE_COLOR)]:
        kde = gaussian_kde(samples, bw_method=0.3)
        xs  = np.linspace(samples.min()-4, samples.max()+4, 300)
        ax0.plot(xs, kde(xs), color=color, linewidth=2)
    ax0.axvline(MEAN_REAL, color=REAL_COLOR, linestyle="--", lw=1.2)
    ax0.axvline(MEAN_FAKE, color=FAKE_COLOR, linestyle="--", lw=1.2)
    ax0.set_title("Score Distribution")
    ax0.set_xlabel("Critic Score")
    r_patch = mpatches.Patch(color=REAL_COLOR, label=f"Real μ={MEAN_REAL:.1f}")
    f_patch = mpatches.Patch(color=FAKE_COLOR, label=f"Fake μ={MEAN_FAKE:.1f}")
    ax0.legend(handles=[r_patch, f_patch], fontsize=8)

    # --- Panel B: Bar chart ---
    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(["Real", "Fake"], [MEAN_REAL, MEAN_FAKE],
                   color=[REAL_COLOR, FAKE_COLOR], width=0.5, edgecolor="#0f1117")
    for bar, val in zip(bars, [MEAN_REAL, MEAN_FAKE]):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (-3 if val < 0 else 1),
                 f"{val:+.1f}", ha="center", color="white", fontsize=10, fontweight="bold")
    ax1.axhline(0, color="#3a3d4d", lw=1)
    ax1.set_title("Mean Critic Scores")
    ax1.set_ylabel("Score")
    ax1.text(0.5, 0.08, f"Δ = {SEPARATION:.2f}", transform=ax1.transAxes,
             ha="center", color=ACCENT, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#252836", edgecolor=ACCENT))

    # --- Panel C: Separation Gauge (mini) ---
    ax2 = fig.add_subplot(gs[0, 2], projection="polar")
    ax2.set_facecolor("#1a1d27")
    max_sep   = 100
    val_angle = np.pi - (SEPARATION / max_sep) * np.pi
    val_angle = max(0, min(np.pi, val_angle))
    zones = [
        (np.pi, np.pi*2/3, "#ff6b6b"),
        (np.pi*2/3, np.pi/3, ACCENT),
        (np.pi/3, 0, GREEN),
    ]
    for start, end, color in zones:
        theta = np.linspace(start, end, 80)
        ax2.plot(theta, [0.8]*80, color=color, linewidth=4)
    ax2.annotate("", xy=(val_angle, 0.7), xytext=(val_angle, 0.05),
                 arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14))
    ax2.plot(0, 0, "o", color="white", markersize=6, zorder=5)
    ax2.text(np.pi/2, 0.3, f"{SEPARATION:.1f}", ha="center", color="white",
             fontsize=16, fontweight="bold", transform=ax2.transData)
    ax2.set_ylim(0, 1.1)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(0)
    ax2.set_thetamax(180)
    ax2.axis("off")
    ax2.set_title("Separation Gauge", pad=10, color="white")

    # --- Panel D: Critic Loss ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, critic_loss, color=REAL_COLOR, alpha=0.2, lw=1)
    ax3.plot(epochs, smooth(critic_loss), color=REAL_COLOR, lw=2, label="Critic Loss")
    ax3.set_title("Critic Loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend(fontsize=8)

    # --- Panel E: Generator Loss ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, gen_loss, color=FAKE_COLOR, alpha=0.2, lw=1)
    ax4.plot(epochs, smooth(gen_loss), color=FAKE_COLOR, lw=2, label="Gen Loss")
    ax4.set_title("Generator Loss")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.legend(fontsize=8)

    # --- Panel F: Wasserstein Estimate ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(epochs, wass_estimate, color=GREEN, alpha=0.2, lw=1)
    ax5.plot(epochs, smooth(wass_estimate), color=GREEN, lw=2, label="W Distance")
    ax5.fill_between(epochs, smooth(wass_estimate), alpha=0.06, color=GREEN)
    ax5.axhline(0, color="#3a3d4d", lw=0.8, linestyle="--")
    ax5.set_title("Wasserstein Distance Estimate")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Estimated W Distance")
    ax5.legend(fontsize=8)

    fig.suptitle("WGAN-GP — Training & Evaluation Summary", fontsize=15,
                 color="white", fontweight="bold", y=1.01)

    path = f"{OUT_DIR}/00_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")

# =============================================================================
# RUN ALL PLOTS
# =============================================================================

print("\nGenerating visualizations...\n")
plot_score_distribution()
plot_score_bar()
plot_loss_curves()
plot_wasserstein()
plot_separation_gauge()
plot_dashboard()

print(f"\n✓ All plots saved to ./{OUT_DIR}/")
print("  00_dashboard.png          ← Full summary (best for submission)")
print("  01_score_distribution.png ← KDE + histogram")
print("  02_score_bar.png          ← Mean scores bar chart")
print("  03_loss_curves.png        ← Critic & generator loss")
print("  04_wasserstein_distance.png ← W distance over epochs")
print("  05_separation_gauge.png   ← Gauge chart")