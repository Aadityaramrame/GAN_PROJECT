# =============================================================================
# WGAN-GP — Image Classifier (Real vs Fake)
# University GAN Lab
#
# USAGE:
#   python predict.py --image your_image.jpg
#   python predict.py --image your_image.jpg --model saved_models/wgan_gp_final.pt
#
# HOW IT WORKS:
#   The trained critic scores your image on the Wasserstein scale.
#   The threshold (midpoint of real/fake means from training eval) decides
#   whether your image is classified as REAL or FAKE.
# =============================================================================

import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# =============================================================================
# CRITIC ARCHITECTURE
# Must match exactly what was used in wgan_gp.py
# =============================================================================

class Critic(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
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

    def forward(self, x):
        return self.net(x).view(x.size(0), 1)

# =============================================================================
# LOAD MODEL FROM CHECKPOINT
# =============================================================================

def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"\n  ERROR: Model file not found at '{model_path}'")
        print("  Make sure you have run wgan_gp.py first to train and save the model.")
        exit(1)

    checkpoint = torch.load(model_path, map_location=device)

    features_d   = checkpoint.get("features_d",   64)
    img_channels = checkpoint.get("img_channels",  3)

    critic = Critic(img_channels=img_channels, features_d=features_d).to(device)
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic.eval()

    return critic, checkpoint

# =============================================================================
# PREPROCESS IMAGE
# Must match the same transform used in wgan_gp.py training
# =============================================================================

def preprocess_image(image_path, img_size=128):
    if not os.path.exists(image_path):
        print(f"\n  ERROR: Image not found at '{image_path}'")
        exit(1)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path)
    return transform(img).unsqueeze(0)   # → (1, 3, 128, 128)

# =============================================================================
# CLASSIFY IMAGE
# =============================================================================

def classify(image_path, model_path, device):

    print("\n" + "=" * 55)
    print("  WGAN-GP Image Classifier — Real vs Fake")
    print("=" * 55)

    # Load model
    print(f"\n  Loading model   : {model_path}")
    critic, checkpoint = load_model(model_path, device)

    num_epochs = checkpoint.get("num_epochs", "unknown")
    mean_real  = checkpoint.get("mean_real_score", None)
    mean_fake  = checkpoint.get("mean_fake_score", None)
    threshold  = checkpoint.get("threshold", None)
    img_size   = checkpoint.get("img_size", 128)

    print(f"  Trained epochs  : {num_epochs}")

    # Use saved threshold if available, otherwise fall back to 0
    if threshold is not None:
        print(f"  Mean real score : {mean_real:+.4f}  (from training eval)")
        print(f"  Mean fake score : {mean_fake:+.4f}  (from training eval)")
        print(f"  Threshold       : {threshold:+.4f}  (midpoint)")
    else:
        threshold = 0.0
        print(f"  Threshold       : {threshold}  (default — retrain to get calibrated threshold)")

    # Preprocess and score image
    print(f"\n  Image           : {image_path}")
    tensor = preprocess_image(image_path, img_size).to(device)

    with torch.no_grad():
        score = critic(tensor).item()

    # Classify
    verdict    = "REAL" if score >= threshold else "FAKE"
    margin     = abs(score - threshold)

    if margin < 3:
        confidence = "Low      (borderline case)"
    elif margin < 10:
        confidence = "Moderate"
    else:
        confidence = "High"

    # Print result
    print("\n" + "=" * 55)
    if verdict == "REAL":
        print("  VERDICT  :  ✅  REAL IMAGE")
    else:
        print("  VERDICT  :  ❌  FAKE / AI-GENERATED IMAGE")

    print(f"  Score    :  {score:+.4f}")
    print(f"  Threshold:  {threshold:+.4f}")
    print(f"  Margin   :  {margin:.4f}")
    print(f"  Confidence: {confidence}")
    print("=" * 55)

    # Detailed explanation
    print(f"""
  How this score was determined:
  ─────────────────────────────────────────────────────
  Your image scored     :  {score:+.4f}
  Real images average   :  {mean_real:+.4f}  (from held-out eval)
  Fake images average   :  {mean_fake:+.4f}  (from held-out eval)
  Decision boundary     :  {threshold:+.4f}  (midpoint)

  Score >= {threshold:+.4f}  →  classified as REAL
  Score <  {threshold:+.4f}  →  classified as FAKE

  Note: In WGAN-GP, scores are relative Wasserstein values.
  Negative scores are completely normal — only the position
  relative to the threshold matters, not the absolute value.
  ─────────────────────────────────────────────────────
    """)

    return {"verdict": verdict, "score": score, "confidence": confidence}

# =============================================================================
# MAIN — Interactive or Command Line
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WGAN-GP Real vs Fake Image Classifier"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file to classify (jpg, png, etc.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/Trial4_GAN_Model_5class.pt",
        help="Path to trained model (default: saved_models/wgan_gp_final.pt)"
    )

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If no --image argument given, ask the user interactively
    if args.image is None:
        print("\n  WGAN-GP Real vs Fake Classifier")
        print("  ─────────────────────────────────")
        image_path = input("  Enter image path: ").strip().strip('"').strip("'")
    else:
        image_path = args.image

    classify(
        image_path = image_path,
        model_path = args.model,
        device     = device
    )
