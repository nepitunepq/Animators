# train_lora.py
# รันบน Colab A100 หลัง dataset พร้อม

import os
import json
import subprocess
from pathlib import Path

from config import (
    MODEL_DIR,
    DATASET_DIR,
    LORA_OUTPUT,
    LORA_RANK,
    LORA_ALPHA,
    LEARNING_RATE,
    MAX_STEPS,
    BATCH_SIZE,
    GRAD_ACCUM,
    SAVE_EVERY,
    SEED,
    TARGET_W,
    TARGET_H,
    NUM_FRAMES,
)

CAPTION_DIR = os.path.join(DATASET_DIR, "captions")
VIDEO_DIR   = os.path.join(DATASET_DIR, "videos")

def check_dataset():
    """เช็คว่า dataset พร้อมก่อน train"""
    videos   = list(Path(VIDEO_DIR).glob("*.mp4"))
    captions = list(Path(CAPTION_DIR).glob("*.txt"))

    print("=" * 40)
    print("  Dataset Check")
    print("=" * 40)
    print(f"  Videos   : {len(videos)}")
    print(f"  Captions : {len(captions)}")

    if len(videos) == 0:
        print("  ERROR: ไม่มี videos — รอ Yoshi ส่ง clips มาก่อน")
        return False
    if len(videos) != len(captions):
        print(f"  WARNING: videos ({len(videos)}) != captions ({len(captions)})")
        return False

    print("  Dataset OK ✓")
    print("=" * 40)
    return True

def save_config():
    """Save config เป็น JSON"""
    config = {
        "model_dir"   : MODEL_DIR,
        "dataset_dir" : VIDEO_DIR,
        "caption_dir" : CAPTION_DIR,
        "output_dir"  : LORA_OUTPUT,
        "lora_rank"   : LORA_RANK,
        "lora_alpha"  : LORA_ALPHA,
        "lr"          : LEARNING_RATE,
        "max_steps"   : MAX_STEPS,
        "batch_size"  : BATCH_SIZE,
        "grad_accum"  : GRAD_ACCUM,
        "save_every"  : SAVE_EVERY,
        "seed"        : SEED,
        "resolution"  : [TARGET_W, TARGET_H],
        "num_frames"  : NUM_FRAMES,
    }
    Path("lora_config.json").write_text(json.dumps(config, indent=2))
    print("Config saved to lora_config.json ✓")
    return config

def setup_trainer():
    """Clone diffusion-pipe สำหรับ training"""
    if not Path("diffusion-pipe").exists():
        print("Cloning diffusion-pipe...")
        subprocess.run([
            "git", "clone", "--recurse-submodules",
            "https://github.com/tdrussell/diffusion-pipe",
            "diffusion-pipe"
        ], check=True)
        subprocess.run([
            "pip", "install", "-r",
            "diffusion-pipe/requirements.txt", "-q"
        ], check=True)
        print("diffusion-pipe ready ✓")
    else:
        print("diffusion-pipe already exists ✓")

def download_model():
    """Download Wan 2.2 จาก HuggingFace"""
    if Path(MODEL_DIR).exists():
        print(f"Model already exists: {MODEL_DIR} ✓")
        return

    print("Downloading Wan 2.2 I2V 480P...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Wan-AI/Wan2.2-I2V-14B-480P",
        local_dir=MODEL_DIR,
        ignore_patterns=["*.bin"],
    )
    print("Download complete ✓")

def run_training(config):
    """Run LoRA training"""
    Path(LORA_OUTPUT).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "diffusion-pipe/train.py",
        "--model_dir",   config["model_dir"],
        "--dataset_dir", config["dataset_dir"],
        "--caption_dir", config["caption_dir"],
        "--output_dir",  config["output_dir"],
        "--lora_rank",   str(config["lora_rank"]),
        "--lora_alpha",  str(config["lora_alpha"]),
        "--lr",          str(config["lr"]),
        "--steps",       str(config["max_steps"]),
        "--batch_size",  str(config["batch_size"]),
        "--grad_accum",  str(config["grad_accum"]),
        "--save_every",  str(config["save_every"]),
        "--seed",        str(config["seed"]),
        "--resolution",  f"{TARGET_W}x{TARGET_H}",
        "--num_frames",  str(config["num_frames"]),
    ]

    print("Starting LoRA training...")
    print(f"  Steps     : {config['max_steps']}")
    print(f"  LoRA rank : {config['lora_rank']}")
    print(f"  LR        : {config['lr']}")
    print("  (ใช้เวลาประมาณ 3-5 ชั่วโมงบน A100)")
    subprocess.run(cmd, check=True)

def monitor_loss():
    """Plot loss curve"""
    import glob
    import matplotlib.pyplot as plt

    log_files = sorted(glob.glob(f"{LORA_OUTPUT}/logs/*.jsonl"))
    if not log_files:
        print("No logs yet — training ยังไม่เริ่ม")
        return

    steps, losses = [], []
    with open(log_files[-1]) as f:
        for line in f:
            d = json.loads(line)
            if "train_loss" in d:
                steps.append(d["step"])
                losses.append(d["train_loss"])

    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses, color="purple", label="Train loss")
    plt.axhline(0.02, color="green", linestyle="--", label="Target < 0.02")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("LoRA Training Loss — Stop if loss stops decreasing")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()
    print(f"Latest loss : {losses[-1]:.4f}")
    print("Saved: loss_curve.png ✓")

if __name__ == "__main__":
    print("=" * 40)
    print("  Daisy — LoRA Training Pipeline")
    print("=" * 40)

    if not check_dataset():
        print("หยุดก่อน — รอ clips จาก Yoshi แล้วรัน dataset_prep.py ก่อน")
        exit()

    config = save_config()
    setup_trainer()
    download_model()
    run_training(config)
    monitor_loss()

    print("\nTraining complete!")
    print(f"LoRA weights saved to: {LORA_OUTPUT}/")