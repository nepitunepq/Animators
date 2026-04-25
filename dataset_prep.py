# dataset_prep.py
# รันหลังได้ clips จาก Yoshi
# ตอนนี้แค่เขียนให้เสร็จก่อน ยังไม่ต้องรัน

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

from config import (
    BLENDER_CLIPS,
    DATASET_DIR,
    TARGET_W,
    TARGET_H,
    OBJECT_NAME,
    FPS,
)

def resize_video(src, dst, w, h):
    """Resize video ให้ตรง resolution ที่ Wan 2.2 ต้องการ"""
    cmd = [
        "ffmpeg", "-i", src,
        "-vf", (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=white"
        ),
        "-c:v", "libx264", "-crf", "18",
        "-r", str(FPS), "-t", "5",
        "-y", dst
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.decode()}")
    else:
        print(f"  Resized: {Path(dst).name}")

def make_caption(filename):
    """สร้าง caption สำหรับแต่ละ clip"""
    parts = Path(filename).stem.split("_")
    elev  = next((p.replace("elev","") for p in parts if p.startswith("elev")), "20")
    return (
        f"A {OBJECT_NAME} rotating 360 degrees, "
        f"smooth constant angular velocity orbital camera, "
        f"camera elevation {elev} degrees, "
        f"mathematically perfect SLERP interpolation, "
        f"studio lighting, white background, no jitter, no shake."
    )

def flip_video(src, dst):
    """Flip horizontal เพื่อ double dataset"""
    cmd = [
        "ffmpeg", "-i", src,
        "-vf", "hflip",
        "-c:v", "libx264", "-crf", "18",
        "-y", dst
    ]
    subprocess.run(cmd, capture_output=True)

def prepare_dataset():
    videos_dir   = Path(DATASET_DIR) / "videos"
    captions_dir = Path(DATASET_DIR) / "captions"
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    clips = list(Path(BLENDER_CLIPS).glob("*.mp4"))

    if not clips:
        print("ยังไม่มี clips — รอ Yoshi ส่งมาก่อน")
        print(f"วาง clips ไว้ที่: {BLENDER_CLIPS}")
        return

    print(f"Found {len(clips)} clips — processing...")
    manifest = []

    for i, clip in enumerate(tqdm(clips, desc="Processing")):
        # 1. Resize
        out_v = videos_dir / clip.name
        resize_video(str(clip), str(out_v), TARGET_W, TARGET_H)

        # 2. Caption
        caption = make_caption(clip.name)
        (captions_dir / (clip.stem + ".txt")).write_text(caption)

        # 3. Flip augmentation
        flip_name = clip.stem + "_flip.mp4"
        flip_video(str(out_v), str(videos_dir / flip_name))
        (captions_dir / (clip.stem + "_flip.txt")).write_text(caption)

        manifest.append({
            "id"      : i,
            "file"    : clip.name,
            "flipped" : flip_name,
            "caption" : caption
        })

    # 4. Save manifest
    (Path(DATASET_DIR) / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    total = len(list(videos_dir.glob("*.mp4")))
    print(f"\nDone! {total} videos ready")
    print(f"Saved to: {DATASET_DIR}")

if __name__ == "__main__":
    prepare_dataset()