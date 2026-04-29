"""
compare_video.py
Combine three MP4 clips side-by-side with white text labels.

Usage:
    python compare_video.py \
        --a ground_truth_orbit.mp4 \
        --b ai_trained_output.mp4 \
        --c ai_vanilla_output.mp4 \
        --out comparison_final.mp4

All three clips are scaled to 640×360 before stacking, so mismatched
source resolutions are handled automatically.
"""

import argparse
import subprocess
import sys
import os


# ---------------------------------------------------------------------------
# Default file paths (edit here if you prefer not to use CLI flags)
# ---------------------------------------------------------------------------

DEFAULT_A   = "ground_truth.gif"
DEFAULT_B   = "vanilla.gif"
DEFAULT_C   = "slerp.gif"
DEFAULT_OUT = "comparison.gif"
PANEL_W     = 256
PANEL_H     = 256

# PANEL_W = 640
# PANEL_H = 360

LABEL_FONT_SIZE = 28
LABEL_COLOR     = "white"
LABEL_BOX_COLOR = "0x000000@0.45"   # semi-transparent black backing
LABEL_X         = 10
LABEL_Y         = 10


# ---------------------------------------------------------------------------
# Build ffmpeg filter_complex string
# ---------------------------------------------------------------------------

import subprocess, sys, os
from PIL import Image, ImageDraw, ImageFont

def add_label(frame, text, panel_w, panel_h, font_size=20):
    img = frame.copy().resize((panel_w, panel_h)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.rectangle([8, 8, 8+tw+8, 8+th+8], fill=(0,0,0,180))
    draw.text((12, 10), text, font=font, fill=(255,255,255))
    return img

def load_gif_frames(path):
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(gif.copy().convert("RGB"))
            gif.seek(gif.tell()+1)
    except EOFError:
        pass
    return frames

def compare(path_a, path_b, path_c, output_path,
            label_a="A: Ground Truth",
            label_b="B: Quaternion+SLERP",
            label_c="C: Vanilla"):

    fa = load_gif_frames(path_a)
    fb = load_gif_frames(path_b)
    fc = load_gif_frames(path_c)

    n = max(len(fa), len(fb), len(fc))
    fa = [fa[i % len(fa)] for i in range(n)]
    fb = [fb[i % len(fb)] for i in range(n)]
    fc = [fc[i % len(fc)] for i in range(n)]

    combined = []
    for a, b, c in zip(fa, fb, fc):
        a = add_label(a, label_a, PANEL_W, PANEL_H)
        b = add_label(b, label_b, PANEL_W, PANEL_H)
        c = add_label(c, label_c, PANEL_W, PANEL_H)
        out = Image.new("RGB", (PANEL_W*3, PANEL_H))
        out.paste(a, (0, 0))
        out.paste(b, (PANEL_W, 0))
        out.paste(c, (PANEL_W*2, 0))
        combined.append(out)

    combined[0].save(output_path, save_all=True,
                     append_images=combined[1:], loop=0, duration=125)
    print(f"✅ Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stack three MP4s side-by-side with labels."
    )
    parser.add_argument("--a",   default=DEFAULT_A,   help="Path to clip A (ground truth)")
    parser.add_argument("--b",   default=DEFAULT_B,   help="Path to clip B (AI trained)")
    parser.add_argument("--c",   default=DEFAULT_C,   help="Path to clip C (Vanilla)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output MP4 path")
    parser.add_argument("--label-a", default="A: Math Ground Truth")
    parser.add_argument("--label-b", default="B: Vanilla")
    parser.add_argument("--label-c", default="C: Quaternion+SLERP")
    args = parser.parse_args()

    compare(
        path_a=args.a,
        path_b=args.b,
        path_c=args.c,
        output_path=args.out,
        label_a=args.label_a,
        label_b=args.label_b,
        label_c=args.label_c,
    )


if __name__ == "__main__":
    main()
