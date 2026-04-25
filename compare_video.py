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

DEFAULT_A   = "ground_truth_orbit.mp4"
DEFAULT_B   = "ai_trained_output.mp4"
DEFAULT_C   = "ai_vanilla_output.mp4"
DEFAULT_OUT = "comparison_final.mp4"

PANEL_W = 640
PANEL_H = 360

LABEL_FONT_SIZE = 28
LABEL_COLOR     = "white"
LABEL_BOX_COLOR = "0x000000@0.45"   # semi-transparent black backing
LABEL_X         = 10
LABEL_Y         = 10


# ---------------------------------------------------------------------------
# Build ffmpeg filter_complex string
# ---------------------------------------------------------------------------

def build_filter(label_a, label_b, label_c):
    """
    Scale each input to PANEL_W×PANEL_H, stamp a label, then hstack.
    ffmpeg stream indices: [0:v] = A, [1:v] = B, [2:v] = C
    """

    def panel(stream_idx, label, out_tag):
        # Escape colons so ffmpeg doesn't misparse the drawtext option string
        safe_label = label.replace(":", "\\:")
        return (
            f"[{stream_idx}:v]"
            f"scale={PANEL_W}:{PANEL_H}:force_original_aspect_ratio=decrease,"
            f"pad={PANEL_W}:{PANEL_H}:(ow-iw)/2:(oh-ih)/2,"
            f"drawtext="
            f"text='{safe_label}':"
            f"fontsize={LABEL_FONT_SIZE}:"
            f"fontcolor={LABEL_COLOR}:"
            f"box=1:boxcolor={LABEL_BOX_COLOR}:boxborderw=6:"
            f"x={LABEL_X}:y={LABEL_Y}"
            f"[{out_tag}]"
        )

    parts = [
        panel(0, label_a, "v0"),
        panel(1, label_b, "v1"),
        panel(2, label_c, "v2"),
        "[v0][v1][v2]hstack=inputs=3[out]",
    ]

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Run ffmpeg
# ---------------------------------------------------------------------------

def compare(path_a, path_b, path_c, output_path,
            label_a="A: Math Ground Truth",
            label_b="B: AI Trained",
            label_c="C: AI Vanilla"):

    for path, name in [(path_a, "A"), (path_b, "B"), (path_c, "C")]:
        if not os.path.exists(path):
            print(f"[compare_video] WARNING: input {name} not found at '{path}'")

    filter_complex = build_filter(label_a, label_b, label_c)

    cmd = [
        "ffmpeg", "-y",
        "-i", path_a,                    # input 0 = ground truth
        "-i", path_b,                    # input 1 = AI trained
        "-i", path_c,                    # input 2 = AI vanilla
        "-filter_complex", filter_complex,
        "-map", "[out]",                 # use the stacked video stream
        "-c:v", "libx264",               # H264 encode
        "-crf", "18",                    # high quality
        "-pix_fmt", "yuv420p",           # broad player compatibility
        "-movflags", "+faststart",       # web-friendly: moov atom at front
        output_path,
    ]

    print("Running ffmpeg …")
    print("  " + " ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[compare_video] ffmpeg exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nSaved: {output_path}")
    print(f"Panel size per clip: {PANEL_W}×{PANEL_H}  →  total: {PANEL_W*3}×{PANEL_H}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stack three MP4s side-by-side with labels."
    )
    parser.add_argument("--a",   default=DEFAULT_A,   help="Path to clip A (ground truth)")
    parser.add_argument("--b",   default=DEFAULT_B,   help="Path to clip B (AI trained)")
    parser.add_argument("--c",   default=DEFAULT_C,   help="Path to clip C (AI vanilla)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output MP4 path")
    parser.add_argument("--label-a", default="A: Math Ground Truth")
    parser.add_argument("--label-b", default="B: AI Trained")
    parser.add_argument("--label-c", default="C: AI Vanilla")
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
