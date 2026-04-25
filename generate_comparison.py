# generate_comparison.py
# รันบน Colab A100 หลัง training เสร็จ

import torch
import subprocess
from pathlib import Path

from config import (
    MODEL_DIR,
    LORA_OUTPUT,
    REFERENCE_IMAGE,
    FINAL_OUTPUT,
    PROMPT,
    NUM_FRAMES,
    GUIDANCE_SCALE,
    INFERENCE_STEPS,
    LORA_SCALE,
    SEED,
    VIDEO_A,
    VIDEO_B,
    VIDEO_C,
    COMPARISON_OUT,
)

from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

LORA_WEIGHTS = str(Path(LORA_OUTPUT) / f"checkpoint-{1500}")

Path(FINAL_OUTPUT).mkdir(parents=True, exist_ok=True)

def load_pipe(use_lora=False):
    print(f"Loading {'trained' if use_lora else 'vanilla'} model...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16
    )
    if use_lora:
        pipe.load_lora_weights(LORA_WEIGHTS)
        print(f"  LoRA loaded: {LORA_WEIGHTS}")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe

def generate(pipe, output_path):
    image = load_image(REFERENCE_IMAGE).resize((832, 480))
    print(f"Generating {Path(output_path).name}...")
    out = pipe(
        image=image,
        prompt=PROMPT,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=INFERENCE_STEPS,
        generator=torch.Generator().manual_seed(SEED),
    ).frames[0]
    export_to_video(out, output_path, fps=24)
    print(f"Saved: {output_path} ✓")
    return output_path

def make_comparison():
    """ต่อ 3 videos เป็น side-by-side"""
    print("Creating 3-screen comparison...")
    cmd = [
        "ffmpeg",
        "-i", VIDEO_A,
        "-i", VIDEO_B,
        "-i", VIDEO_C,
        "-filter_complex",
        (
            "[0:v]scale=640:480,"
            "drawtext=text='A\\: Math Ground Truth':"
            "fontsize=18:fontcolor=white:x=10:y=10:"
            "box=1:boxcolor=black@0.6[v0];"

            "[1:v]scale=640:480,"
            "drawtext=text='B\\: AI Trained (SLERP LoRA)':"
            "fontsize=18:fontcolor=white:x=10:y=10:"
            "box=1:boxcolor=black@0.6[v1];"

            "[2:v]scale=640:480,"
            "drawtext=text='C\\: AI Vanilla':"
            "fontsize=18:fontcolor=white:x=10:y=10:"
            "box=1:boxcolor=black@0.6[v2];"

            "[v0][v1][v2]hstack=inputs=3[out]"
        ),
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18",
        "-y", COMPARISON_OUT
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {COMPARISON_OUT} ✓")

if __name__ == "__main__":
    # Video B — trained LoRA
    pipe_B = load_pipe(use_lora=True)
    generate(pipe_B, VIDEO_B)
    del pipe_B

    # Video C — vanilla baseline
    pipe_C = load_pipe(use_lora=False)
    generate(pipe_C, VIDEO_C)
    del pipe_C

    # 3-screen comparison
    make_comparison()

    print("\nAll done!")
    print(f"  B : {VIDEO_B}")
    print(f"  C : {VIDEO_C}")
    print(f"  Comparison : {COMPARISON_OUT}")