# daisy/generate_comparison.py
# รันบน Colab A100 หลัง training เสร็จ

import torch
import subprocess
from pathlib import Path
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

# ===== CONFIG =====
MODEL_DIR    = "./models/wan2.2"
LORA_WEIGHTS = "./lora_output/checkpoint-1500"
INPUT_IMAGE  = "./shared/reference_object.jpg"  # รูป object เดียวกับ Blender
OUTPUT_DIR   = "./shared/outputs"
NUM_FRAMES   = 81    # ~3 วินาที @ 24fps
PROMPT = (
    "A 3D object rotating 360 degrees, "
    "smooth constant angular velocity orbital camera, "
    "mathematically perfect rotation, "
    "studio lighting, white background, no jitter"
)
# ==================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_pipe(use_lora=False):
    print(f"Loading {'trained' if use_lora else 'vanilla'} model...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16
    )
    if use_lora:
        pipe.load_lora_weights(LORA_WEIGHTS)
        print(f"  LoRA loaded from: {LORA_WEIGHTS}")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe

def generate(pipe, output_name, seed=42):
    image = load_image(INPUT_IMAGE).resize((832, 480))
    print(f"Generating {output_name}...")
    out = pipe(
        image=image,
        prompt=PROMPT,
        num_frames=NUM_FRAMES,
        guidance_scale=5.0,
        num_inference_steps=25,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    path = f"{OUTPUT_DIR}/{output_name}"
    export_to_video(out, path, fps=24)
    print(f"Saved: {path} ✓")
    return path

def make_comparison(path_a, path_b, path_c):
    """ต่อ 3 videos เป็น side-by-side"""
    out = f"{OUTPUT_DIR}/comparison_final.mp4"
    cmd = [
        "ffmpeg",
        "-i", path_a, "-i", path_b, "-i", path_c,
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
        "-y", out
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {out} ✓")

if __name__ == "__main__":
    # Video B — trained
    pipe_B = load_pipe(use_lora=True)
    path_B = generate(pipe_B, "ai_trained_output.mp4")
    del pipe_B

    # Video C — vanilla
    pipe_C = load_pipe(use_lora=False)
    path_C = generate(pipe_C, "ai_vanilla_output.mp4")
    del pipe_C

    # 3-screen comparison
    path_A = "./shared/blender_clips/ground_truth_orbit.mp4"
    make_comparison(path_A, path_B, path_C)
    print("\nAll done! Files ready:")
    print(f"  {path_B}")
    print(f"  {path_C}")
    print(f"  {OUTPUT_DIR}/comparison_final.mp4")