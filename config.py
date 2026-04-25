# config.py
# ไฟล์ config กลาง — ทุกคนใช้ร่วมกัน
# แก้ที่นี่ที่เดียว ทุกไฟล์จะอัปเดตตาม

import os

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

# Input
BLENDER_CLIPS   = os.path.join(BASE_DIR, "blender_clips")      # Yoshi วาง clips ที่นี่
REFERENCE_IMAGE = os.path.join(BASE_DIR, "reference_object.jpg")  # รูป object สำหรับ gen

# Outputs
DATASET_DIR     = os.path.join(BASE_DIR, "wan-dataset")
LORA_OUTPUT     = os.path.join(BASE_DIR, "lora_output")
FINAL_OUTPUT    = os.path.join(BASE_DIR, "outputs")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "wan2.2")

# Nine's files
POSES_FILE      = os.path.join(BASE_DIR, "camera_poses.json")
MANIFEST_FILE   = os.path.join(BASE_DIR, "dataset_manifest.json")

# ─────────────────────────────────────────────
#  Camera / Math (Nine)
# ─────────────────────────────────────────────

CENTER          = [0.0, 0.0, 0.0]
N_FRAMES        = 120
ELEVATIONS      = [-20, 0, 20, 40]
AZIMUTHS        = [0, 45, 90, 135]
RADII           = [4, 5, 6]

# ─────────────────────────────────────────────
#  Dataset (Daisy)
# ─────────────────────────────────────────────

TARGET_W        = 832
TARGET_H        = 480
FPS             = 24
OBJECT_NAME     = "bunny"

# ─────────────────────────────────────────────
#  LoRA Training (Daisy)
# ─────────────────────────────────────────────

LORA_RANK       = 16
LORA_ALPHA      = 16
LEARNING_RATE   = 1e-4
MAX_STEPS       = 1500
BATCH_SIZE      = 1
GRAD_ACCUM      = 4
SAVE_EVERY      = 250
SEED            = 42
NUM_FRAMES      = 81       # 81 frames @ 24fps = ~3 วินาที

# ─────────────────────────────────────────────
#  Generation (Daisy)
# ─────────────────────────────────────────────

PROMPT = (
    "A 3D object rotating 360 degrees, "
    "smooth constant angular velocity orbital camera, "
    "mathematically perfect SLERP rotation, "
    "studio lighting, white background, no jitter, no shake"
)
GUIDANCE_SCALE      = 5.0
INFERENCE_STEPS     = 25
LORA_SCALE          = 0.85

# ─────────────────────────────────────────────
#  Evaluation (Rose)
# ─────────────────────────────────────────────

EVAL_OUTPUT     = os.path.join(BASE_DIR, "eval_results")

# Videos to compare
VIDEO_A         = os.path.join(FINAL_OUTPUT, "ground_truth_orbit.mp4")   # Yoshi (Blender)
VIDEO_B         = os.path.join(FINAL_OUTPUT, "ai_trained_output.mp4")    # Daisy (trained)
VIDEO_C         = os.path.join(FINAL_OUTPUT, "ai_vanilla_output.mp4")    # Daisy (vanilla)
COMPARISON_OUT  = os.path.join(FINAL_OUTPUT, "comparison_final.mp4")