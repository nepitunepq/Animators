# config.py
# ไฟล์ config กลาง — ทุกคนใช้ร่วมกัน
# แก้ที่นี่ที่เดียว ทุกไฟล์จะอัปเดตตาม

import os

# ─────────────────────────────────────────────
#  Base
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
#  Paths — Nine (Math)
# ─────────────────────────────────────────────

CAMERA_OUTPUT   = os.path.join(BASE_DIR, "camera_output")
POSES_FILE      = os.path.join(CAMERA_OUTPUT, "camera_poses.json")
MANIFEST_FILE   = os.path.join(CAMERA_OUTPUT, "dataset_manifest.json")

# ─────────────────────────────────────────────
#  Paths — Yoshi (Blender)
# ─────────────────────────────────────────────

RENDERS_DIR     = os.path.join(BASE_DIR, "renders")       # PNG frames จาก Blender
CLIPS_DIR       = os.path.join(BASE_DIR, "clips")         # MP4 clips หลัง ffmpeg
GROUND_TRUTH    = os.path.join(CLIPS_DIR, "ground_truth_orbit.mp4")

# Blender settings
BLENDER_OBJECT  = "Suzanne"   # monkey head — เปลี่ยนเป็น bunny ได้ถ้ามี .obj
BLENDER_FPS     = 24
BLENDER_RES_X   = 1920
BLENDER_RES_Y   = 1080

# ─────────────────────────────────────────────
#  Paths — Daisy (Dataset + Training)
# ─────────────────────────────────────────────

BLENDER_CLIPS   = os.path.join(BASE_DIR, "clips")         # รับ clips จาก Yoshi
DATASET_DIR     = os.path.join(BASE_DIR, "wan-dataset")
LORA_OUTPUT     = os.path.join(BASE_DIR, "lora_output")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "wan2.2")
REFERENCE_IMAGE = os.path.join(BASE_DIR, "camera_path.png")

# ─────────────────────────────────────────────
#  Paths — Output videos
# ─────────────────────────────────────────────

FINAL_OUTPUT    = os.path.join(BASE_DIR, "outputs")
VIDEO_A         = GROUND_TRUTH                                              # Blender
VIDEO_B         = os.path.join(FINAL_OUTPUT, "ai_trained_output.mp4")      # AI trained
VIDEO_C         = os.path.join(FINAL_OUTPUT, "ai_vanilla_output.mp4")      # AI vanilla
COMPARISON_OUT  = os.path.join(FINAL_OUTPUT, "comparison_final.mp4")

# ─────────────────────────────────────────────
#  Paths — Rose (Evaluation)
# ─────────────────────────────────────────────

EVAL_OUTPUT     = os.path.join(BASE_DIR, "eval_results")

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
BATCH_SIZE      = 1        # ต้องเป็น 1 เพราะ VRAM 8GB
GRAD_ACCUM      = 8        # เพิ่มจาก 4 → 8 เพื่อชดเชย batch size
SAVE_EVERY      = 250
SEED            = 42
NUM_FRAMES      = 49       # ลดจาก 81 → 49 ประหยัด VRAM
TARGET_W        = 480      # ลดจาก 832 → 480
TARGET_H        = 288      # ลดจาก 480 → 288       # 81 frames @ 24fps = ~3 วินาที

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