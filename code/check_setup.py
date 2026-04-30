# daisy/setup_check.py
# ทดสอบว่า environment พร้อมสำหรับโปรเจกต์ทั้งหมด

import importlib, sys

REQUIRED = [
    "torch",
    "cv2",          # opencv-python
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "numpy",
    "PIL",          # pillow
    "tqdm",
    "ffmpeg",       # ffmpeg-python
]

print("=" * 40)
print("  Daisy — Environment Check")
print("=" * 40)

all_ok = True
for pkg in REQUIRED:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "ok")
        print(f"  ✓ {pkg:<20} {version}")
    except ImportError:
        print(f"  ✗ {pkg:<20} NOT FOUND")
        all_ok = False

print("=" * 40)

# เช็ค GPU
import torch
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("  GPU : Apple MPS (Mac) — ใช้ได้บางส่วน")
else:
    print("  GPU : None — training ต้องไปรันบน Colab/RunPod")

print(f"  Python : {sys.version.split()[0]}")
print("=" * 40)

if all_ok:
    print("  All good! พร้อมเขียน code ✓")
else:
    print("  มี library ขาด — รัน pip install ด้านล่างก่อน")
    print()
    print("  pip install torch torchvision diffusers transformers")
    print("  pip install accelerate huggingface_hub opencv-python")
    print("  pip install pillow tqdm ffmpeg-python numpy")