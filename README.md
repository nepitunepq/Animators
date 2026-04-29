# Animators — Orbital Camera Control with LoRA + MotionCtrl

โปรเจกต์นี้ทดลองสอนให้ AI วิดีโอ (Wan 2.2) เรียนรู้การเคลื่อนกล้องแบบ orbital โดยใช้ quaternion + SLERP interpolation แล้วเปรียบเทียบกับ MotionCtrl

## ทีม

| คนชื่อ | หน้าที่ | ไฟล์หลัก |
|---|---|---|
| Nine | คำนวณ camera path (quaternion + SLERP) | `camera_math.py` |
| Yoshi | render ฉากด้วย Blender | `blender_render.py` |
| Daisy | เตรียม dataset + train LoRA | `dataset_prep.py`, `train_lora.py` |
| Rose | evaluate และ เปรียบเทียบผล | `compare_video.py` |

---

## Hardware ที่ต้องการ

| Phase | เครื่องที่ใช้ | GPU |
|---|---|---|
| camera math, dataset prep, compare | เครื่องตัวเอง | ไม่จำเป็น |
| Blender render | เครื่องตัวเอง | CPU ได้ (แต่ช้า) |
| LoRA training | Kaggle / Colab | A100 หรือ T4 (T4 ใช้ Wan 2.1) |
| MotionCtrl inference | Kaggle | GPU 16GB+ |

---

## Setup

**ต้องการ:** Python 3.10+, ffmpeg, Blender 4.x

```bash
# 1. clone repo
git clone https://github.com/nepitunepq/Animators.git
cd Animators

# 2. สร้าง virtual env
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. ตรวจสอบ environment
python check_setup.py
```

**ffmpeg** — ดาวน์โหลดจาก https://ffmpeg.org/download.html แล้วเพิ่มใน PATH

**Blender** — ดาวน์โหลดจาก https://www.blender.org/download/

---

## Workflow

### Phase 1 — Camera Math (Nine)
```bash
python camera_math.py
# → camera_output/camera_poses.json
```
สร้าง orbital camera path ครบ 360° หลาย elevation และ radius

---

### Phase 2 — Blender Render (Yoshi)
```bash
blender --background --python blender_render.py
# → renders/  (PNG frames)
# → clips/    (MP4 วิดีโอ)
```
ใช้ `camera_poses.json` จาก Phase 1 render ฉาก Suzanne (monkey head) หลาย angle

---

### Phase 3 — Dataset Prep (Daisy)
```bash
python dataset_prep.py
# → wan-dataset/videos/   (MP4 resize แล้ว)
# → wan-dataset/captions/ (text description)
```
Resize คลิปให้ตรงกับ resolution ที่ Wan รับ และสร้าง caption อัตโนมัติ

---

### Phase 4A — LoRA Training บน Kaggle/Colab (Daisy)

อัพโหลด `wan-dataset/` ขึ้น Kaggle Dataset แล้วรัน notebook `train_colab.ipynb`
หรือรัน script ตรงๆ บนเครื่องที่มี A100:

```bash
python train_lora.py
# → lora_output/  (LoRA weights)
# ใช้เวลา ~3-5 ชม. บน A100
```

> **หมายเหตุ:** T4 (Colab free) รองรับแค่ Wan 2.1 — ดู `train_colab.ipynb`

---

### Phase 4B — MotionCtrl Inference บน Kaggle (Yoshi/Daisy)

เปิด notebook `train-with-motionctrl.ipynb` บน Kaggle แล้วทำตามขั้นตอนนี้ก่อน:

1. ดาวน์โหลด `motionctrl.pth` จาก [TencentARC/MotionCtrl releases](https://github.com/TencentARC/MotionCtrl)
2. อัพโหลด `motionctrl.pth` และ `camera_poses.json` ขึ้น Kaggle Dataset
3. Add dataset เข้า notebook แล้วแก้ `YOUR_DATASET` ใน CONFIG cell
4. รัน cell ตามลำดับ
5. → ได้ GIF เปรียบเทียบ vanilla vs SLERP orbit

---

### Phase 5 — เปรียบเทียบผล (Rose)
```bash
python compare_video.py
# → comparison.gif  (3 panel side-by-side)
```

---

## ไฟล์สำคัญ

```
config.py                   — ค่า config ทุกอย่างรวมไว้ที่นี่ที่เดียว
camera_math.py              — quaternion + SLERP logic (Nine)
blender_render.py           — Blender automation (Yoshi)
dataset_prep.py             — เตรียม dataset (Daisy)
train_lora.py               — LoRA training script (Daisy)
train_colab.ipynb           — notebook สำหรับ Colab T4 (Wan 2.1)
train-with-motionctrl.ipynb — MotionCtrl inference บน Kaggle
compare_video.py            — สร้าง comparison GIF (Rose)
check_setup.py              — ตรวจสอบ environment
```

---

## ผลลัพธ์ที่คาดหวัง

| วิดีโอ | ที่มา | ความหมาย |
|---|---|---|
| `ground_truth.gif` | Blender render | กล้องหมุน perfect จาก math |
| `vanilla.gif` | Wan 2.2 ไม่ train | AI ยังไม่รู้จัก orbital motion |
| `slerp.gif` | Wan 2.2 + LoRA | AI เรียนรู้ orbital motion แล้ว |
| `comparison.gif` | compare_video.py | เปรียบเทียบ 3 วิธีแบบ side-by-side |
