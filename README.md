# Animators — Orbital Camera Control with MotionCtrl

โปรเจกต์นี้ใช้ MotionCtrl ควบคุมการเคลื่อนกล้องแบบ orbital โดยคำนวณ camera path ด้วย quaternion + SLERP interpolation แล้วเปรียบเทียบผลกับ Blender ground truth

## ทีม

| คนชื่อ | หน้าที่ | ไฟล์หลัก |
|---|---|---|
| Nine | คำนวณ camera path (quaternion + SLERP) | `camera_math.py` |
| Yoshi | render ฉากด้วย Blender | `blender_render.py` |
| Daisy | รัน MotionCtrl inference บน Kaggle | `train-with-motionctrl.ipynb` |
| Rose | evaluate และ เปรียบเทียบผล | `compare_video.py` |

---

## Hardware ที่ต้องการ

| Phase | เครื่องที่ใช้ | GPU |
|---|---|---|
| camera math, compare | เครื่องตัวเอง | ไม่จำเป็น |
| Blender render | เครื่องตัวเอง | CPU ได้ (แต่ช้า) |
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

### Phase 3 — MotionCtrl Inference บน Kaggle (Daisy)

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
train-with-motionctrl.ipynb — MotionCtrl inference บน Kaggle (Daisy)
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
