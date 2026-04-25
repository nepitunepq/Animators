"""
blender_render.py
Run with:  blender --background --python blender_render.py

Reads poses_elev*_azim*_r*.json files, renders each clip using EEVEE,
then encodes each clip to MP4 via ffmpeg.  Also renders a 1080p ground-truth
clip from poses_elev20_azim0_r5.json.
"""

import bpy        # type: ignore  (Blender-internal; install fake-bpy-module-latest for IDE stubs)
import mathutils  # type: ignore
import json
import os
import glob
import subprocess
import re
import shutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All paths are anchored to the script's own directory so they work
# regardless of what directory Blender was launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

POSES_DIR   = os.path.join(SCRIPT_DIR, "camera_output")
RENDERS_DIR = os.path.join(SCRIPT_DIR, "renders")
CLIPS_DIR   = os.path.join(SCRIPT_DIR, "clips")

# ── Set this manually if ffmpeg is still not found automatically ──────────
FFMPEG_OVERRIDE = ""   # set this if ffmpeg is not on PATH, e.g. r"C:\ffmpeg\bin\ffmpeg.exe"
# ──────────────────────────────────────────────────────────────────────────

def _find_ffmpeg():
    if FFMPEG_OVERRIDE:
        return FFMPEG_OVERRIDE
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        # Krita's ffmpeg is intentionally last — it lacks libx264
        r"C:\Program Files\Krita (x64)\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

FFMPEG = _find_ffmpeg()

DEFAULT_RES_X   = 1920
DEFAULT_RES_Y   = 1080
DEFAULT_RES_PCT = 50       # 50 % → 960 × 540 for speed; override for GT

GROUND_TRUTH = {
    "file":    "poses_elev20_azim0_r5.json",
    "name":    "ground_truth_orbit",
    "res_pct": 100,
}


# ---------------------------------------------------------------------------
# SECTION 1 — Scene initialisation
# ---------------------------------------------------------------------------

def clear_scene():
    """Delete every object that ships with a fresh Blender scene."""
    bpy.ops.object.select_all(action="SELECT")   # select cube, camera, light
    bpy.ops.object.delete(use_global=False)       # delete them all


def add_suzanne():
    """Add the default Suzanne monkey head at the world origin."""
    bpy.ops.mesh.primitive_monkey_add(           # built-in monkey mesh
        size=2.0,
        location=(0.0, 0.0, 0.0),
    )
    monkey = bpy.context.active_object           # grab just-created object
    monkey.name = "Suzanne"
    return monkey


def add_camera():
    """Add a camera and register it as the scene's active camera."""
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))   # position later per-frame
    cam_obj = bpy.context.active_object                     # the new camera object
    cam_obj.name = "RenderCam"
    bpy.context.scene.camera = cam_obj                      # make it the render camera
    return cam_obj


# ---------------------------------------------------------------------------
# SECTION 2 — 3-point studio lighting
# ---------------------------------------------------------------------------

def _make_area_light(name, location, rotation_euler, energy):
    """Helper: create one AREA lamp and return it."""
    bpy.ops.object.light_add(
        type="AREA",                    # area lights give soft studio shadows
        location=location,
    )
    light_obj = bpy.context.active_object
    light_obj.name = name
    light_obj.rotation_euler = rotation_euler   # point it at the subject
    light_obj.data.energy = energy              # watts
    light_obj.data.size   = 3.0                 # larger = softer shadow
    return light_obj


def setup_three_point_lighting():
    """Create key, fill, and back lights for a classic studio look."""

    # Key light  — strong, upper-left of subject
    _make_area_light(
        name="KeyLight",
        location=(-4.0, -4.0, 6.0),
        rotation_euler=(0.9, 0.0, -0.6),   # tilted down-right
        energy=800,
    )

    # Fill light — softer, upper-right, opens up shadows
    _make_area_light(
        name="FillLight",
        location=(4.0, -3.0, 4.0),
        rotation_euler=(0.8, 0.0, 0.6),    # tilted down-left
        energy=300,
    )

    # Back light — rim/hair light behind and above subject
    _make_area_light(
        name="BackLight",
        location=(0.0, 5.0, 5.0),
        rotation_euler=(-0.8, 0.0, 0.0),   # tilted forward toward subject
        energy=500,
    )


# ---------------------------------------------------------------------------
# SECTION 3 — Render settings
# ---------------------------------------------------------------------------

def setup_render(res_x=DEFAULT_RES_X, res_y=DEFAULT_RES_Y, res_pct=DEFAULT_RES_PCT):
    """Configure EEVEE renderer with a pure-white background."""

    scene = bpy.context.scene

    # --- engine ---
    scene.render.engine = "BLENDER_EEVEE"          # fast GPU rasteriser

    # --- resolution ---
    scene.render.resolution_x          = res_x
    scene.render.resolution_y          = res_y
    scene.render.resolution_percentage = res_pct   # % of res_x × res_y

    # --- output format ---
    scene.render.image_settings.file_format = "PNG"

    # --- white background ---
    # Ensure the world shader exists
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    scene.world.use_nodes = True                   # activate node graph
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = scene.world.node_tree.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)    # RGBA white
    bg_node.inputs["Strength"].default_value = 1.0

    # Transparent film would show alpha; override to opaque white
    scene.render.film_transparent = False


# ---------------------------------------------------------------------------
# SECTION 4 — Per-frame camera placement and render
# ---------------------------------------------------------------------------

def render_clip(poses, frames_dir, cam_obj):
    """
    Iterate over a list of pose dicts, set the camera for each frame,
    and render a numbered PNG into frames_dir.
    """
    os.makedirs(frames_dir, exist_ok=True)
    scene = bpy.context.scene

    cam_obj.rotation_mode = "QUATERNION"    # switch before the loop; stays set

    for pose in poses:
        frame_num = pose["frame"]
        w, x, y, z = pose["q"]             # unpack quaternion components
        px, py, pz  = pose["pos"]          # unpack world-space position

        # --- move camera ---
        cam_obj.location = mathutils.Vector((px, py, pz))   # world position

        # --- orient camera (quaternion avoids gimbal lock) ---
        cam_obj.rotation_quaternion = mathutils.Quaternion([w, x, y, z])

        # --- advance the timeline (keeps keyframe system consistent) ---
        scene.frame_set(frame_num)

        # --- set output path for this frame ---
        scene.render.filepath = os.path.join(
            frames_dir, f"frame_{frame_num:04d}.png"
        )

        # --- render and save PNG ---
        bpy.ops.render.render(write_still=True)   # write_still=True saves the file


# ---------------------------------------------------------------------------
# SECTION 5 — ffmpeg: PNG frames → MP4
# ---------------------------------------------------------------------------

def frames_to_mp4(frames_dir, output_path, fps=24):
    """Call ffmpeg to encode all frame_XXXX.png files into an H264 MP4."""
    if not FFMPEG:
        raise FileNotFoundError(
            "ffmpeg not found. Install it and add to PATH, or place it at C:\\ffmpeg\\bin\\ffmpeg.exe"
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        FFMPEG, "-y",                         # -y: overwrite without asking
        "-framerate", str(fps),               # input frame rate
        "-i", os.path.join(frames_dir, "frame_%04d.png"),   # numbered input
        "-c:v", "libx264",                    # H264 video codec
        "-pix_fmt", "yuv420p",                # broad compatibility
        "-crf", "18",                         # quality (lower = better, 18 ≈ visually lossless)
        output_path,
    ]

    subprocess.run(cmd, check=True)           # raises on non-zero exit


# ---------------------------------------------------------------------------
# SECTION 5 — Batch loop
# ---------------------------------------------------------------------------

def batch_render():
    """
    Find every poses_elev*_azim*_r*.json, render all frames, encode to MP4.
    Progress is printed as "Rendering clip N/M: elevX_azimY_rZ".
    """
    pattern = os.path.join(POSES_DIR, "poses_elev*_azim*_r*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"[batch_render] No pose files found matching: {pattern}")
        return

    total = len(json_files)

    for idx, json_path in enumerate(json_files, start=1):
        fname = os.path.basename(json_path)

        # Extract elev / azim / r from filename
        m = re.match(r"poses_elev(-?\d+)_azim(\d+)_r(\d+)", fname)
        if not m:
            print(f"  [skip] Cannot parse filename: {fname}")
            continue

        elev, azim, r = m.group(1), m.group(2), m.group(3)
        clip_label   = f"elev{elev}_azim{azim}_r{r}"

        print(f"Rendering clip {idx}/{total}: {clip_label}")

        # Directories for this clip
        frames_dir = os.path.join(RENDERS_DIR, clip_label)
        mp4_path   = os.path.join(CLIPS_DIR, f"orbit_{clip_label}.mp4")

        # Load pose data
        with open(json_path) as f:
            poses = json.load(f)

        # --- fresh scene for each clip ---
        clear_scene()
        add_suzanne()
        cam_obj = add_camera()
        setup_three_point_lighting()
        setup_render()          # 50 % resolution for batch clips

        # Render all frames
        render_clip(poses, frames_dir, cam_obj)

        # Encode to MP4
        frames_to_mp4(frames_dir, mp4_path)

        print(f"  -> saved {mp4_path}")


# ---------------------------------------------------------------------------
# SECTION 6 — Ground truth (1080p)
# ---------------------------------------------------------------------------

def render_ground_truth():
    """
    Render poses_elev20_azim0_r5.json at full 1080p.
    Output: ./clips/ground_truth_orbit.mp4
    """
    gt_path = os.path.join(POSES_DIR, GROUND_TRUTH["file"])  # camera_output/poses_elev20_azim0_r5.json
    if not os.path.exists(gt_path):
        print(f"[ground_truth] File not found: {gt_path}  — skipping.")
        return

    print(f"\nRendering ground-truth clip at 1080p: {GROUND_TRUTH['file']}")

    with open(gt_path) as f:
        poses = json.load(f)

    clip_label = GROUND_TRUTH["name"]
    frames_dir = os.path.join(RENDERS_DIR, clip_label)
    mp4_path   = os.path.join(CLIPS_DIR, f"{clip_label}.mp4")

    # Fresh scene at full resolution
    clear_scene()
    add_suzanne()
    cam_obj = add_camera()
    setup_three_point_lighting()
    setup_render(
        res_x=DEFAULT_RES_X,
        res_y=DEFAULT_RES_Y,
        res_pct=GROUND_TRUTH["res_pct"],    # 100 % = true 1920 × 1080
    )

    render_clip(poses, frames_dir, cam_obj)
    frames_to_mp4(frames_dir, mp4_path)

    print(f"  -> ground truth saved: {mp4_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(RENDERS_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR,   exist_ok=True)

    batch_render()
    render_ground_truth()

    print("\nAll done.")
