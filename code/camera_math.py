from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

CENTER     = [0.0, 0.0, 0.0]
N_FRAMES   = 120
ELEVATIONS = [-20, 0, 20, 40]
AZIMUTHS   = [0, 45, 90, 135]
RADII      = [4, 5, 6]


# ─────────────────────────────────────────────
#  Quaternion
# ─────────────────────────────────────────────

@dataclass
class Quat:
    w: float
    x: float
    y: float
    z: float

    # ── arithmetic ──────────────────────────

    def __mul__(self, other: "Quat") -> "Quat":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quat(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

    def conjugate(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def dot(self, other: "Quat") -> float:
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Quat":
        n = self.norm()
        return Quat(self.w/n, self.x/n, self.y/n, self.z/n)

    # ── geometry ────────────────────────────

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        vq = Quat(0.0, float(v[0]), float(v[1]), float(v[2]))
        rq = self * vq * self.conjugate()
        return np.array([rq.x, rq.y, rq.z])

    # ── serialisation ───────────────────────

    def to_list(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]

    # ── constructors ────────────────────────

    @staticmethod
    def identity() -> "Quat":
        return Quat(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle_rad: float) -> "Quat":
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return Quat.identity()
        axis = axis / n
        half = angle_rad / 2.0
        s = math.sin(half)
        return Quat(math.cos(half), axis[0]*s, axis[1]*s, axis[2]*s)

    @staticmethod
    def look_at(
        eye: np.ndarray,
        center: np.ndarray,
        up: np.ndarray = np.array([0.0, 1.0, 0.0]),
    ) -> "Quat":
        forward = center - eye
        forward = forward / np.linalg.norm(forward)

        if abs(np.dot(forward, up)) > 0.9999:
            up = np.array([0.0, 0.0, 1.0])

        right   = np.cross(forward, up)
        right   = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)

        R = np.array([
            [ right[0],  true_up[0], -forward[0]],
            [ right[1],  true_up[1], -forward[1]],
            [ right[2],  true_up[2], -forward[2]],
        ])

        return Quat._matrix_to_quat(R).normalized()

    @staticmethod
    def _matrix_to_quat(R: np.ndarray) -> "Quat":
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            return Quat(0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s)
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return Quat((R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s)
        if R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return Quat((R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s)
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return Quat((R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s)


# ─────────────────────────────────────────────
#  SLERP
# ─────────────────────────────────────────────

def slerp(q1: Quat, q2: Quat, t: float) -> Quat:
    dot = q1.dot(q2)

    if dot < 0.0:
        q2  = Quat(-q2.w, -q2.x, -q2.y, -q2.z)
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 1.0 - 1e-4:
        return Quat(
            q1.w + t*(q2.w - q1.w),
            q1.x + t*(q2.x - q1.x),
            q1.y + t*(q2.y - q1.y),
            q1.z + t*(q2.z - q1.z),
        ).normalized()

    omega     = math.acos(dot)
    sin_omega = math.sin(omega)
    k1        = math.sin((1.0 - t) * omega) / sin_omega
    k2        = math.sin(t * omega)          / sin_omega

    return Quat(
        k1*q1.w + k2*q2.w,
        k1*q1.x + k2*q2.x,
        k1*q1.y + k2*q2.y,
        k1*q1.z + k2*q2.z,
    )


# ─────────────────────────────────────────────
#  Camera path generation
# ─────────────────────────────────────────────

def orbital_path(
    center: List[float] = (0.0, 0.0, 0.0),
    radius: float = 5.0,
    n_frames: int = 120,
    elevation_deg: float = 20.0,
    azimuth_offset_deg: float = 0.0,
) -> List[dict]:
    center_arr = np.asarray(center, dtype=float)
    elev_rad   = math.radians(elevation_deg)
    azim0_rad  = math.radians(azimuth_offset_deg)
    world_up   = np.array([0.0, 0.0, 1.0] if abs(elevation_deg) > 85.0 else [0.0, 1.0, 0.0])

    poses = []
    for i in range(n_frames):
        phi = azim0_rad + 2.0 * math.pi * i / n_frames
        x   = center_arr[0] + radius * math.cos(elev_rad) * math.cos(phi)
        y   = center_arr[1] + radius * math.sin(elev_rad)
        z   = center_arr[2] + radius * math.cos(elev_rad) * math.sin(phi)
        eye = np.array([x, y, z])
        q   = Quat.look_at(eye, center_arr, world_up)
        poses.append({"frame": i, "q": q.to_list(), "pos": [x, y, z]})

    return poses


def generate_dataset(output_dir: str = ".") -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)
    configs  = list(itertools.product(ELEVATIONS, AZIMUTHS, RADII))
    manifest = []

    print(f"Generating {len(configs)} orbital configurations ...")
    for uid, (elev, azim, r) in enumerate(configs):
        poses = orbital_path(
            center=CENTER, radius=r, n_frames=N_FRAMES,
            elevation_deg=elev, azimuth_offset_deg=azim,
        )
        fname = f"poses_elev{elev}_azim{azim}_r{r}.json"
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(poses, f, indent=2)
        manifest.append({"id": uid, "file": fname, "elevation": elev, "azimuth": azim, "radius": r})

    with open(os.path.join(output_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} pose files + dataset_manifest.json -> {output_dir}/")
    return manifest


# ─────────────────────────────────────────────
#  Unit tests
# ─────────────────────────────────────────────

def _angular_velocity_variance(poses: List[dict]) -> float:
    n = len(poses)
    angles = []
    for i in range(n):
        q1  = Quat(*poses[i]["q"])
        q2  = Quat(*poses[(i + 1) % n]["q"])
        dot = min(abs(q1.dot(q2)), 1.0)
        angles.append(2.0 * math.acos(dot))
    return float(np.var(angles))


def _forward_direction(q: Quat) -> np.ndarray:
    return q.rotate_vector(np.array([0.0, 0.0, -1.0]))


def run_tests(poses: List[dict], label: str = "") -> bool:
    prefix   = f"[{label}] " if label else ""
    center   = np.array(CENTER)
    all_pass = True

    # T1 — unit norm
    max_err = max(abs(Quat(*p["q"]).norm() - 1.0) for p in poses)
    t1 = max_err < 1e-6
    print(f"{prefix}T1 Unit quaternions   : {'PASS' if t1 else 'FAIL'}  (max |q|-1 = {max_err:.2e})")
    all_pass = all_pass and t1

    # T2 — constant angular velocity
    var = _angular_velocity_variance(poses)
    t2  = var < 0.001
    print(f"{prefix}T2 Const angular vel  : {'PASS' if t2 else 'FAIL'}  (var = {var:.6f} rad^2)")
    all_pass = all_pass and t2

    # T3 — look-at direction
    min_dot = min(
        float(np.dot(
            _forward_direction(Quat(*p["q"])),
            (center - np.array(p["pos"])) / np.linalg.norm(center - np.array(p["pos"])),
        ))
        for p in poses
    )
    t3 = min_dot > 0.999
    print(f"{prefix}T3 Camera looks at ctr: {'PASS' if t3 else 'FAIL'}  (min dot = {min_dot:.6f})")
    all_pass = all_pass and t3

    return all_pass


def test_all_configs(output_dir: str = ".") -> bool:
    with open(os.path.join(output_dir, "dataset_manifest.json")) as f:
        manifest = json.load(f)

    results = []
    for entry in manifest:
        with open(os.path.join(output_dir, entry["file"])) as f:
            poses = json.load(f)
        label = f"elev={entry['elevation']:+d} azim={entry['azimuth']:3d} r={entry['radius']}"
        results.append(run_tests(poses, label=label))

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} configs -> {'ALL PASS' if passed == total else 'FAILURES'}")
    return passed == total


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def visualise_path(poses: List[dict], title: str = "Orbital Camera Path") -> None:
    xs = [p["pos"][0] for p in poses]
    ys = [p["pos"][1] for p in poses]
    zs = [p["pos"][2] for p in poses]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(xs + [xs[0]], ys + [ys[0]], zs + [zs[0]],
            "b-", linewidth=1.5, label="Camera path", alpha=0.7)
    ax.scatter(xs, ys, zs, c="royalblue", s=12, zorder=5)

    for p in poses[::10]:
        fwd = Quat(*p["q"]).rotate_vector(np.array([0.0, 0.0, -1.0]))
        ax.quiver(*p["pos"], *fwd, length=0.6, color="red", linewidth=0.8)

    ax.scatter([0], [0], [0], c="gold", s=120, marker="*", label="Scene center", zorder=10)

    r    = np.linalg.norm(poses[0]["pos"])
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    ax.plot_wireframe(r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v),
                      color="grey", alpha=0.08, linewidth=0.4)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("camera_path.png", dpi=150)
    print("Saved visualisation -> camera_path.png")
    plt.show()


# ─────────────────────────────────────────────
#  Statistical comparison
# ─────────────────────────────────────────────

def statistical_comparison(var_A: float, var_B: float, var_C: float, n: int = 120) -> dict:
    from math import erfc

    diff_B    = abs(var_B - var_A)
    diff_C    = abs(var_C - var_A)
    se        = math.sqrt(2 * var_A**2 / (n - 1))
    pooled_se = math.sqrt(2) * se
    z         = (diff_C - diff_B) / pooled_se if pooled_se > 0 else float("inf")
    p_value   = 0.5 * erfc(z / math.sqrt(2))

    return {
        "diff_B_from_A":         diff_B,
        "diff_C_from_A":         diff_C,
        "improvement_C_minus_B": diff_C - diff_B,
        "z_score":               z,
        "p_value":               p_value,
        "B_closer_than_C":       diff_B < diff_C,
        "significant_at_0.05":   p_value < 0.05,
    }


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    OUT = "camera_output"
    os.makedirs(OUT, exist_ok=True)

    # Single default orbit
    print("=" * 60)
    print("Single default orbit (elev=20, r=5, 120 frames)")
    print("=" * 60)
    default_poses = orbital_path(center=[0,0,0], radius=5, n_frames=120, elevation_deg=20)
    with open(os.path.join(OUT, "camera_poses.json"), "w") as f:
        json.dump(default_poses, f, indent=2)
    print("Wrote camera_poses.json")
    run_tests(default_poses, label="default")
    visualise_path(default_poses, title="Orbital Camera Path (elev=20, r=5)")

    # Multi-config dataset
    print("\n" + "=" * 60)
    print("Multi-configuration dataset")
    print("=" * 60)
    generate_dataset(output_dir=OUT)

    # Unit tests
    print("\n" + "=" * 60)
    print("Unit tests on all configs")
    print("=" * 60)
    test_all_configs(output_dir=OUT)

    # Statistical comparison
    print("\n" + "=" * 60)
    print("Statistical comparison (replace with real variance scores)")
    print("=" * 60)
    stats = statistical_comparison(var_A=0.0012, var_B=0.0018, var_C=0.0087)
    for k, v in stats.items():
        print(f"  {k}: {v}")