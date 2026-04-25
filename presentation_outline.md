# Mathematical-Driven Camera Synthesis: From Quaternions to Generative Models
### Presentation Outline — 12 slides, ~12 minutes

---

## SLIDE 1 — Title

**Title:** Mathematical-Driven Camera Synthesis: From Quaternions to Generative Models

**Subtitle:** Teaching AI to Move a Camera the Way Physics Says It Should

**Author / Date / Institution**

---
**Speaker notes (50 words):**
"Welcome. Today I want to show you a specific problem in AI video generation — the camera doesn't move correctly — and a principled, math-first attempt to fix it. The solution involves 3D rotation algebra, Blender dataset rendering, and LoRA fine-tuning. Let's start with the problem."

---

## SLIDE 2 — Problem: AI Video Fails at Camera Control

**Headline:** Modern video AI can't hold a steady orbit

**Visual:**
Two side-by-side video previews (or stills):
- LEFT — Smooth perfect circular orbit around an object (labelled "What we want")
- RIGHT — Camera drifting, jumping, wobbling around the same object (labelled "What AI does")

**Bullet points:**
- Text-to-video models have no concept of camera geometry
- Rotation is predicted implicitly → Euler-angle ambiguity → gimbal lock → jerky output
- Angular velocity is not constant → the orbit "breathes" and drifts
- No way to specify exact elevation, azimuth, radius

---
**Speaker notes (50 words):**
"Current text-to-video models generate camera motion as learned patterns, not physics. The result is a camera that wobbles — some frames it rotates fast, some slow. You can see this on the right. This is the baseline we're comparing against: vanilla AI output with no geometric guidance."

---

## SLIDE 3 — Solution: 3-Phase Pipeline Overview

**Headline:** Three-phase approach: Math → Data → Model

**DIAGRAM — Pipeline flowchart (left to right):**

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   PHASE 1: MATH     │────▶│  PHASE 2: DATASET    │────▶│  PHASE 3: EVAL      │
│                     │     │                      │     │                     │
│ Quaternion algebra  │     │ Blender renders 48   │     │ LoRA fine-tune on   │
│ SLERP interpolation │     │ orbital clips        │     │ video model         │
│ Orbital path math   │     │ camera_math.py →     │     │ Compare: Ground     │
│ Unit-test verified  │     │ blender_render.py    │     │ Truth vs Trained    │
│                     │     │                      │     │ vs Vanilla          │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

---
**Speaker notes (50 words):**
"Here's the full pipeline. Phase 1 is pure math — we derive correct quaternion-based orbital paths. Phase 2 uses Blender to turn those paths into rendered video data. Phase 3 fine-tunes a video AI on that data and measures whether it absorbed the geometric constraint. Simple, reproducible, testable."

---

## SLIDE 4 — Phase 1 Math: Quaternions vs Euler Angles + Gimbal Lock

**Headline:** Why quaternions? Euler angles break at the poles.

### Left diagram — Gimbal Lock (draw/animate)

```
      OUTER RING (Yaw — blue)
    ╔═══════════════════════╗
    ║  MIDDLE RING (Pitch — green)
    ║  ╔═════════════════╗  ║
    ║  ║  INNER RING     ║  ║
    ║  ║  (Roll — red)   ║  ║
    ║  ║  [CAMERA 📷]    ║  ║
    ║  ╚═════════════════╝  ║
    ╚═══════════════════════╝

When Pitch = ±90°:
  → Middle ring aligns with Outer ring
  → Roll and Yaw become IDENTICAL axes
  → ONE DEGREE OF FREEDOM LOST ⚠️
```

*Annotation:* "The camera can no longer rotate independently around one axis."

### Right diagram — Quaternion unit sphere

```
            q = (w, x, y, z),  |q| = 1

                  ·  q
                 /
    ────────────O────────────  Unit sphere
                 \
                  ·  -q   (same rotation, opposite sign)

  No singularity.  No locked axes.  SLERP-compatible.
```

**Key equation:**
```
q = cos(θ/2) + sin(θ/2)(xi + yj + zk)
```

---
**Speaker notes (50 words):**
"Euler angles describe rotation as three sequential rotations. The problem: when two axes align — called gimbal lock — you lose a degree of freedom and the camera snaps. Quaternions encode the same rotation as a point on a 4D unit sphere. There's no singularity, so no snap."

---

## SLIDE 5 — Phase 1 Math: SLERP Formula + What It Guarantees

**Headline:** SLERP = constant angular velocity, provably.

### Left diagram — SLERP vs LERP on the unit sphere

```
           q2
          ·
         /|
   arc  / |  chord
(SLERP)/  |(LERP)
      /   |
     ·    ·  — midpoint is OFF sphere
    q1   (LERP midpoint — re-normalising
              moves it, speed NOT constant)

  SLERP midpoint ·  stays ON sphere
                    (speed IS constant)

  "Arc-length ∝ t  →  dθ/dt = const"
```

### Right diagram — Orbital path with constant Δθ

```
         top-down view

               ★  Suzanne (origin)
             /   \
          ·         ·
       Δθ↑           ↑Δθ
       ·               ·
       Δθ↑           ↑Δθ
          ·         ·
             \   /
               ·

  All Δθ arrows are equal → uniform orbit
```

**SLERP formula:**
```
SLERP(q₁, q₂, t) = sin((1-t)Ω)/sinΩ · q₁  +  sin(tΩ)/sinΩ · q₂

where Ω = arccos(q₁ · q₂)
```

**Result:** angular velocity variance < 0.001 rad² (verified by unit test T2 across all 48 configs)

---
**Speaker notes (50 words):**
"LERP — straight-line interpolation — shortcuts through the sphere interior, so speed changes. SLERP follows the arc, so arc-length is proportional to t, and dθ/dt is exactly constant. Our test suite verifies this across all 48 orbital configurations. The math guarantees what the AI should learn to imitate."

---

## SLIDE 6 — Phase 2: Blender Dataset Generation Pipeline

**Headline:** 48 orbital configurations → 5,760 rendered frames

**DIAGRAM — Blender pipeline:**

```
camera_math.py                    blender_render.py
──────────────                    ─────────────────
orbital_path()                    clear_scene()
  ├─ elevation: [-20,0,20,40]°      add_suzanne()          PNG frames
  ├─ azimuth:   [0,45,90,135]°  →   add_camera()        →  ./renders/
  └─ radius:    [4,5,6] units       setup_lights()            │
                                    setup_eevee()          ffmpeg
poses_elev20_azim45_r5.json  ──▶   render_clip()      →  ./clips/
  [{"frame":0, "q":[…], "pos":[…]}, …]                orbit_*.mp4
```

**Numbers:**
| Parameter | Values | Count |
|-----------|--------|-------|
| Elevation | −20, 0, 20, 40° | 4 |
| Azimuth   | 0, 45, 90, 135° | 4 |
| Radius    | 4, 5, 6 units   | 3 |
| **Total configs** | | **48** |
| Frames per clip | 120 | — |
| **Total frames** | | **5,760** |

---
**Speaker notes (50 words):**
"We generate the dataset in two stages. camera_math.py produces 48 JSON files, each with 120 camera poses verified by unit tests. blender_render.py reads each JSON, positions the camera per-frame using the quaternion, renders with EEVEE, then calls ffmpeg to encode MP4. Fully automated, no manual keyframing."

---

## SLIDE 7 — Phase 2: LoRA Training Setup

**Headline:** Fine-tune a video model to move like the math

**Content:**

**Why LoRA (Low-Rank Adaptation)?**
- Updates ~0.1% of model parameters
- Preserves general video generation ability
- Injects geometric prior without full retraining

**Training setup:**
| Setting | Value |
|---------|-------|
| Base model | [video diffusion model] |
| LoRA rank | 16 |
| Training data | 48 orbital MP4 clips |
| Frames per clip | 120 |
| Learning rate | 1e-4 |
| Epochs | [N] |

**Hypothesis:** Fine-tuning on quaternion-consistent trajectories will bias the model toward constant-angular-velocity orbits.

---
**Speaker notes (50 words):**
"LoRA injects small adapter matrices into the attention layers. The model sees 48 clips of perfectly smooth, mathematically correct orbits and should update its internal distribution toward that behaviour. The key unknown — which this experiment tests — is whether LoRA's limited rank is sufficient to inject a global geometric constraint."

---

## SLIDE 8 — Phase 3: Evaluation Method

**Headline:** Measuring whether AI learned the geometric constraint

**Metric — Angular Velocity Variance:**

```python
for each consecutive frame pair (q₁, q₂):
    angle = 2 · arccos(|q₁ · q₂|)   # angle between rotations

variance = Var(angles)   # lower = smoother = better
```

**Three conditions:**
- **A — Ground truth:** mathematical SLERP orbit (variance ≈ 0)
- **B — AI trained:** LoRA-tuned output on same prompt
- **C — AI vanilla:** base model output, no fine-tuning

**Statistical test:** z-score comparing |var_B − var_A| vs |var_C − var_A|

---
**Speaker notes (50 words):**
"We extract each video's per-frame camera orientation, compute the angle between consecutive quaternions, and measure variance. The ground truth variance is near zero by construction. The question is whether the trained model's variance is meaningfully lower than the vanilla model's. A p-value below 0.05 confirms statistical significance."

---

## SLIDE 9 — Results: 3-Screen Comparison Video

**Headline:** See the difference: math vs trained vs vanilla

**[FULL-SLIDE VIDEO PLACEHOLDER]**

```
┌──────────────────┬──────────────────┬──────────────────┐
│                  │                  │                  │
│  A: Math         │  B: AI Trained   │  C: AI Vanilla   │
│  Ground Truth    │                  │                  │
│                  │                  │                  │
│  Smooth orbit    │  Near-smooth?    │  Wavy / drifty   │
│                  │                  │                  │
└──────────────────┴──────────────────┴──────────────────┘
         ← Play comparison_final.mp4 here →
```

**What to watch for:**
- Does B hold the circle as steadily as A?
- Does C show visible wobble / drift that B avoids?
- Does the camera consistently face Suzanne in all three?

---
**Speaker notes (100 words):**
"Let's watch. The left panel is our mathematical ground truth — generated by SLERP, verified by unit tests, perfectly smooth. The right panel is vanilla AI — no geometric training. Watch the camera drift: it slows down, speeds up, sometimes loses the subject entirely. That wavy motion is what we're trying to fix.

The center panel is the LoRA-trained model on the same text prompt. Ask yourself: is the orbit smoother? Does it hold the circle? Does it face Suzanne on every frame? The visual impression matters here, but the next slide quantifies it with hard numbers."

---

## SLIDE 10 — Results: Velocity Curve Graphs

**Headline:** Flat line = good. Wavy line = AI guessing.

**DIAGRAM — Three angular velocity curves (line chart):**

```
Angular velocity (rad/frame)
│
│  ────────────────────────────────  A: Ground truth (blue, flat)
│
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  B: AI trained (orange, small bumps)
│
│  /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\   C: AI vanilla (red, high variance)
│
└──────────────────────────────────────▶ Frame
   0                              120
```

**Results table:**

| Method | Variance (rad²) | vs Ground Truth |
|--------|----------------|-----------------|
| A: Ground truth | ~0.000 | — |
| B: AI trained | [value] | [Δ%] |
| C: AI vanilla | [value] | [Δ%] |

**p-value:** [value] — [significant / not significant] at α=0.05

---
**Speaker notes (100 words):**
"These curves show angular velocity per frame — how fast the camera rotated between each pair of frames. The ideal is a flat horizontal line: constant speed, no wobble.

The blue line is our ground truth. Flat by construction. The orange line is LoRA-trained AI. The red line is vanilla AI.

A flat line means the model learned the geometric constraint. A wavy line means it's still guessing. The variance numbers on the table summarize this — lower is better. The p-value tells us whether the gap between trained and vanilla is statistically real or just noise. Read the p-value aloud and interpret it."

---

## SLIDE 11 — Discussion: Did AI Learn Physics?

**Headline:** What the results actually tell us

**Two scenarios to present honestly:**

### If B < C (trained is smoother than vanilla):
- LoRA successfully transferred the geometric prior
- The model learned to approximate SLERP-like behaviour from data
- Remaining variance suggests temporal attention still has geometric limits
- Next step: geometry-aware loss during training

### If B ≈ C (no improvement):
- LoRA rank is insufficient for global geometric constraints
- SLERP is not a local token-level pattern — it's a global sequence property
- The mathematical dataset is validated; the limitation is in the adaptation method
- This is still a valid result: it tells us *where* the bottleneck is
- Next step: target temporal attention layers directly, or add a smoothness loss

**Key insight either way:** The math pipeline is correct. The question was always about the model's capacity to absorb it.

---
**Speaker notes (50 words):**
"A negative result is not a failed project. If the trained model isn't better, we've learned that LoRA's low-rank updates can't inject a global geometric constraint. That's an important finding about fine-tuning methodology. It points to specific architectural interventions for future work, which is exactly what good research does."

---

## SLIDE 12 — Conclusion

**Headline:** Summary and what's next

**What we built:**
1. A mathematically rigorous orbital camera system (quaternions, SLERP, unit-tested)
2. An automated Blender rendering pipeline (48 configs, 5,760 frames)
3. A LoRA fine-tuning experiment on synthetic geometric data
4. A quantitative evaluation framework (angular velocity variance, p-test)

**What we learned:**
- Quaternions are the right representation for this problem
- Generating clean synthetic data is tractable
- [Conclusion from results: LoRA succeeded / or: LoRA alone is insufficient]

**Future work:**
- Geometry-aware loss function (penalise angular velocity deviation)
- Larger dataset: more elevations, radii, objects
- Evaluate on temporal self-attention layers specifically
- Test with real camera trajectories (COLMAP reconstruction)

---
**Speaker notes (50 words):**
"To summarise: we went from 'AI cameras wobble' to a full mathematical pipeline with a quantitative answer. The camera math is solid — SLERP, quaternions, unit tests all confirm it. Whether a LoRA is sufficient to teach a video model that math is the open question this project answered. Thank you."

---

---

# Q&A Preparation

## Q1: Why quaternions instead of Euler angles?

**Model answer:**
Euler angles describe rotation as three sequential rotations around fixed axes. The problem is gimbal lock: when two axes align, you lose one degree of freedom — the camera can no longer rotate independently around one axis. This causes the snapping and flipping you see in naively animated cameras at elevation extremes like ±90°.

Quaternions avoid this by living on a 4D unit sphere. There's no sequence of rotations, so no two axes can ever align. Additionally, SLERP is only mathematically meaningful on the quaternion manifold — you can't SLERP Euler tuples and get constant angular velocity. The geometry requires quaternions.

---

## Q2: Prove that SLERP guarantees constant angular velocity.

**Model answer:**
SLERP is defined as:

```
SLERP(q₁, q₂, t) = sin((1-t)Ω)/sinΩ · q₁  +  sin(tΩ)/sinΩ · q₂
```

where Ω = arccos(q₁ · q₂) is the angle between the two quaternions.

The key insight: on a unit sphere, arc-length equals angle. The formula advances by arc-length proportional to t. So when t increases from 0 to 1 uniformly, the arc-length increases uniformly, which means the angle swept per unit time is constant — i.e., constant angular velocity.

Our T2 unit test verifies this empirically: angular velocity variance across all 120 frames is below 0.001 rad² for all 48 configurations.

---

## Q3: Isn't LoRA too limited for geometric learning?

**Model answer:**
That's a fair concern and arguably the central hypothesis of the experiment. LoRA updates low-rank adapter matrices inside attention layers — typically updating ~0.1% of parameters. Geometric priors like SLERP are global sequence constraints: the relationship between frame 1 and frame 120 matters, not just adjacent tokens.

Our hypothesis was that fine-tuning on a large set of consistent orbital trajectories would shift the model's distribution toward those patterns implicitly — without needing to encode the constraint explicitly. Whether that's sufficient is exactly what the experiment tests. If it fails, the conclusion is that we need either higher-rank adapters, geometric loss supervision, or architectural changes to temporal attention layers.

---

## Q4: How valid is angular velocity variance as an evaluation metric?

**Model answer:**
Angular velocity variance is a direct, interpretable proxy for the smoothness constraint we care about — and it's the exact quantity that SLERP minimises by construction, so it's internally consistent with our ground truth.

Its limitation: it measures temporal smoothness but not spatial accuracy. A camera could have low variance but still not face the object. Ideally we'd also compute center-pointing error — the angle between the camera's forward vector and the vector toward the object. We chose variance as the primary metric because it directly tests the geometric property we tried to inject via LoRA.

---

## Q5: Why only 48 configurations? Is that enough training data?

**Model answer:**
48 configs × 120 frames = 5,760 training frame-pairs. For LoRA fine-tuning at rank 16, updating on the order of millions of parameters, this is a reasonable starting point — comparable to domain-adaptation experiments in the literature.

The limitation is generalization: we used four fixed elevations, four fixed azimuths, and three radii. A model trained on this set may not generalize well to unseen combinations. This is a scoped proof-of-concept. Expanding to continuous elevation/azimuth sampling and multiple objects would be a natural next step. The pipeline supports this — camera_math.py parametrizes all of these as arguments.

---

## Handling a negative result (if B is NOT better than C)

**What to say:**
"A negative result is still a scientific contribution. We've established that LoRA fine-tuning on quaternion-consistent orbital data does not reliably transfer a global geometric constraint to a video diffusion model. This is informative: it tells us that the bottleneck is not the training data — our dataset is mathematically verified — but the adaptation method itself.

LoRA modifies attention weights, which operate locally on token sequences. SLERP is a global property of the entire trajectory. These are structurally mismatched. The natural next step is to either use a geometry-aware loss during training that directly penalises angular velocity deviation, or to target the temporal self-attention layers specifically, which govern how the model attends across frames."

**Key framing:** The math pipeline is correct. The hypothesis about LoRA was falsified. That's a real finding.
