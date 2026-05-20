# Thesis: Topic & Methodology

**Title:** Heliostat Motor Error Correction using Physics-Aware Deep Learning
**Subtitle:** A Two-Stage Coarse-to-Fine Calibration Approach for Concentrating Solar Power Plants
**Author:** Alexandru Grigore — TU Delft, Faculty of Aerospace Engineering

---

## Topic

Concentrating Solar Power (CSP) plants use large fields of mirrors (heliostats) to redirect sunlight onto a central receiver. Each heliostat is steered by two motors (azimuth and elevation), and sub-milliradian pointing accuracy is required for efficient operation. In practice, manufacturing tolerances, installation imprecision, and environmental effects (thermal expansion, wind, mechanical wear) cause the real mirror orientation to deviate from what the geometric model predicts.

This thesis addresses the problem of **correcting these motor errors** through a two-stage pipeline called **Coarse-to-Fine Error Learning**:

- **Coarse Error Learning (CEL):** gradient-based optimization of rigid-body kinematic parameters using the ARTIST differentiable ray tracer. Achieves ~2–3 mrad accuracy, which is the ceiling for purely geometric approaches.
- **Fine Error Learning (FEL):** a deep neural network trained on top of CEL outputs to predict residual corrections that the rigid-body optimizer cannot capture. Uses flux images (rather than tabular centroid data) as the supervision signal, with gradients flowing through the full physical simulation via ARTIST's differentiable ray tracer.

The work is evaluated on the **PAINT dataset** (Solar Tower Jülich, 1893 heliostats, 218,713 calibration images) using the ARTIST simulation framework. The target is to push accuracy below 1 mrad — a threshold no prior method has reliably achieved on real field data.

---

## Methodology

The proposed pipeline has four stages:

### Stage 1 — Scenario Generation
An ARTIST simulation scenario is constructed from three PAINT data sources: tower geometry (global coordinates and calibration target dimensions), static heliostat properties (kinematic parameters, physical dimensions, installation orientation), and deflectometry measurements (high-resolution surface normals used to build NURBS mirror representations). Without deflectometry data, ARTIST assumes an ideal flat mirror surface, introducing systematic simulation error.

### Stage 2 — Coarse Motor Error Optimization
The ARTIST `KinematicReconstructor` (KR) performs gradient-based calibration of the 28 rigid-body kinematic parameters per heliostat. Two loss functions are used in sequence:

- **FocalSpotLoss** — Euclidean distance in 3D space between the predicted and measured flux centroid:
  `L_focal = ||ĉ - c||₂`
  Provides reliable gradients even when the heliostat is only partially on target.

- **PixelLoss** — Mean squared error between predicted and measured full flux bitmaps:
  `L_pixel = Σ(Î_{h,w} - I_{h,w})²`
  Richer supervision signal but requires the heliostat to already be approximately on target (zero gradient if flux misses the calibration target entirely).

**Two-phase training:** Phase 1 runs 100 epochs with FocalSpotLoss to bring all heliostats on target. Phase 2 runs 300 epochs with PixelLoss to refine using the full spatial structure of the flux images.

### Stage 3 — Fine Motor Error Correction Network
A deep neural network takes a single heliostat description as input and outputs a correction vector Δθ added to the KR-calibrated parameters:

`θ_final = θ_KR + Δθ`

The corrected parameters are passed through ARTIST to produce a predicted flux image, and the network is trained end-to-end via backpropagation through the ray tracer. Training proceeds in two sub-stages:

- **Stage 3.1 — Single-Heliostat Training:** A mask isolates one heliostat at a time. The ray tracer simulates the full field but only the unmasked heliostat contributes to the loss, providing a clean and unambiguous learning signal per heliostat.

- **Stage 3.2 — Multi-Heliostat Fine-Tuning:** The mask is removed. The training target becomes a desired uniform flux distribution across the entire receiver. The loss is computed against the collective optical output of all heliostats, bridging individual calibration accuracy and real operational performance.

### Stage 4 — Experimental Setup
Experiments use the **Balanced Split** benchmark from the PAINT database: 376 heliostats, all with deflectometry data. The split partitions calibration samples into train/validation/test sets using k-means clustering over solar azimuth and elevation angles, ensuring all sets cover a representative and evenly distributed range of sun positions. Each simulation uses the exact solar geometry recorded at measurement time.

---

## Research Objectives

1. Can Fine Error Learning achieve sub-milliradian accuracy on individual heliostats, improving upon the 1–2 mrad ceiling of geometric models?
2. How does the multi-heliostat trained model perform compared to geometric baselines on realistic field scenarios?
3. Do shared kinematic deviation patterns across heliostats allow the network to exploit common structure during Stage 2 training?
4. Can the trained network generalize to heliostats with error patterns not seen during training (practical deployability)?

---

## Key Tools & Data

| Component | Description |
|---|---|
| **ARTIST** | Open-source differentiable ray tracer (PyTorch). Enables gradient flow through full physical simulation. |
| **KinematicReconstructor** | ARTIST module for gradient-based kinematic calibration. |
| **PAINT dataset** | Real-world calibration data from Solar Tower Jülich: 1893 heliostats, 218,713 images, 654 deflectometry scans. |
| **Balanced Split** | k-means benchmark split ensuring uniform coverage of sun positions across train/val/test. |
