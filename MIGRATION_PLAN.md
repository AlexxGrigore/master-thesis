# Migration Plan: Update to latest ARTIST & PAINT

**Scope**: `src/five_heliostats_synth/` + shared `src/utils/evaluation.py`  
**Trigger**: Pulling latest `main` from ARTIST (~240 new commits) and PAINT (4 new commits).

---

## Prerequisites

Before writing any code, confirm the following.

### 1. Locate PAINT raw data for the 5 heliostats

The new `create_scenario.py` needs these files for each of AA31, AQ28, BA37, BC32, AZ55:
- `<heliostat>/heliostat-properties.json`
- `<heliostat>/deflectometry.h5`

And one shared file:
- `tower-measurements.json`

On DAIC these are somewhere under `/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/src/paint_benchmarks/`.
Find the exact subdirectory structure before starting.

### 2. Verify the new `solar_tower` API

After pulling ARTIST, check `artist/field/solar_tower.py` to confirm:
- How to get target area **centers** (replaces `scenario.target_areas.centers[mask]`)
- How to get target area **dimensions** (replaces `scenario.target_areas.dimensions[mask]`)
- The exact signature of `get_centers_of_target_areas()`

These are used in at least 6 places across `data.py` and `evaluation.py`.

### 3. Verify `trace_rays` return signature

Check `artist/core/heliostat_ray_tracer.py` → `trace_rays()` to confirm it now returns a 4-tuple:
`flux, intercept_factor, blocking_factor, on_target_factor`

### 4. Verify `reconstruct_kinematics` return signature

Check `artist/core/kinematics_reconstructor.py` → `reconstruct_kinematics()` to confirm it now
returns `(loss, loss_history)` instead of just `loss`.

---

## Changes

### Change 1 — `create_scenario.py` (full rewrite)

**Current approach**: copies 5 heliostat groups from `deflectometry_scenario.h5` using raw `h5py`.  
**Problem**: The new HDF5 schema requires a `geometry` field on target areas and the `solar_tower`
group structure. The old file cannot be used as a copy source.  
**New approach**: Build from scratch using `H5ScenarioGenerator` + `paint_scenario_parser`,
following `ARTIST/tutorials/00_generate_scenario_from_paint_tutorial.py`.

#### New structure:

```python
# 1. Tower + target areas from PAINT tower-measurements.json
power_plant_config, target_area_list_config = (
    paint_scenario_parser.extract_paint_tower_measurements(
        tower_measurements_path=tower_file, device=device
    )
)

# 2. Light source (same parameters as before)
light_source_list_config = LightSourceListConfig([
    LightSourceConfig(
        light_source_key="sun_1",
        light_source_type=config_dictionary.sun_key,
        number_of_rays=10,
        distribution_type=config_dictionary.light_source_distribution_is_normal,
        mean=0.0,
        covariance=4.3681e-06,
    )
])

# 3. Heliostats with fitted deflectometry surfaces (NURBS 20x20, same as tutorial)
heliostat_files_list = [
    ("AA31", Path(".../AA31/heliostat-properties.json"), Path(".../AA31/deflectometry.h5")),
    ("AQ28", ...),
    ("BA37", ...),
    ("BC32", ...),
    ("AZ55", ...),
]

heliostat_list_config, prototype_config = (
    paint_scenario_parser.extract_paint_heliostats_fitted_surface(
        paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        number_of_nurbs_control_points=torch.tensor([20, 20]),
        deflectometry_step_size=100,
        nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
        nurbs_fit_tolerance=1e-10,
        nurbs_fit_max_epoch=400,
        nurbs_fit_optimizer=...,   # Adam lr=1e-3
        nurbs_fit_scheduler=...,   # ReduceLROnPlateau, same as tutorial
        device=device,
    )
)

# 4. Write HDF5
H5ScenarioGenerator(
    file_path=output_path,
    power_plant_config=power_plant_config,
    target_area_list_config=target_area_list_config,
    light_source_list_config=light_source_list_config,
    prototype_config=prototype_config,
    heliostat_list_config=heliostat_list_config,
).generate_scenario()
```

Keep the existing CLI flags (`--force`, `--daic`) and path logic.  
Note: NURBS fitting runs on each heliostat — this takes a few minutes, but runs once.

---

### Change 2 — `src/utils/evaluation.py`

This is a **shared file** used by all experiments. Changes here affect `kr_training_defl_only`
and `kr_train_3_losses` too — update carefully and test each experiment after.

#### 2a. `scenario.target_areas` → `scenario.solar_tower` (4 occurrences)

| Line | Current | Replace with |
|------|---------|--------------|
| 130 | `scenario.target_areas.centers[:, :3].mean(dim=0)` | `scenario.solar_tower.<new API>` |
| 157 | `scenario.target_areas.centers[target_area_mask]` | `scenario.solar_tower.get_centers_of_target_areas(...)` |
| 183 | `scenario.target_areas.centers[target_area_mask[sample_indices]]` | `scenario.solar_tower.get_centers_of_target_areas(...)` |
| 184–188 | `scenario.target_areas.dimensions[...][..., target_area_width/height]` | `scenario.solar_tower.<new dimensions API>` |
| 325 | `scenario.target_areas.centers[target_area_mask]` (in `compute_pixel_test_loss`) | `scenario.solar_tower.get_centers_of_target_areas(...)` |

#### 2b. `ray_tracer.trace_rays()` returns 4-tuple (2 occurrences)

Lines 171–176 and 338–343:
```python
# OLD
predicted_flux = ray_tracer.trace_rays(...)

# NEW
predicted_flux, _, _, _ = ray_tracer.trace_rays(...)
```

---

### Change 3 — `src/five_heliostats_synth/data.py`

#### 3a. `scenario.target_areas` → `scenario.solar_tower` (3 occurrences)

| Line | Current | Replace with |
|------|---------|--------------|
| 136 | `scenario.target_areas.centers[target_mask]` | `scenario.solar_tower.get_centers_of_target_areas(...)` |
| 161 | `scenario.target_areas.centers[target_mask[sample_indices]]` | `scenario.solar_tower.get_centers_of_target_areas(...)` |
| 162–166 | `scenario.target_areas.dimensions[...][..., target_area_width/height]` | `scenario.solar_tower.<new dimensions API>` |

#### 3b. `ray_tracer.trace_rays()` returns 4-tuple (line 151)

```python
# OLD
flux = ray_tracer.trace_rays(
    incident_ray_directions=incident_rays,
    active_heliostats_mask=active_mask,
    target_area_indices=target_mask,
    device=device,
)

# NEW
flux, _, _, _ = ray_tracer.trace_rays(
    incident_ray_directions=incident_rays,
    active_heliostats_mask=active_mask,
    target_area_indices=target_mask,
    device=device,
)
```

#### 3c. `HeliostatRayTracer` constructor — check `world_size`/`rank` removal

Lines 142–149 use `world_size=1, rank=0, random_seed=42`. The new constructor in `evaluation.py`
already omits these. Confirm whether they were removed or became optional in the new ARTIST version,
and align `data.py` accordingly.

---

### Change 4 — `src/five_heliostats_synth/train.py`

#### 4a. `reconstruct_kinematics()` returns 2-tuple (line 140)

```python
# OLD
reconstructor.reconstruct_kinematics(
    loss_definition=FocalSpotLoss(scenario=scenario), device=device
)

# NEW
_loss, _loss_history = reconstructor.reconstruct_kinematics(
    loss_definition=FocalSpotLoss(scenario=scenario), device=device
)
```

Note: `reconstructor._convergence_history` is still accessed on line 147 — verify it still
exists as an attribute in the new version (it may now be returned directly as `_loss_history`
instead of stored on the object).

---

### Change 5 — PAINT constant rename (minor, check all files)

PAINT renamed `KINEMATIC_PROPERTIES_KEY` → `KINEMATICS_PROPERTIES_KEY` (and the suffix variant).
Grep across `src/` to find any references:

```bash
grep -r "KINEMATIC_PROPERTIES" src/
```

Replace all matches. This likely does not affect `five_heliostats_synth` directly but may affect
other experiments or shared utilities.

---

## Order of execution

1. Pull latest ARTIST and PAINT.
2. Verify the new APIs (solar_tower, trace_rays, reconstruct_kinematics) — prerequisites above.
3. Rewrite `create_scenario.py` (Change 1). Run it once to generate the new scenario HDF5.
4. Fix `src/utils/evaluation.py` (Change 2) — shared, highest blast radius.
5. Fix `src/five_heliostats_synth/data.py` (Change 3).
6. Fix `src/five_heliostats_synth/train.py` (Change 4).
7. Fix PAINT constant rename if found (Change 5).
8. Run a quick smoke test: `main.py` with `n_train=10`, 1 reconstructor class, few epochs.

## Out of scope

- `kr_training_defl_only/` and `kr_train_3_losses/` — will also need `evaluation.py` fixes (Change 2)
  and potentially the same `trace_rays` / `reconstruct_kinematics` fixes. Do separately.
- `deflectometry_scenario.h5` — the large 376-heliostat scenario will also need regeneration
  eventually, but is not needed for `five_heliostats_synth`.
