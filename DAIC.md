# DAIC Cluster Access & Layout

Operational notes for working on TU Delft's DAIC HPC cluster. Intended for both
humans and AI agents working on this repository.

## Connecting from the Mac

SSH alias `daic` is configured in `~/.ssh/config`:

```
Host daic
    HostName login1.hpc.tudelft.nl
    User agrigore
    ProxyJump tudelft-bastion          # student-linux.tudelft.nl
    ControlMaster auto
    ControlPath ~/.ssh/cm-daic
    ControlPersist 8h
```

- **VPN required.** Without the TU Delft VPN, connections die with
  `Connection timed out during banner exchange` (network-level, before auth).
- **Auth:** the bastion (`student-linux.tudelft.nl`) accepts the local SSH key
  `~/.ssh/id_ed25519`. `login1` accepts **password only** (server-side policy);
  the password is stored in the macOS keychain (service `daic-hpc`, account
  `agrigore`) and fed by the askpass helper `~/.ssh/daic-askpass.sh`.
- **Persistent master connection** is kept alive by autossh (installed via
  Homebrew). All `ssh daic` / `scp` / `rsync ... daic:...` commands multiplex
  over it — no password prompts while it is up.

### (Re)starting the master connection

Needed after a reboot or VPN drop (autossh dies with the VPN):

```bash
SSH_ASKPASS=~/.ssh/daic-askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
    AUTOSSH_GATETIME=0 autossh -M 0 -f -N daic
```

Verify with: `ssh daic hostname` → should print `login1.hpc.tudelft.nl`
with no prompt.

### For agents / non-interactive shells

Non-interactive shells (e.g. AI agent Bash tools) have no TTY, so always export
the askpass variables first, otherwise a dead master causes a silent auth
failure:

```bash
export SSH_ASKPASS=$HOME/.ssh/daic-askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
ssh daic '<command>'
```

## Important paths on DAIC

| Path | Contents |
|---|---|
| `/home/nfs/agrigore/projects/githubProjects/master-thesis` | This repo (branch `train-one-heliostat-at-a-time`) |
| `/home/nfs/agrigore/projects/ARTIST` | ARTIST clone — **copied into the SIF at build time**; must match local commit |
| `/home/nfs/agrigore/projects/PAINT` | PAINT clone — same, copied into the SIF |
| `/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif` | The Apptainer image all GPU jobs run in |
| `/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.def` | Image definition (see also `artist-local.def` in this repo root) |
| `/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint` | PAINT benchmark + generated synthetic datasets |
| `.../datasets/paint/synthetic/balanced_dataset/dataset` | The 62-heliostat synthetic dataset used by FEL/stage-1/2 (12113 PNG + json, byte-mirrored from the Mac via rsync — **do not regenerate on DAIC**, see below) |
| `<repo>/logs/` | SLURM stdout/stderr logs (`*_<jobid>.log`) |
| `<repo>/outputs/fine_error_learning/` | FEL run outputs (models, results.json, evaluation) |
| `<repo>/outputs/fine_error_learning/all63_stage1_synth/` | Stage-1 checkpoints (62, committed to git) |

`/home/nfs` = home (small, backed-up-ish); `/tudelft.net/staff-umbrella/...` =
bulk storage for images and datasets. Compute nodes see both.

## Running jobs

Everything runs inside the SIF via Apptainer; sbatch scripts live in
`src/sbatch_files/`. Standard invocation pattern (from `<repo>/src`):

```bash
apptainer exec --nv --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python <script.py> --daic ...
```

- `--daic` switches config paths from local to the DAIC paths above.
- **QOS:** jobs over ~4 h walltime need `#SBATCH --qos=medium`, otherwise
  `sbatch` fails with `QOSMaxWallDurationPerJobLimit`.
- GPU jobs: `#SBATCH --gres=gpu:a40:1` (A40, ~3 GB used by FEL training runs,
  ~11 s/epoch for the 62-heliostat FEL matrix).

Key scripts:

- `src/sbatch_files/build_artist_sif.sh` — rebuilds `artist-local.sif` from the
  DAIC ARTIST/PAINT clones (CPU job, ~30–60 min, `--force` overwrites in place).
- `src/sbatch_files/run_fel_daic_matrix.sh` — FEL experiment matrix (3 × 300
  epochs + evals); includes pre-flight checks for scenarios and dataset.
- `src/fine_error_learning/check_daic_setup.py` — verifies imports, paths,
  scenario/dataset/checkpoint presence on DAIC before training.

## Reproducibility rules (learned the hard way, 2026-08)

1. **Keep ARTIST/PAINT commits identical on Mac and DAIC.** The synthetic
   dataset's motor positions and focal spots are *computed* by ARTIST during
   generation; a version drift between the local editable install and the SIF
   silently produced a different dataset (~1 mrad warm-start shift). Check
   `git log -1` in both clones before regenerating anything.
2. **Don't regenerate the balanced dataset on DAIC** — rsync the Mac-rendered
   one instead (GPU vs CPU ray-tracer RNG differs). Verify with a checksum
   dry-run: `rsync -nci --delete --no-perms dataset/ daic:.../dataset/` should
   print nothing.
3. rsync to the umbrella share fails to set permissions ("Operation not
   permitted") — harmless for content; use `--no-perms` to silence, and expect
   exit code 23 even on success.
4. Sanity gate for FEL runs on this dataset: warm-start (stage-1 checkpoint)
   test-split centroid error must be ≈ **1.678 mrad mean / 1.288 median**. If
   "before" is ~2.67, the dataset on DAIC is stale/wrong.
