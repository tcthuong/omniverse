# CFD Surrogate — Implementation Notes
_Last updated: 2026-04-30  |  All units: RPM (user-facing), rad/s (internal model)_

---

## Codebase location

| Component | Path |
|---|---|
| Training + inference | `D:\work\` |
| Kit app template | `D:\nvidia\kit-app-template\` |
| Web UI | `D:\work\web_ui\` |
| Kit build output | `D:\nvidia\kit-app-template\_build\windows-x86_64\release\` |
| Kit extension source | `D:\nvidia\kit-app-template\source\extensions\thuong.tc_extension\` |

---

## Architecture overview

```
SimScale / OpenFOAM cases (input/)
        ↓ PyVista POpenFOAMReader
01_export_openfoam.py  →  out_gt/omega_<ω>/{internal_mesh.vtu, surface.vtp, U.vdb, p.vdb}
        ↓
┌── 02_train_mgn.py  (arch=MGN)  →  out_outputs/   ─┐
└── 02_train_fno.py  (arch=FNO)  →  out_fno/       ─┤
        ↓                                          │
03_export_model.py --arch auto  →  cfd_surrogate.ts (TorchScript, arch-agnostic I/O)
        ↓
Kit streaming app loads cfd_surrogate.ts
        ↓
web_ui RPM slider  →  cfd.setOmega {rpm, omega_rad_s}  →  WebRTC  →  Kit message bus
        ↓
cfd_inference.py listens  →  model(omega_rad_s)  →  update USD Points prim live
        ↓
WebRTC stream → browser viewport
```

---

## Key files

### `D:\work\01_export_openfoam.py` _(Step 1 — ground truth export)_
Exports each OpenFOAM case to standard interchange formats for archiving,
Paraview inspection, and Omniverse ingestion.

Outputs per omega under `out_gt/omega_<ω>/`:
| File | Format | Contents |
|---|---|---|
| `internal_mesh.vtu` | VTK UnstructuredGrid | Full volumetric mesh, all fields (U, p, k, nut, omega_turb) |
| `surface.vtp` | VTK PolyData | Extracted surface, point-interpolated fields |
| `U.vdb` | OpenVDB Vec3SGrid | Velocity field on regular voxel grid (requires pyopenvdb) |
| `p.vdb` | OpenVDB FloatGrid | Pressure field on regular voxel grid (requires pyopenvdb) |

VDB pipeline: cell-center cloud → PyVista ImageData interpolation → pyopenvdb grid.

**Run:**
```powershell
pip install pyopenvdb                          # one-time, optional dep
python 01_export_openfoam.py                   # all cases → out_gt/
python 01_export_openfoam.py --no-vdb          # skip VDB (no pyopenvdb needed)
python 01_export_openfoam.py --grid-res 256    # finer VDB voxel grid
```

### `D:\work\02_train_mgn.py`
- Reads OpenFOAM cases via `cfd_cases.discover_cases()`
- omega read from `constant/MRFProperties` or folder name (rad/s, NOT RPM)
- 3 cases: 150 / 250 / 350 rad/s; middle = validation LOO split
- Builds KNN graph (k=8, physical units)
- **PhysicsNeMo MeshGraphNet**: input_dim_nodes=4 (xyz_norm + omega_norm), output_dim=7 (U×3, p, k, nut, omega_turb)
- Physics-informed normalization: U/ω, p/ω², k/ω², log(ω_turb)
- **PI loss (Phase 1)**: `--pi-weight λ` activates continuity residual ∇·U≈0
  - MLS gradient operator precomputed from KNN rel positions (N, 3, K)
  - Per-step cost: 1 gather + 1 einsum (no autograd through MGN)
  - `loss = data_mse + λ * mean((Σ U_std_i · ∂U_norm_i/∂x_i)²)`
  - Log field `pi` added to training_log.json per epoch
- Inference sweep: 21 frames linspace(ω_min, ω_max) → predictions.npz
- Checkpoints: `out_outputs/checkpoint_best.pt`, `checkpoint.pt`

**New args:**
```
--pi-weight 0.1      # enable continuity loss (default 0 = off)
--pi-ridge  1e-6     # MLS normal equation ridge (default 1e-6)
```

### `D:\work\02_train_fno.py` _(FNO — preferred by roadmap)_
- **3D Fourier Neural Operator** via `physicsnemo.models.fno.FNO`
- Voxelizes cell-center cloud to regular `(G, G, G)` grid (nearest-neighbor via cKDTree)
- Input: `(1, 4, G, G, G)` = normalized coord grid [−1,1]³ + omega channel
- Output: `(1, 7, G, G, G)` = all fields; same physics-informed normalization as MGN
- **PI loss**: finite-difference `∇·U≈0` on the regular grid (central diff, simpler than MLS)
- Saves `norm_stats.json` with `arch: fno`, `grid_res`, `bmin`, `bmax`, FNO hyperparams
- Saves `cell_centers.npy` = regular grid physical coords `(G³, 3)` for Kit USD Points prim
- Output dir: `out_fno/` (separate from MGN's `out_outputs/`)

**Run:**
```powershell
python 02_train_fno.py                                     # defaults: G=64, 3000 epochs
python 02_train_fno.py --grid-res 128 --fno-modes 16       # higher resolution
python 02_train_fno.py --pi-weight 1e-3                    # with continuity loss
python 03_export_model.py --arch fno --out-dir out_fno     # export after training
```

### `D:\work\03_export_model.py` _(Step 3 — export TorchScript, arch-agnostic)_
- `--arch auto` (default) reads `arch` field from `norm_stats.json`
- **MGN path** (`arch=mgn`): wraps `MeshGraphNet` in `CFDSurrogate` with KNN graph baked in
- **FNO path** (`arch=fno`): wraps `FNO` in `FNOSurrogate` with coord-grid baked in
- Both surrogates share the **same interface**: input = scalar `omega_rad_s`, output = `(N, 7)`
  - `cfd_inference.py` works without any change for either architecture
- Traces via `torch.jit.trace` → saves `cfd_surrogate.ts` + `cfd_surrogate.json`

**Run:**
```powershell
python 03_export_model.py                                  # MGN from out_outputs/
python 03_export_model.py --arch fno --out-dir out_fno     # FNO from out_fno/
python 03_export_model.py --device cuda --test-omega 300   # sanity check on GPU
```

### `D:\work\kit_extension\cfd_inference.py` _(new — Phase 3, staged copy)_
Canonical source at `D:\work\kit_extension\cfd_inference.py`.
Deployed to `…\thuong.tc_extension\thuong\tc_extension\cfd_inference.py`.

- `CFDInferenceExtension(omni.ext.IExt)` — auto-discovered by Kit, no changes to extension.py
- `CFDInferenceBridge`:
  - Loads TorchScript model via `torch.jit.load`
  - Reads `cell_centers` buffer from model
  - Creates USD Xform + Points prim at `/World/CFD/Predictions`
  - Points: positions = cell_centers, widths uniform, `displayColor` primvar = turbo ramp on |U|
  - Subscribes to carb event bus: `carb.events.type_from_string("cfd.setOmega")`
  - Each event → `model(omega)` → update `displayColor` on Points prim
  - Logs `omega, |U|max, inference_ms` per call

**Carb settings (override in .kit file or via `--/`):**
```
/exts/thuong.tc_extension/cfd/enabled      = true
/exts/thuong.tc_extension/cfd/model_path   = "D:/work/out_outputs/cfd_surrogate.ts"
/exts/thuong.tc_extension/cfd/device       = "auto"   # auto|cpu|cuda
/exts/thuong.tc_extension/cfd/stage_root   = "/World/CFD"
/exts/thuong.tc_extension/cfd/point_width  = 0.005
```

**To update after edits:**
```powershell
Copy-Item d:\work\kit_extension\cfd_inference.py `
  "D:\nvidia\kit-app-template\source\extensions\thuong.tc_extension\thuong\tc_extension\cfd_inference.py" -Force
```
(Build dir is a symlink → propagates immediately, no rebuild needed.)

### `D:\work\web_ui\src\main.js`
- Sends on slider change (debounced 80ms):
```json
{
  "event_type": "cfd.setOmega",
  "payload": {
    "omega_rad_s": 250,
    "frame": 10,
    "omega_min_rad_s": 150,
    "omega_max_rad_s": 350,
    "frame_step_rad_s": 10
  }
}
```
- `omega_rad_s` is what `cfd_inference.py` reads.
- `frame` is kept for backward-compat with timeline seek.

---

## End-to-end run sequence

```powershell
# Step 1 — train (first time, or retrain with PI loss)
cd D:\work
python 02_train_mgn.py --pi-weight 0.1 --epochs 3000 --device cuda
# or: python 02_train_fno.py --pi-weight 1e-3 --epochs 3000 --device cuda

# Step 2 — export TorchScript
python 03_export_model.py
# verify: out_outputs/cfd_surrogate.ts exists

# Step 3 — install torch into Kit's Python (one-time, see note below)
$kitpy = "D:\nvidia\kit-app-template\_build\windows-x86_64\release\kit\python\python.exe"
& $kitpy -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu124

# Step 4 — start Kit streaming app
cd D:\nvidia\kit-app-template\_build\windows-x86_64\release
.\thuong.tc_streaming.kit.bat --no-window `
  "--/exts/thuong.tc_extension/cfd/model_path=D:/work/out_outputs/cfd_surrogate.ts" `
  "--/exts/thuong.tc_extension/cfd/device=cuda"

# Step 5 — start web UI
cd D:\work\web_ui
npm run dev
# open http://localhost:5177/?host=127.0.0.1&port=49100&mediaPort=47998
```

---

## Known gaps / remaining work

| Gap | Status | Notes |
|---|---|---|
| .vtp / .vdb export | ✅ Done | `01_export_openfoam.py` → per-omega `.vtu` / `.vtp` / `.vdb` under `out_gt/` |
| RPM unit everywhere | ✅ Done | Web UI slider in RPM; payload contains both `rpm` and `omega_rad_s`; Kit logs in RPM; `CfdCase.rpm` property |
| FNO architecture | ✅ Done | `02_train_fno.py` — standalone 3D FNO training; exports via `03_export_model.py --arch fno` |
| `arch` field in norm_stats.json | ✅ Done | Both `02_train_mgn.py` (mgn) and `02_train_fno.py` (fno) write `arch` key; `03_export_model.py --arch auto` reads it |
| FNO PI loss (FD continuity) | ✅ Done | `02_train_fno.py --pi-weight λ` uses central finite-difference ∇·U=0 on regular grid |
| Navier-Stokes momentum loss | ❌ Deferred | Continuity only; momentum needs ν_eff + ∇p — requires pressure-velocity coupling, post-MVP |
| SimScale → OpenFOAM bridge | ❌ External | Requires SimScale Converter Extension + API credentials; out of scope |
| Volumetric / VDB rendering | ⚠️ Partial | VDB files exported; Kit live volume renderer not wired (needs OmniGraph VDB prim) |
| Action Graph node | ❌ Optional | Using IExt Python class; Action Graph is artist-friendly alternative, post-MVP |
| AI optimization agent | ❌ Post-MVP | Find ideal RPM for target airflow via gradient-based or evolutionary search |
| Streamlines live update | ❌ Post-MVP | Too heavy per-frame; needs async compute + BasisCurves USD update |
| torch in Kit Python | ⚠️ Manual step | Must pip-install into Kit's embedded Python (see run sequence above) |

---

## Dependency versions (d:\work)

```
nvidia-physicsnemo==1.3.0
torch==2.11.0  (cu124)
torch-geometric==2.7.0
torch-scatter==2.1.2
pyvista==0.47.3
numpy==2.4.4
scipy==1.17.1
```

## Known hardware issue (local machine)

RTX 3050 Laptop (Display Active: Disabled) + AMD iGPU hybrid config causes:
`NVST_DISCONN_SERVER_VIDEO_ADAPTER_CUDA_CREATE_CONTEXT_FAILED`
WebRTC streaming fails at NVENC CUDA context init. Not a code issue.
Fix: attach external monitor to NVIDIA port, or use MUX dGPU-only mode, or upgrade driver to ≥595.97.
