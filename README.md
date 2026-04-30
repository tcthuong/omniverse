# CFD MeshGraphNet Surrogate

This project trains and visualizes a CFD surrogate model for OpenFOAM rotor cases.
The current dataset contains three incompressible solution fields at 150, 250, and
350 rad/s. The training script learns on sampled OpenFOAM cells, generates a 21
frame omega sweep, and exports visualization assets for PyVista and Omniverse.

## Repository Map

- `input/`: OpenFOAM case folders with `case.foam` and solution fields.
- `01_export_openfoam.py`: exports each OpenFOAM case to `.vtu` / `.vtp` / `.vdb` ground-truth files.
- `02_train_mgn.py`: trains PhysicsNeMo MeshGraphNet surrogate → `out_outputs/`.
- `02_train_fno.py`: trains PhysicsNeMo FNO surrogate (roadmap-preferred) → `out_fno/`.
- `03_export_model.py`: wraps trained model in a TorchScript surrogate → `cfd_surrogate.ts`.
- `04_viz.py`: renders predicted streamlines to PNG and exports animated USD.
- `04_export_streamlines.py`: lower-level USD streamline exporter.
- `cfd_cases.py`: shared OpenFOAM case discovery helpers (imported by pipeline scripts).
- `omniverse_ui.py`: Omniverse Kit in-app RPM slider UI.
- `kit_extension/cfd_inference.py`: real-time inference Kit extension (Phase 3).
- `web_ui/`: Vite WebRTC dashboard for Omniverse streaming and omega control.
- `scripts/`: PowerShell helpers for rebuilding split artifact files.

## Setup

Python dependencies are pinned in `requirements.txt`.

```powershell
cd D:\Work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

USD export and colormap dependencies are optional:

```powershell
python -m pip install -r requirements-optional.txt
```

The web dashboard is a separate Vite app:

```powershell
cd D:\Work\web_ui
npm install
npm run dev
```

## Data Layout

The pipeline scripts expect case folders under `input/`:

```text
input/
  XSNNN-Incompressible-150RAD-SOLUTION_FIELDS/
    case.foam
    constant/MRFProperties
    ...
  XSNNN-Incompressible-250RAD-SOLUTION_FIELDS/
  XSNNN-Incompressible-350RAD-SOLUTION_FIELDS/
```

The rotation speed is read from `constant/MRFProperties` first, then from the
folder name as a fallback.

## Training

Default training output is `out_outputs/`. Visualization output remains `out/`.

**Step 1 — Export ground truth:**
```powershell
python 01_export_openfoam.py                 # → out_gt/
python 01_export_openfoam.py --no-vdb        # skip VDB
```

**Step 2 — Train surrogate (choose one):**
```powershell
python 02_train_mgn.py                       # MeshGraphNet → out_outputs/
python 02_train_fno.py                       # FNO (roadmap-preferred) → out_fno/
python 02_train_fno.py --grid-res 128 --pi-weight 1e-3
```

JSON config workflow:
```powershell
python 02_train_mgn.py --write-default-config train_config.json
python 02_train_mgn.py --config train_config.json
python 02_train_mgn.py --resume out_outputs/checkpoint_best.pt --epochs 500
```

**Step 3 — Export TorchScript:**
```powershell
python 03_export_model.py                    # MGN from out_outputs/
python 03_export_model.py --arch fno --out-dir out_fno
```

## Evaluation

Compare prediction arrays to OpenFOAM truth on the sampled cells:

```powershell
python evaluate.py
```

Default inputs:

- predictions: `out_outputs/predictions.npz`
- sampled indices: `out_outputs/sub_idx.npy`
- output JSON: `out_outputs/evaluation_metrics.json`

Custom paths:

```powershell
python evaluate.py --predictions out_outputs/predictions.npz --sub-idx out_outputs/sub_idx.npy
```

The report includes RMSE/MAE for scalar fields, component and magnitude metrics
for `U`, and negative prediction fractions for non-negative fields such as `k`,
`nut`, and `omega_turb`.

## Visualization

Render predicted streamlines:

```powershell
python 04_viz.py
```

Render OpenFOAM truth streamlines for comparison:

```powershell
python compare.py
```

Export animated streamline USD directly:

```powershell
python 04_export_streamlines.py
```

If `usd-core` is missing, install optional dependencies:

```powershell
python -m pip install -r requirements-optional.txt
```

## Split Artifacts

Large generated files may be split into `.partNN` files. Rebuild them with:

```powershell
.\scripts\rebuild_split_files.ps1
```

Or rebuild only `out_outputs/predictions.npz`:

```powershell
.\scripts\rebuild_predictions.ps1
```

## Web UI And Omniverse Contract

Run the Vite app:

```powershell
cd D:\Work\web_ui
npm run dev
```

The dashboard connects to Omniverse WebRTC on `127.0.0.1:49100` by default.
Override it with query parameters:

```text
http://localhost:5173/?host=192.168.1.20&port=49100
```

When connected, the rotor speed slider (in **RPM**) sends this custom message through
`AppStreamer.sendMessage`:

```json
{
  "event_type": "cfd.setOmega",
  "payload": {
    "rpm": 2387,
    "omega_rad_s": 249.9,
    "frame": 10,
    "omega_min_rad_s": 150,
    "omega_max_rad_s": 350,
    "frame_step_rad_s": 10
  }
}
```

`cfd_inference.py` accepts either `omega_rad_s` or `rpm` (converts automatically).
The local `omniverse_ui.py` RPM slider follows the same frame mapping:

```text
omega_rad_s = rpm * π / 30
frame = (omega_rad_s - 150) / 10
```

## Verification

Quick checks:

```powershell
python -m py_compile cfd_cases.py 01_export_openfoam.py 02_train_mgn.py 02_train_fno.py 03_export_model.py 04_viz.py 04_export_streamlines.py
cd web_ui
npm run build
```

## Notes

- Generated training files belong in `out_outputs/`.
- Rendered images and USD exports belong in `out/`.
- This README does not change how large artifacts are stored in git.
