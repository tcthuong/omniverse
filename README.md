# CFD MeshGraphNet Surrogate

This project trains and visualizes a CFD surrogate model for OpenFOAM rotor cases.
The current dataset contains three incompressible solution fields at 150, 250, and
350 rad/s. The training script learns on sampled OpenFOAM cells, generates a 21
frame omega sweep, and exports visualization assets for PyVista and Omniverse.

## Repository Map

- `input/`: OpenFOAM case folders with `case.foam` and solution fields.
- `train.py`: reads OpenFOAM cases, trains PhysicsNeMo MeshGraphNet, and writes predictions.
- `evaluate.py`: compares generated predictions with OpenFOAM truth on sampled cells.
- `viz.py`: renders predicted streamlines to PNG and exports animated USD.
- `compare.py`: renders OpenFOAM truth streamlines for visual comparison.
- `export_streamlines.py`: lower-level USD streamline exporter.
- `omniverse_ui.py`: Omniverse Kit in-app UI for timeline control.
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

`train.py` expects case folders under `input/`:

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

```powershell
cd D:\Work
python train.py
```

Useful overrides:

```powershell
python train.py --out out_outputs --subsample 300000 --epochs 3000
python train.py --validation-omega 250 --infer-frames 21
python train.py --device cuda
python train.py --no-zip
```

Write a default JSON config, edit it, then run from it:

```powershell
python train.py --write-default-config train_config.json
python train.py --config train_config.json
```

CLI arguments override values from `--config`.

Resume from an existing model checkpoint:

```powershell
python train.py --resume out_outputs/checkpoint_best.pt --epochs 500
```

`--resume` loads model weights from the checkpoint before training. Existing
checkpoints saved by this repo are raw PyTorch `state_dict` files and remain
compatible.

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
python viz.py
```

Render OpenFOAM truth streamlines for comparison:

```powershell
python compare.py
```

Export animated streamline USD directly:

```powershell
python export_streamlines.py
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

When connected, the rotor speed slider sends this custom message through
`AppStreamer.sendMessage`:

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

The streamed Kit app or Action Graph should handle `cfd.setOmega` and map
`payload.frame` to the USD timeline frame. The local `omniverse_ui.py` follows
the same mapping:

```text
frame = (omega_rad_s - 150) / 10
```

## Verification

Quick checks:

```powershell
python -m py_compile train.py evaluate.py viz.py export_streamlines.py compare.py omniverse_ui.py
cd web_ui
npm run build
```

## Notes

- Generated training files belong in `out_outputs/`.
- Rendered images and USD exports belong in `out/`.
- This README does not change how large artifacts are stored in git.
