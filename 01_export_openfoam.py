"""Export OpenFOAM ground-truth cases to .vtu / .vtp / .vdb formats.

Outputs per omega case:
  out_gt/omega_<ω>/internal_mesh.vtu   — full volumetric mesh (UnstructuredGrid, all fields)
  out_gt/omega_<ω>/surface.vtp         — extracted surface (PolyData, all fields)
  out_gt/omega_<ω>/U.vdb               — velocity vector grid  (requires pyopenvdb)
  out_gt/omega_<ω>/p.vdb               — pressure scalar grid  (requires pyopenvdb)

Usage:
    python 01_export_openfoam.py
    python 01_export_openfoam.py --input input --out out_gt
    python 01_export_openfoam.py --no-vdb        # skip OpenVDB export
    python 01_export_openfoam.py --grid-res 128  # VDB voxel grid resolution
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from cfd_cases import discover_cases, resolve_input_dir


FIELD_NAMES = ["U", "p", "k", "nut", "omega"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="input", help="OpenFOAM cases directory.")
    p.add_argument("--out", default="out_gt", help="Output directory.")
    p.add_argument("--no-vdb", action="store_true", help="Skip OpenVDB export.")
    p.add_argument(
        "--grid-res",
        type=int,
        default=128,
        help="Voxel grid resolution per axis for VDB interpolation.",
    )
    return p.parse_args()


def _wsl_path(win_path: Path) -> str:
    """Convert a Windows path to WSL /mnt/X/... notation."""
    s = str(win_path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def _foam_to_vtk(case_path: Path) -> None:
    """Run ``foamToVTK -latestTime`` on *case_path*.

    On Windows the command is forwarded to WSL via ``wsl bash -lc '...'``
    so that the OpenFOAM environment is sourced automatically through
    the WSL login shell (``-l`` flag).  On Linux/macOS it runs directly.
    """
    if platform.system() == "Windows":
        wsl_case = _wsl_path(case_path)
        cmd = ["wsl", "bash", "-lc",
               f"foamToVTK -latestTime -case '{wsl_case}'"]
    else:
        if not shutil.which("foamToVTK"):
            raise RuntimeError(
                "foamToVTK not found in PATH.  Source OpenFOAM env first:"
                "  source /opt/openfoam*/etc/bashrc"
            )
        cmd = ["foamToVTK", "-latestTime", "-case", str(case_path)]

    print(f"    foamToVTK {case_path.name} ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"foamToVTK failed (exit {r.returncode}):\n{r.stderr.strip()}"
        )


def _find_internal_vtu(vtk_dir: Path) -> Path:
    """Locate the internalMesh file produced by foamToVTK.

    OpenFOAM \u2265 v2006 / OF9 writes:
        VTK/internalMesh/internalMesh_<N>.vtu
    Older releases write:
        VTK/internalMesh_<N>.vtk
    """
    hits = sorted(vtk_dir.glob("internalMesh/internalMesh_*.vtu"))
    if hits:
        return hits[-1]
    hits = sorted(vtk_dir.glob("internalMesh_*.vtk"))
    if hits:
        return hits[-1]
    raise FileNotFoundError(
        f"foamToVTK output not found under {vtk_dir}\n"
        "Expected: internalMesh/internalMesh_N.vtu  or  internalMesh_N.vtk"
    )


def load_case(case_path: Path) -> pv.UnstructuredGrid:
    """Export via foamToVTK CLI, then read the resulting VTU with PyVista."""
    _foam_to_vtk(case_path)
    vtu = _find_internal_vtu(case_path / "VTK")
    print(f"    reading {vtu.relative_to(case_path)}")
    return pv.read(str(vtu))


def export_vtu(mesh: pv.UnstructuredGrid, out: Path) -> None:
    mesh.save(str(out))
    print(f"    .vtu  {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")


def export_vtp(mesh: pv.UnstructuredGrid, out: Path) -> None:
    surface = mesh.extract_surface()
    surface = surface.cell_data_to_point_data(pass_cell_data=False)
    surface.save(str(out))
    print(f"    .vtp  {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")


def export_vdb(
    mesh: pv.UnstructuredGrid,
    out_dir: Path,
    grid_res: int,
) -> None:
    try:
        import pyopenvdb as vdb
    except ImportError:
        print("    .vdb  SKIP — pyopenvdb not installed  (pip install pyopenvdb)")
        return

    bmin = np.asarray(mesh.bounds[::2])
    bmax = np.asarray(mesh.bounds[1::2])
    span = bmax - bmin

    # Interpolate cell data onto a regular ImageData grid.
    grid = pv.ImageData(
        dimensions=(grid_res, grid_res, grid_res),
        origin=tuple(bmin),
        spacing=tuple(span / (grid_res - 1)),
    )
    point_mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    interp = grid.interpolate(point_mesh, radius=float(span.max() / grid_res * 3))

    # Velocity vector VDB.
    if "U" in interp.point_data:
        u_arr = np.asarray(interp.point_data["U"], dtype=np.float32)
        u_arr = u_arr.reshape(grid_res, grid_res, grid_res, 3)
        vgrid = vdb.Vec3SGrid()
        vgrid.name = "U"
        vgrid.copyFromArray(u_arr)
        vdb_path = out_dir / "U.vdb"
        vdb.write(str(vdb_path), grids=[vgrid])
        print(f"    .vdb  U.vdb  ({vdb_path.stat().st_size / 1e6:.1f} MB)")

    # Pressure scalar VDB.
    if "p" in interp.point_data:
        p_arr = np.asarray(interp.point_data["p"], dtype=np.float32)
        p_arr = p_arr.reshape(grid_res, grid_res, grid_res)
        pgrid = vdb.FloatGrid()
        pgrid.name = "p"
        pgrid.copyFromArray(p_arr)
        vdb_path = out_dir / "p.vdb"
        vdb.write(str(vdb_path), grids=[pgrid])
        print(f"    .vdb  p.vdb  ({vdb_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    HERE = Path(__file__).parent
    INPUT = resolve_input_dir(args.input, base_dir=HERE)
    OUT = HERE / args.out
    OUT.mkdir(exist_ok=True)

    cases = discover_cases(INPUT)
    print(f"Found {len(cases)} case(s) in {INPUT}")

    for case in cases:
        t0 = time.time()
        omega_tag = f"omega_{case.omega:.0f}"
        case_out = OUT / omega_tag
        case_out.mkdir(exist_ok=True)

        print(f"\n[{omega_tag}]  {case.path.name}")
        mesh = load_case(case.path)
        n_cells = mesh.n_cells
        fields_present = [f for f in FIELD_NAMES if f in mesh.cell_data]
        print(f"  cells={n_cells:,}  fields={fields_present}  t_load={time.time()-t0:.1f}s")

        export_vtu(mesh, case_out / "internal_mesh.vtu")
        export_vtp(mesh, case_out / "surface.vtp")

        if not args.no_vdb:
            export_vdb(mesh, case_out, args.grid_res)

        print(f"  done  ({time.time()-t0:.1f}s total)")

    print(f"\nAll cases exported to {OUT.resolve()}")


if __name__ == "__main__":
    main()
