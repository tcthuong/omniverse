"""Export OpenFOAM ground-truth cases to VTU/VTP and optional VDB files.

Outputs per omega case:
  out_gt/omega_<omega>/internal_mesh.vtu  full volumetric mesh
  out_gt/omega_<omega>/surface.vtp        extracted surface
  out_gt/omega_<omega>/U.vdb              velocity grid, optional pyopenvdb
  out_gt/omega_<omega>/p.vdb              pressure grid, optional pyopenvdb

Usage:
    python 01_export_openfoam.py --no-vdb
    python 01_export_openfoam.py --input input --out out_gt --no-vdb
    python 01_export_openfoam.py --grid-res 128
"""
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="input", help="OpenFOAM cases directory.")
    parser.add_argument("--out", default="out_gt", help="Output directory.")
    parser.add_argument("--no-vdb", action="store_true", help="Skip OpenVDB export.")
    parser.add_argument(
        "--grid-res",
        type=int,
        default=128,
        help="Voxel grid resolution per axis for VDB interpolation.",
    )
    return parser.parse_args()


def _wsl_path(win_path: Path) -> str:
    """Convert a Windows path to WSL /mnt/X/... notation."""
    text = str(win_path).replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = f"/mnt/{text[0].lower()}{text[2:]}"
    return text


def _foam_to_vtk(case_path: Path) -> None:
    """Run foamToVTK on a case directory."""
    if platform.system() == "Windows":
        wsl_case = _wsl_path(case_path)
        cmd = ["wsl", "bash", "-lc", f"foamToVTK -latestTime -case '{wsl_case}'"]
    else:
        if not shutil.which("foamToVTK"):
            raise RuntimeError(
                "foamToVTK not found in PATH. Source OpenFOAM first, e.g. "
                ". /opt/openfoam13/etc/bashrc"
            )
        cmd = ["foamToVTK", "-latestTime", "-case", str(case_path)]

    print(f"    foamToVTK {case_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        details = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        raise RuntimeError(f"foamToVTK failed (exit {result.returncode}):\n{details}")


def _find_internal_vtu(vtk_dir: Path) -> Path:
    """Locate the internal volume mesh produced by foamToVTK."""
    # OpenFOAM-13 writes the internal mesh as <caseName>_<time>.vtk at the
    # VTK root rather than under an internalMesh/ subdirectory.
    case_stem = vtk_dir.parent.name
    patterns = [
        "internalMesh/internalMesh_*.vtu",
        "*/internal.vtu",
        "**/internal.vtu",
        "**/internalMesh_*.vtu",
        "internalMesh_*.vtk",
        "**/internalMesh_*.vtk",
        f"{case_stem}_*.vtk",
    ]
    for pattern in patterns:
        hits = sorted(vtk_dir.glob(pattern))
        if hits:
            return hits[-1]

    all_vtk = sorted(
        path.relative_to(vtk_dir).as_posix()
        for path in vtk_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".vtu", ".vtk", ".vtm"}
    )
    found = "\nFound VTK files:\n  " + "\n  ".join(all_vtk[:30]) if all_vtk else ""
    raise FileNotFoundError(
        f"foamToVTK output not found under {vtk_dir}\n"
        "Expected an internal volume mesh file such as:\n"
        "  internalMesh/internalMesh_N.vtu\n"
        "  <case>_<time>/internal.vtu\n"
        "  internalMesh_N.vtk\n"
        f"  {case_stem}_<time>.vtk"
        f"{found}"
    )


def load_case(case_path: Path) -> pv.UnstructuredGrid:
    """Export via foamToVTK, then read the internal mesh with PyVista."""
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


def export_vdb(mesh: pv.UnstructuredGrid, out_dir: Path, grid_res: int) -> None:
    """Export optional VDB grids.

    pyopenvdb is not required for the project pipeline. If it is missing or its
    binary wheel is incompatible with the host Python, skip VDB and keep VTU/VTP.
    """
    try:
        import pyopenvdb as vdb
    except Exception as exc:
        print(
            "    .vdb  SKIP - pyopenvdb unavailable "
            f"({type(exc).__name__}: {exc})"
        )
        return

    bmin = np.asarray(mesh.bounds[::2])
    bmax = np.asarray(mesh.bounds[1::2])
    span = bmax - bmin

    grid = pv.ImageData(
        dimensions=(grid_res, grid_res, grid_res),
        origin=tuple(bmin),
        spacing=tuple(span / (grid_res - 1)),
    )
    point_mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    interp = grid.interpolate(point_mesh, radius=float(span.max() / grid_res * 3))

    if "U" in interp.point_data:
        u_arr = np.asarray(interp.point_data["U"], dtype=np.float32)
        u_arr = u_arr.reshape(grid_res, grid_res, grid_res, 3)
        u_grid = vdb.Vec3SGrid()
        u_grid.name = "U"
        u_grid.copyFromArray(u_arr)
        vdb_path = out_dir / "U.vdb"
        vdb.write(str(vdb_path), grids=[u_grid])
        print(f"    .vdb  U.vdb  ({vdb_path.stat().st_size / 1e6:.1f} MB)")

    if "p" in interp.point_data:
        p_arr = np.asarray(interp.point_data["p"], dtype=np.float32)
        p_arr = p_arr.reshape(grid_res, grid_res, grid_res)
        p_grid = vdb.FloatGrid()
        p_grid.name = "p"
        p_grid.copyFromArray(p_arr)
        vdb_path = out_dir / "p.vdb"
        vdb.write(str(vdb_path), grids=[p_grid])
        print(f"    .vdb  p.vdb  ({vdb_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    here = Path(__file__).parent
    input_dir = resolve_input_dir(args.input, base_dir=here)
    out_dir = here / args.out
    out_dir.mkdir(exist_ok=True)

    cases = discover_cases(input_dir)
    print(f"Found {len(cases)} case(s) in {input_dir}")

    for case in cases:
        t0 = time.time()
        omega_tag = f"omega_{case.omega:.0f}"
        case_out = out_dir / omega_tag
        case_out.mkdir(exist_ok=True)

        print(f"\n[{omega_tag}]  {case.path.name}")
        mesh = load_case(case.path)
        fields_present = [field for field in FIELD_NAMES if field in mesh.cell_data]
        print(
            f"  cells={mesh.n_cells:,}  fields={fields_present}  "
            f"t_load={time.time() - t0:.1f}s"
        )

        export_vtu(mesh, case_out / "internal_mesh.vtu")
        export_vtp(mesh, case_out / "surface.vtp")

        meta = {
            "omega": case.omega,
            "cells": mesh.n_cells,
            "fields": fields_present,
            "source": case.path.name,
        }
        (case_out / "meta.json").write_text(json.dumps(meta, indent=2))

        if not args.no_vdb:
            try:
                export_vdb(mesh, case_out, args.grid_res)
            except Exception as exc:
                print(
                    "    .vdb  SKIP - VDB export failed "
                    f"({type(exc).__name__}: {exc})"
                )

        print(f"  done  ({time.time() - t0:.1f}s total)")

    print(f"\nAll cases exported to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
