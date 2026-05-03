#!/usr/bin/env python3
"""
Convert VTK files to USD format for Omniverse visualization.

Usage:
    python 05_vtk_to_usd.py input/case_name/VTK/case_1000.vtk
    python 05_vtk_to_usd.py input/case_name/VTK/ --animated
"""

import argparse
import glob
import numpy as np
import pyvista as pv
from pathlib import Path
from pxr import Usd, UsdGeom, Sdf, Vt
import sys


def vtk_to_usd_single(vtk_path: Path, usd_path: Path, mesh_name: str = "Mesh"):
    """Convert a single VTK file to USD."""
    print(f"Reading VTK: {vtk_path}")
    mesh = pv.read(str(vtk_path))
    
    print(f"Creating USD stage: {usd_path}")
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    
    # Create World xform
    world = UsdGeom.Xform.Define(stage, "/World")
    
    # Create mesh
    usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/{mesh_name}")
    
    # Convert points
    points = mesh.points.astype(np.float32)
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    
    # Convert cells to faces (handle UnstructuredGrid)
    if hasattr(mesh, 'faces') and mesh.faces.size > 0:
        # PolyData
        faces_raw = mesh.faces
        face_vertex_counts = []
        face_vertex_indices = []
        i = 0
        while i < len(faces_raw):
            n = int(faces_raw[i])
            face_vertex_counts.append(n)
            face_vertex_indices.extend([int(x) for x in faces_raw[i+1:i+1+n]])
            i += n + 1
        usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))
        usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))
    elif hasattr(mesh, 'cells') and mesh.n_cells > 0:
        # UnstructuredGrid - extract surface
        surf = mesh.extract_surface(algorithm='dataset_surface')
        if hasattr(surf, 'faces') and surf.faces.size > 0:
            faces_raw = surf.faces
            face_vertex_counts = []
            face_vertex_indices = []
            i = 0
            while i < len(faces_raw):
                n = int(faces_raw[i])
                face_vertex_counts.append(n)
                face_vertex_indices.extend([int(x) for x in faces_raw[i+1:i+1+n]])
                i += n + 1
            usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))
            usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))
    
    # Add velocity field if present
    if "U" in mesh.array_names:
        velocity = mesh["U"].astype(np.float32)
        velocity_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "velocity", Sdf.ValueTypeNames.Vector3fArray, interpolation=UsdGeom.Tokens.vertex
        )
        velocity_pv.Set(Vt.Vec3fArray.FromNumpy(velocity))
        
        # Add velocity magnitude as color
        u_mag = np.linalg.norm(velocity, axis=1)
        u_norm = np.clip(u_mag / (u_mag.max() + 1e-8), 0, 1)
        
        # Simple color mapping: blue (low) -> red (high)
        colors = np.zeros((len(u_norm), 3), dtype=np.float32)
        colors[:, 0] = u_norm  # Red channel
        colors[:, 2] = 1 - u_norm  # Blue channel
        
        color_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, interpolation=UsdGeom.Tokens.vertex
        )
        color_pv.Set(Vt.Vec3fArray.FromNumpy(colors))
    
    # Add pressure field if present
    if "p" in mesh.array_names:
        pressure = mesh["p"].astype(np.float32)
        pressure_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "pressure", Sdf.ValueTypeNames.FloatArray, interpolation=UsdGeom.Tokens.vertex
        )
        pressure_pv.Set(Vt.FloatArray.FromNumpy(pressure))
    
    stage.GetRootLayer().Save()
    size_mb = usd_path.stat().st_size / 1e6
    print(f"Saved USD: {usd_path} ({size_mb:.1f} MB)")
    print(f"  Points: {len(points):,}")
    print(f"  Cells: {mesh.n_cells:,}")


def vtk_to_usd_animated(vtk_paths: list, usd_path: Path, mesh_name: str = "Mesh"):
    """Convert multiple VTK files to animated USD."""
    print(f"Creating animated USD with {len(vtk_paths)} frames")
    
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(vtk_paths) - 1)
    stage.SetTimeCodesPerSecond(10)
    
    # Create World xform
    world = UsdGeom.Xform.Define(stage, "/World")
    
    # Create mesh
    usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/{mesh_name}")
    
    points_attr = usd_mesh.CreatePointsAttr()
    counts_attr = usd_mesh.CreateFaceVertexCountsAttr()
    indices_attr = usd_mesh.CreateFaceVertexIndicesAttr()
    
    # Check if velocity field exists
    first_mesh = pv.read(str(vtk_paths[0]))
    has_velocity = "U" in first_mesh.array_names
    has_pressure = "p" in first_mesh.array_names
    
    if has_velocity:
        velocity_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "velocity", Sdf.ValueTypeNames.Vector3fArray, interpolation=UsdGeom.Tokens.vertex
        )
        color_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, interpolation=UsdGeom.Tokens.vertex
        )
    
    if has_pressure:
        pressure_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "pressure", Sdf.ValueTypeNames.FloatArray, interpolation=UsdGeom.Tokens.vertex
        )
    
    # Calculate global max velocity for consistent coloring
    global_vmax = 0
    if has_velocity:
        print("Calculating global velocity range...")
        for vtk_path in vtk_paths:
            mesh = pv.read(str(vtk_path))
            vmax = np.linalg.norm(mesh["U"], axis=1).max()
            if vmax > global_vmax:
                global_vmax = vmax
        print(f"  Global max velocity: {global_vmax:.3f}")
    
    # Process each frame
    for t, vtk_path in enumerate(vtk_paths):
        print(f"Processing frame {t+1}/{len(vtk_paths)}: {Path(vtk_path).name}")
        mesh = pv.read(str(vtk_path))
        
        # Points
        points = mesh.points.astype(np.float32)
        points_attr.Set(Vt.Vec3fArray.FromNumpy(points), time=t)
        
        # Faces (only set once if topology doesn't change)
        if t == 0:
            if hasattr(mesh, 'faces') and mesh.faces.size > 0:
                # PolyData
                faces_raw = mesh.faces
                face_vertex_counts = []
                face_vertex_indices = []
                i = 0
                while i < len(faces_raw):
                    n = int(faces_raw[i])
                    face_vertex_counts.append(n)
                    face_vertex_indices.extend([int(x) for x in faces_raw[i+1:i+1+n]])
                    i += n + 1
                counts_attr.Set(Vt.IntArray(face_vertex_counts))
                indices_attr.Set(Vt.IntArray(face_vertex_indices))
            elif hasattr(mesh, 'cells') and mesh.n_cells > 0:
                # UnstructuredGrid - extract surface
                surf = mesh.extract_surface(algorithm='dataset_surface')
                if hasattr(surf, 'faces') and surf.faces.size > 0:
                    faces_raw = surf.faces
                    face_vertex_counts = []
                    face_vertex_indices = []
                    i = 0
                    while i < len(faces_raw):
                        n = int(faces_raw[i])
                        face_vertex_counts.append(n)
                        face_vertex_indices.extend([int(x) for x in faces_raw[i+1:i+1+n]])
                        i += n + 1
                    counts_attr.Set(Vt.IntArray(face_vertex_counts))
                    indices_attr.Set(Vt.IntArray(face_vertex_indices))
        
        # Velocity
        if has_velocity:
            velocity = mesh["U"].astype(np.float32)
            velocity_pv.Set(Vt.Vec3fArray.FromNumpy(velocity), time=t)
            
            # Color by velocity magnitude
            u_mag = np.linalg.norm(velocity, axis=1)
            u_norm = np.clip(u_mag / (global_vmax + 1e-8), 0, 1)
            
            colors = np.zeros((len(u_norm), 3), dtype=np.float32)
            colors[:, 0] = u_norm  # Red
            colors[:, 2] = 1 - u_norm  # Blue
            
            color_pv.Set(Vt.Vec3fArray.FromNumpy(colors), time=t)
        
        # Pressure
        if has_pressure:
            pressure = mesh["p"].astype(np.float32)
            pressure_pv.Set(Vt.FloatArray.FromNumpy(pressure), time=t)
    
    stage.GetRootLayer().Save()
    size_mb = usd_path.stat().st_size / 1e6
    print(f"Saved animated USD: {usd_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VTK files to USD format for Omniverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single VTK file
  python 05_vtk_to_usd.py input/case/VTK/case_1000.vtk
  
  # Directory (all VTK files)
  python 05_vtk_to_usd.py input/case/VTK/ --animated
  
  # Specify output path
  python 05_vtk_to_usd.py input/case/VTK/ --animated --output out/my_case.usda
        """
    )
    
    parser.add_argument(
        "input",
        help="VTK file or directory containing VTK files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output USD file path (default: out/<input_name>.usda)"
    )
    parser.add_argument(
        "-a", "--animated",
        action="store_true",
        help="Create animated USD from multiple VTK files"
    )
    parser.add_argument(
        "-n", "--name",
        default="Mesh",
        help="Name for the mesh in USD (default: Mesh)"
    )
    
    args = parser.parse_args()
    
    # Collect VTK files
    input_path = Path(args.input)
    vtk_files = []
    
    if input_path.is_dir():
        # Directory: find all .vtk files
        vtk_files = sorted(input_path.glob("*.vtk"))
    elif input_path.exists() and input_path.suffix.lower() == ".vtk":
        # Single file
        vtk_files = [input_path]
    else:
        # Try glob pattern (for wildcards)
        matches = glob.glob(args.input)
        vtk_files = [Path(m) for m in matches if Path(m).suffix.lower() == ".vtk"]
    
    if not vtk_files:
        print(f"Error: No VTK files found in: {args.input}")
        sys.exit(1)
    
    vtk_files = sorted(set(vtk_files))
    print(f"Found {len(vtk_files)} VTK file(s):")
    for vf in vtk_files:
        print(f"  - {vf}")
    
    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True)
        base_name = vtk_files[0].stem
        if args.animated and len(vtk_files) > 1:
            base_name += "_anim"
        out_path = out_dir / f"{base_name}.usda"
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    if args.animated and len(vtk_files) > 1:
        vtk_to_usd_animated(vtk_files, out_path, args.name)
    elif len(vtk_files) == 1:
        vtk_to_usd_single(vtk_files[0], out_path, args.name)
    else:
        print("Error: Multiple VTK files found but --animated not specified")
        print("Use --animated flag to create animated USD")
        sys.exit(1)


if __name__ == "__main__":
    main()
