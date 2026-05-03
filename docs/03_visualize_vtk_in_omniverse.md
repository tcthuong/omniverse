Hiển thị VTK trong Omniverse
============================

Hướng dẫn chuyển đổi VTK sang USD và hiển thị trong NVIDIA Omniverse.


Tổng quan workflow
------------------

```
OpenFOAM → VTK → USD → Omniverse Viewer
```

1. Chuyển OpenFOAM sang VTK (xem `02_convert_openfoam_to_vtk.md`)
2. Chuyển VTK sang USD format
3. Load USD vào Omniverse Viewer


Yêu cầu
-------

- Python environment với PyVista và usd-core
- NVIDIA Omniverse Kit hoặc USD Viewer
- VTK files đã được tạo từ OpenFOAM


Bước 1: Cài đặt Python packages
--------------------------------

```bash
# Kích hoạt Python venv
. venv/bin/activate

# Cài đặt packages cần thiết
python -m pip install pyvista==0.47.3 usd-core==26.5
```


Bước 2: Chuyển đổi VTK sang USD
--------------------------------

### Option A: Sử dụng script có sẵn

```bash
# Kích hoạt Python venv
. venv/bin/activate

# Convert single VTK file
python 05_vtk_to_usd.py input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/VTK/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS_1000.vtk

# Convert multiple VTK files thành animation
python 05_vtk_to_usd.py input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/VTK/*.vtk --animated

# Specify output path
python 05_vtk_to_usd.py input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/VTK/*.vtk --animated -o out/fan_anim.usda
```

### Option B: Export streamlines (script khác)

```bash
# Export streamlines sang USD (từ predictions)
python 04_export_streamlines.py
```

Output sẽ được tạo trong folder `out/`:

```
out/
├── fan_anim.usda              # Từ 05_vtk_to_usd.py
├── streamlines_anim.usda      # Từ 04_export_streamlines.py
└── ...
```


Bước 3: Mở USD trong Omniverse
-------------------------------

### Option 1: Desktop USD Viewer

Trên Windows, chạy Omniverse Kit USD Viewer:

```powershell
$build = "D:\nvi\kit-app-template\_build\windows-x86_64\release"
& "$build\kit\kit.exe" "$build\apps\my_company.my_usd_viewer.kit" `
  "--/app/auto_load_usd=D:/Work/out/streamlines_anim.usda" `
  "--/renderer/multiGpu/enabled=false" `
  "--/rtx/verifyDriverVersion/enabled=false"
```

### Option 2: Omniverse Streaming (WebRTC)

Start Web UI:

```powershell
cd D:\Work\web_ui
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Start streaming USD Viewer:

```powershell
$build = "D:\nvi\kit-app-template\_build\windows-x86_64\release"
$kit = Join-Path $build "kit\kit.exe"
$app = Join-Path $build "apps\my_company.my_usd_viewer_streaming.kit"

& "$kit" "$app" --no-window `
  "--/app/auto_load_usd=D:/Work/out/streamlines_anim.usda" `
  "--/renderer/multiGpu/enabled=false" `
  "--/app/window/width=1280" "--/app/window/height=720" `
  "--/exts/omni.kit.livestream.app/primaryStream/targetFps=30" `
  "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100" `
  "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998" `
  "--/exts/omni.kit.livestream.app/primaryStream/publicIp=127.0.0.1"
```

Mở browser:

```
http://127.0.0.1:5173/?host=127.0.0.1&port=49100&mediaPort=47998&width=1280&height=720&fps=30
```

### Option 3: Omniverse USD Composer

1. Mở NVIDIA Omniverse Launcher
2. Launch USD Composer (hoặc Create)
3. File → Open → chọn file `.usda`
4. Viewport sẽ hiển thị 3D visualization


Bước 4: Copy USD từ Linux sang Windows
---------------------------------------

Nếu USD được tạo trên Linux remote host, copy về Windows:

```powershell
# Chạy từ Windows PowerShell
scp root@124.197.18.144:/root/work/out/streamlines_anim.usda D:\Work\out\
```


Troubleshooting
---------------

### VTK file quá lớn

Nếu VTK file quá lớn, giảm kích thước:

```bash
# Chỉ export timestep cuối
foamToVTK -latestTime

# Hoặc chỉ export các field cần thiết
foamToVTK -fields '(U p)'
```

### Streaming không hoạt động

Kiểm tra:

- NVIDIA GPU có active display
- Driver version (khuyến nghị R595 595.97+)
- Logs tại: `C:\Users\<user>\.nvidia-omniverse\logs\Kit\`

Xem chi tiết tại `docs/omniverse_usd_viewer_research.md`

### USD file không load

Kiểm tra:

```bash
# Verify USD file
python -c "from pxr import Usd; stage = Usd.Stage.Open('out/file.usda'); print(stage)"
```


Scripts có sẵn trong project
-----------------------------

- `05_vtk_to_usd.py` - Convert VTK files sang USD (script mới)
- `04_export_streamlines.py` - Export streamlines từ predictions sang USD
- `04_viz.py` - Visualization với PyVista
- `compare.py` - So sánh CFD results

Ví dụ thực tế:

```bash
# Convert VTK sang USD
python 05_vtk_to_usd.py input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/VTK/*.vtk --animated -o out/fan.usda

# Hoặc case khác
python 05_vtk_to_usd.py input/XSNNN-Incompressible-150RAD-SOLUTION_FIELDS/VTK/*.vtk --animated -o out/xsnnn.usda
```


Tham khảo
---------

- USD Viewer template: https://github.com/NVIDIA-Omniverse/kit-app-template
- PyVista docs: https://docs.pyvista.org/
- USD docs: https://openusd.org/
- Omniverse docs: https://docs.omniverse.nvidia.com/
