Chuyển đổi OpenFOAM sang VTK
============================

Hướng dẫn chuyển đổi case OpenFOAM sang định dạng VTK để visualization.


Yêu cầu
-------

- OpenFOAM đã được cài đặt (xem `01_install_openfoam.md`)
- OpenFOAM đã được kích hoạt: `. /opt/openfoam13/etc/bashrc`


Chuyển đổi một case đơn lẻ
--------------------------

```bash
# Di chuyển vào thư mục case OpenFOAM
cd input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS

# Kích hoạt OpenFOAM (nếu chưa)
. /opt/openfoam13/etc/bashrc

# Chuyển đổi sang VTK
foamToVTK

# Kiểm tra output
ls -la VTK/
```

VTK files sẽ được tạo trong folder `VTK/` bên trong case.


Chuyển đổi tất cả các case
---------------------------

Để chuyển đổi tất cả các case trong folder `input`:

```bash
cd input

# Chuyển đổi từng case
for case_dir in */; do
    echo "Converting $case_dir"
    cd "$case_dir"
    foamToVTK
    cd ..
done
```


Chuyển đổi với options
----------------------

```bash
# Chỉ chuyển đổi timestep cuối cùng
foamToVTK -latestTime

# Chỉ chuyển đổi các field cụ thể
foamToVTK -fields '(U p)'

# Chuyển đổi với ASCII format (thay vì binary)
foamToVTK -ascii

# Xem tất cả options
foamToVTK -help
```


Cấu trúc output
---------------

Sau khi chuyển đổi, cấu trúc thư mục sẽ như sau:

```
input/
└── Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/
    ├── 0/
    ├── 1000/
    ├── constant/
    ├── system/
    └── VTK/                    # Folder mới được tạo
        ├── Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS_0.vtk
        ├── Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS_1000.vtk
        └── ...
```


Lưu ý
-----

- VTK files có thể rất lớn, đảm bảo đủ dung lượng disk
- Sử dụng `-latestTime` để tiết kiệm thời gian và dung lượng
- VTK files có thể mở bằng ParaView, VisIt, hoặc PyVista
