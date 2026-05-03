Cài đặt OpenFOAM CLI
====================

Hướng dẫn cài đặt OpenFOAM 13 trên Ubuntu/Debian Linux.


Bước 1: Kiểm tra hệ điều hành
-----------------------------

```bash
cat /etc/os-release
```


Bước 2: Cài đặt OpenFOAM 13
---------------------------

Chạy các lệnh sau:

```bash
# Cập nhật apt
sudo apt update

# Thêm GPG key của OpenFOAM
wget -O - https://dl.openfoam.org/gpg.key | sudo tee /etc/apt/trusted.gpg.d/openfoam.asc

# Thêm repository OpenFOAM
sudo add-apt-repository -y http://dl.openfoam.org/ubuntu

# Cập nhật lại apt
sudo apt update

# Cài đặt OpenFOAM 13
sudo apt install -y openfoam13
```


Bước 3: Kích hoạt OpenFOAM
--------------------------

Thêm OpenFOAM vào bashrc để tự động kích hoạt mỗi khi mở terminal:

```bash
grep -qxF '. /opt/openfoam13/etc/bashrc' ~/.bashrc || echo '. /opt/openfoam13/etc/bashrc' >> ~/.bashrc
```

Kích hoạt OpenFOAM cho session hiện tại:

```bash
. /opt/openfoam13/etc/bashrc
```


Bước 4: Kiểm tra cài đặt
------------------------

```bash
foamRun -help
foamToVTK -help
```

Nếu các lệnh trên hiển thị help, OpenFOAM đã được cài đặt thành công.


Lưu ý
-----

- Hướng dẫn này dành cho Ubuntu/Debian Linux
- OpenFOAM sẽ được cài đặt tại `/opt/openfoam13/`
- Tất cả lệnh cài đặt hệ thống cần `sudo`
- Mỗi khi mở terminal mới, OpenFOAM sẽ tự động được kích hoạt
- Nếu cần sử dụng ngay, chạy: `. /opt/openfoam13/etc/bashrc`
