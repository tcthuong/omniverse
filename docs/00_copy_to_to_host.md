Remote Host Setup
=================

Use this guide to copy the project from Windows to the SSH host, set up the
Python environment, install OpenFOAM, and run the OpenFOAM export step.


Config
------

Local project:

  D:\work

Remote host:

  root@124.197.18.144

Remote project:

  /root/work


1. Copy Project From Windows
----------------------------

Run these commands from Windows PowerShell, not from the SSH session.

If you are already inside the remote host prompt, exit first:

  exit

Create the remote folder:

  ssh root@124.197.18.144 "mkdir -p /root/work"

Copy the project:

  cd D:\work
  scp -r . root@124.197.18.144:/root/work/

Copy one changed file later:

  scp D:\work\01_export_openfoam.py root@124.197.18.144:/root/work/

Verify on the host:

  ssh root@124.197.18.144
  cd /root/work
  ls -la


2. Install System Packages
--------------------------

Run on the remote host:

  cd /root/work

  sudo apt update
  sudo apt install -y software-properties-common build-essential wget gnupg

This project should use Python 3.11 or newer. On Ubuntu 22.04, install Python
3.11 from deadsnakes:

  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv python3.11-dev


3. Create Clean Python Env
--------------------------

Run on the remote host:

  cd /root/work
  deactivate 2>/dev/null || true
  rm -rf venv

  python3.11 -m venv venv
  . venv/bin/activate

  python --version
  python -m pip install --upgrade pip setuptools wheel

The version check should print Python 3.11.x or newer.


4. Install Python Dependencies
------------------------------

Install PyTorch first:

  python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

Check the installed PyTorch CUDA build:

  python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

Install torch-scatter from the matching PyG wheel index:

  CUDA_TAG=$(python - <<'PY'
import torch
cuda = torch.version.cuda
print("cpu" if cuda is None else "cu" + cuda.replace(".", ""))
PY
)

  case "$CUDA_TAG" in
    cpu|cu126|cu128|cu130) ;;
    *) echo "Unsupported PyG CUDA tag: $CUDA_TAG"; exit 1 ;;
  esac

  python -m pip install torch-scatter==2.1.2 -f "https://data.pyg.org/whl/torch-2.11.0+${CUDA_TAG}.html"

Install the remaining required packages:

  python -m pip install numpy==2.4.4 scipy==1.17.1 pyvista==0.47.3 nvidia-physicsnemo==1.3.0 torch-geometric==2.7.0

Optional visualization/USD packages:

  python -m pip install matplotlib==3.10.9 usd-core==26.5

Optional VDB package (pyopenvdb has no pip wheel for Python 3.11):

  sudo apt install -y python3-openvdb libopenvdb-dev

Then symlink into the venv so the venv Python can find it:

  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  SYS_SITE=/usr/lib/python3/dist-packages
  ln -sf $SYS_SITE/pyopenvdb* $SITE/

If python3-openvdb is not available in apt, use conda instead:

  conda install -c conda-forge pyopenvdb

If all methods fail, skip VDB and run export with `--no-vdb`.


5. Verify Python Env
--------------------

Run on the remote host:

  cd /root/work
  . venv/bin/activate

  python -c "import torch, torch_scatter, torch_geometric, pyvista, physicsnemo, numpy, scipy; print('env OK')"
  python -m unittest tests.test_cfd_cases
  python -m py_compile cfd_cases.py 01_export_openfoam.py 02_train_mgn.py 02_train_fno.py 03_export_model.py 04_viz.py 04_export_streamlines.py compare.py evaluate.py


6. Install OpenFOAM
-------------------

See docs/01_install_openfoam.md for detailed instructions.

Quick install:

  sudo apt update
  wget -O - https://dl.openfoam.org/gpg.key | sudo tee /etc/apt/trusted.gpg.d/openfoam.asc
  sudo add-apt-repository -y http://dl.openfoam.org/ubuntu
  sudo apt update
  sudo apt install -y openfoam13

Enable OpenFOAM:

  grep -qxF '. /opt/openfoam13/etc/bashrc' ~/.bashrc || echo '. /opt/openfoam13/etc/bashrc' >> ~/.bashrc
  . /opt/openfoam13/etc/bashrc

Verify:

  foamRun -help
  foamToVTK -help


7. Export OpenFOAM Data
-----------------------

Run on the remote host:

  cd /root/work
  . venv/bin/activate
  . /opt/openfoam13/etc/bashrc

Export without VDB:

  python 01_export_openfoam.py --no-vdb

Export with VDB, only if pyopenvdb installed successfully:

  python 01_export_openfoam.py


Notes
-----

Do not run this command from the remote host:

  scp -r D:\work\* root@124.197.18.144:/root/work/

`D:\work` only exists on Windows. Run copy commands from Windows PowerShell.

Do not use `python -m pip install -r requirements.txt` as the first dependency
install on a clean Linux host. Install PyTorch first, then torch-scatter from
the matching PyG wheel index, then the remaining packages.

If `python 01_export_openfoam.py` fails with `Negative size passed to
PyUnicode_New`, run `python 01_export_openfoam.py --no-vdb`. VDB export is
optional and depends on pyopenvdb binary compatibility.

Useful Linux commands:

  pwd        Show current folder
  ls -la     Show all files
  clear      Clear terminal screen
