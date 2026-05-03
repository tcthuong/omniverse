# Setup Python environment for VTK to USD conversion (Windows)
# Usage: .\setup_env.ps1

Write-Host "=== Setting up Python environment for VTK to USD conversion ===" -ForegroundColor Green

# Check Python version
Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install required packages
Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Yellow
Write-Host "  - pyvista (VTK reader)"
Write-Host "  - usd-core (USD writer)"
Write-Host "  - numpy (array operations)"

python -m pip install pyvista==0.47.3 usd-core==26.5 numpy==2.4.4

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import pyvista; print('PyVista', pyvista.__version__)"
python -c "from pxr import Usd; print('USD (pxr) installed')"
python -c "import numpy; print('NumPy', numpy.__version__)"

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Cyan
Write-Host "  venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To convert VTK to USD:" -ForegroundColor Cyan
Write-Host "  python 05_vtk_to_usd.py input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/VTK/*.vtk --animated"
Write-Host ""
