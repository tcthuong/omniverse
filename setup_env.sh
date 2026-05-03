#!/bin/bash
# Setup Python environment for VTK to USD conversion
# Usage: bash setup_env.sh

set -e

echo "=== Setting up Python environment for VTK to USD conversion ==="

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3.11 -m venv venv || python3 -m venv venv || python -m venv venv
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install required packages for VTK to USD conversion
echo ""
echo "Installing required packages..."
echo "  - pyvista (VTK reader)"
echo "  - usd-core (USD writer)"
echo "  - numpy (array operations)"

python -m pip install pyvista==0.47.3 usd-core==26.5 numpy==2.4.4

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import pyvista; print(f'✓ PyVista {pyvista.__version__}')"
python -c "from pxr import Usd; print('✓ USD (pxr) installed')"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To convert VTK to USD:"
echo "  python 05_vtk_to_usd.py input/Fannn___Copy-Incompressible-6000-SOLUTION_FIELDS/VTK/*.vtk --animated"
echo ""
