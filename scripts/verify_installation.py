#!/usr/bin/env python
"""
verify_installation.py - Verify that the environment is set up correctly

This script checks that all required packages are installed with correct versions
and that the basic imports work. Run this after environment setup to ensure
everything is ready for model training and prediction.

Usage:
    python scripts/verify_installation.py
"""

import sys
import importlib.metadata
from typing import Tuple, List


def get_color_codes() -> Tuple[str, str, str, str]:
    """Get ANSI color codes (or empty strings if not supported)."""
    try:
        # Check if terminal supports colors
        if sys.stdout.isatty():
            GREEN = '\033[92m'
            RED = '\033[91m'
            YELLOW = '\033[93m'
            RESET = '\033[0m'
        else:
            GREEN = RED = YELLOW = RESET = ''
    except:
        GREEN = RED = YELLOW = RESET = ''

    return GREEN, RED, YELLOW, RESET


def check_python_version() -> bool:
    """Check if Python version is 3.10 or higher."""
    GREEN, RED, YELLOW, RESET = get_color_codes()

    print("Checking Python version...")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 10:
        print(f"  {GREEN}✓{RESET} Python {version_str} (meets requirement: 3.10+)")
        return True
    else:
        print(f"  {RED}✗{RESET} Python {version_str} (requires 3.10+)")
        return False


def check_package(package_name: str, min_version: str = None, import_name: str = None) -> bool:
    """
    Check if a package is installed and optionally verify version.

    Args:
        package_name: Package name as it appears in pip/conda
        min_version: Minimum required version (e.g., "2.0.0")
        import_name: Name to use for import (if different from package_name)

    Returns:
        True if package is installed and meets version requirement
    """
    GREEN, RED, YELLOW, RESET = get_color_codes()

    if import_name is None:
        import_name = package_name

    try:
        # Try to import the package
        module = __import__(import_name)

        # Get installed version
        try:
            installed_version = importlib.metadata.version(package_name)
        except:
            # Fallback for packages that don't expose version in metadata
            installed_version = getattr(module, '__version__', 'unknown')

        # Check version if required
        if min_version:
            try:
                from packaging import version
                if version.parse(installed_version) >= version.parse(min_version):
                    print(f"  {GREEN}✓{RESET} {package_name} {installed_version} (>= {min_version})")
                    return True
                else:
                    print(f"  {YELLOW}⚠{RESET} {package_name} {installed_version} (requires >= {min_version})")
                    return False
            except:
                # If packaging not available, just show installed version
                print(f"  {GREEN}✓{RESET} {package_name} {installed_version}")
                return True
        else:
            print(f"  {GREEN}✓{RESET} {package_name} {installed_version}")
            return True

    except ImportError:
        print(f"  {RED}✗{RESET} {package_name} not installed")
        return False


def check_cuda_availability() -> Tuple[bool, str]:
    """Check if CUDA/GPU is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
        else:
            return False, "CPU only"
    except:
        return False, "Unknown"


def main():
    """Main verification function."""
    GREEN, RED, YELLOW, RESET = get_color_codes()

    print("=" * 70)
    print("Verifying Installation for Probabilistic Forecasting of Imbalance Prices")
    print("=" * 70)
    print()

    all_checks_passed = True

    # Check Python version
    print("1. Python Version")
    print("-" * 70)
    if not check_python_version():
        all_checks_passed = False
    print()

    # Check core packages
    print("2. Core Scientific Computing Packages")
    print("-" * 70)
    core_packages = [
        ("numpy", "1.24.0"),
        ("pandas", "2.0.0"),
        ("scipy", "1.7.0"),
        ("scikit-learn", "1.0.0", "sklearn"),
    ]

    for pkg_info in core_packages:
        if len(pkg_info) == 3:
            pkg_name, min_ver, import_name = pkg_info
            if not check_package(pkg_name, min_ver, import_name):
                all_checks_passed = False
        else:
            pkg_name, min_ver = pkg_info
            if not check_package(pkg_name, min_ver):
                all_checks_passed = False
    print()

    # Check deep learning packages
    print("3. Deep Learning Packages")
    print("-" * 70)
    dl_packages = [
        ("torch", "2.0.0"),
        ("torchvision", "0.15.0"),
    ]

    for pkg_name, min_ver in dl_packages:
        if not check_package(pkg_name, min_ver):
            all_checks_passed = False

    # Check CUDA availability
    cuda_available, device_info = check_cuda_availability()
    if cuda_available:
        print(f"  {GREEN}✓{RESET} CUDA available: {device_info}")
    else:
        print(f"  {YELLOW}ℹ{RESET} CUDA not available (using CPU): {device_info}")
    print()

    # Check optimization packages
    print("4. Hyperparameter Optimization")
    print("-" * 70)
    if not check_package("optuna", "3.0.0"):
        all_checks_passed = False
    print()

    # Check data processing packages
    print("5. Data Processing & Storage")
    print("-" * 70)
    data_packages = [
        ("pyarrow", "10.0.0"),
        ("fastparquet", "2023.0.0"),
        ("joblib", "1.1.0"),
        ("pyyaml", "6.0", "yaml"),
    ]

    for pkg_info in data_packages:
        if len(pkg_info) == 3:
            pkg_name, min_ver, import_name = pkg_info
            if not check_package(pkg_name, min_ver, import_name):
                all_checks_passed = False
        else:
            pkg_name, min_ver = pkg_info
            if not check_package(pkg_name, min_ver):
                all_checks_passed = False
    print()

    # Check XGBoost
    print("6. Gradient Boosting")
    print("-" * 70)
    if not check_package("xgboost", "1.5.0"):
        all_checks_passed = False
    print()

    # Check visualization packages
    print("7. Visualization")
    print("-" * 70)
    viz_packages = [
        ("matplotlib", "3.5.0"),
        ("seaborn", "0.11.0"),
    ]

    for pkg_name, min_ver in viz_packages:
        if not check_package(pkg_name, min_ver):
            all_checks_passed = False
    print()

    # Check optional packages
    print("8. Optional Packages (Jupyter)")
    print("-" * 70)
    optional_packages = [
        ("jupyter",),
        ("jupyterlab",),
        ("ipykernel",),
    ]

    for pkg_info in optional_packages:
        pkg_name = pkg_info[0]
        check_package(pkg_name)  # Don't fail if optional packages missing
    print()

    # Test project imports
    print("9. Project Module Imports")
    print("-" * 70)
    try:
        sys.path.insert(0, '.')  # Add current directory to path
        from src.data.loader import DataProcessor
        print(f"  {GREEN}✓{RESET} src.data.loader.DataProcessor")

        from src.models.neural_nets import create_probabilistic_model
        print(f"  {GREEN}✓{RESET} src.models.neural_nets.create_probabilistic_model")

        from src.training.optuna_tuner import OptunaHPTuner
        print(f"  {GREEN}✓{RESET} src.training.optuna_tuner.OptunaHPTuner")

        from src.training.prediction import DailyMarketPredictor
        print(f"  {GREEN}✓{RESET} src.training.prediction.DailyMarketPredictor")

        print(f"  {GREEN}✓{RESET} All project modules imported successfully")
    except ImportError as e:
        print(f"  {RED}✗{RESET} Failed to import project modules: {e}")
        all_checks_passed = False
    print()

    # Final summary
    print("=" * 70)
    if all_checks_passed:
        print(f"{GREEN}✓ Installation verification PASSED{RESET}")
        print()
        print("Your environment is correctly set up!")
        print("You can now proceed with:")
        print("  1. Generating synthetic data: python src/data/synthetic_data.py --all-zones")
        print("  2. Running hyperparameter tuning: python run_tuning.py --zone no1 --distribution jsu --run-id test")
        print("  3. Generating predictions: python run_pred.py --zone no1 --distribution jsu --run-id test --start-date 2024-04-26 --end-date 2024-04-26")
        return 0
    else:
        print(f"{RED}✗ Installation verification FAILED{RESET}")
        print()
        print("Some packages are missing or have incorrect versions.")
        print("Please install missing packages:")
        print("  - Using conda: conda env create -f environment.yml")
        print("  - Using pip: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
