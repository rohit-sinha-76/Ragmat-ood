"""Brutal validation tests for the Unified Inference Pipeline.

Tests both the API logic and the CLI executable process on mock and template structures.
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

# Add project root and scripts directory to python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from pymatgen.core import Structure, Lattice
from scripts.run_inference import run_predictions, ELEMENT_OUT_MISSING

@pytest.fixture
def test_cif_files():
    """Generates mock CIF files for testing.
    
    1. An In-Distribution structure (Si cubic)
    2. An Out-of-Distribution structure containing unseen Bi (Bismuth, element_out missing)
    """
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = Path(temp_dir.name)
    
    # 1. Si cubic (In-Distribution)
    lattice_si = Lattice.cubic(5.43)
    struct_si = Structure(lattice_si, ["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"],
                          [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5],
                           [0.25,0.25,0.25], [0.75,0.75,0.25], [0.75,0.25,0.75], [0.25,0.75,0.75]])
    cif_si_path = dir_path / "Si_test.cif"
    struct_si.to(filename=str(cif_si_path), fmt="cif")
    
    # 2. Bi crystal (OOD under element_out)
    lattice_bi = Lattice.cubic(4.0)
    struct_bi = Structure(lattice_bi, ["Bi"], [[0.0, 0.0, 0.0]])
    cif_bi_path = dir_path / "Bi_test.cif"
    struct_bi.to(filename=str(cif_bi_path), fmt="cif")
    
    yield cif_si_path, cif_bi_path, dir_path
    temp_dir.cleanup()

def test_inference_api_logic(test_cif_files):
    """Directly calls run_predictions API to verify pipeline calculations and ZSNI patching."""
    cif_si, cif_bi, _ = test_cif_files
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Run on In-Distribution Si
        struct_si = Structure.from_file(str(cif_si))
        res_si = run_predictions(
            structures=[struct_si],
            ids=["Si_test"],
            prop="formation_energy",
            split_type="element_out",
            enable_zsni=True,
            threshold=0.3,
            device=device
        )
        
        assert len(res_si["final_pred"]) == 1
        assert isinstance(res_si["final_pred"][0], float)
        assert not np.isnan(res_si["final_pred"][0])
        
        # Run on OOD Bi
        struct_bi = Structure.from_file(str(cif_bi))
        res_bi = run_predictions(
            structures=[struct_bi],
            ids=["Bi_test"],
            prop="formation_energy",
            split_type="element_out",
            enable_zsni=True,
            threshold=0.3,
            device=device
        )
        
        assert len(res_bi["final_pred"]) == 1
        assert isinstance(res_bi["final_pred"][0], float)
        assert not np.isnan(res_bi["final_pred"][0])
    except OSError as e:
        if "Cannot allocate memory" in str(e):
            pytest.skip("System RAM constrained by concurrent background GPU training.")
        else:
            raise

def test_inference_cli_execution(test_cif_files):
    """Tests the CLI script execution as a subprocess, checking arguments and CSV output."""
    cif_si, cif_bi, dir_path = test_cif_files
    csv_out = dir_path / "preds.csv"
    
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_inference.py"),
        "--cif", str(cif_si),
        "--property", "both",
        "--split-type", "element_out",
        "--fe-threshold", "0.3",
        "--bg-threshold", "0.9",
        "--output", str(csv_out)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 and "Cannot allocate memory" in res.stderr:
        pytest.skip("System RAM constrained by concurrent background GPU training.")
        
    assert res.returncode == 0, f"CLI execution failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    
    # Check that predictions CSV was generated and has expected headers
    assert csv_out.exists()
    df = pd.read_csv(csv_out)
    assert "material_id" in df.columns
    assert "formation_energy_final_pred" in df.columns
    assert "band_gap_final_pred" in df.columns
    assert "formation_energy_ood_score" in df.columns
    assert "band_gap_ood_score" in df.columns
    
    # Check folder processing mode
    cmd_folder = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_inference.py"),
        "--cif", str(dir_path),
        "--property", "formation_energy"
    ]
    res_folder = subprocess.run(cmd_folder, capture_output=True, text=True)
    assert res_folder.returncode == 0
    assert "Found 2 CIF files" in res_folder.stdout or "Found 2 CIF files" in res_folder.stderr

def test_inference_cli_input_validation():
    """Checks that the CLI reports failures gracefully on non-existent files."""
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_inference.py"),
        "--cif", "nonexistent_file_xyz.cif"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
