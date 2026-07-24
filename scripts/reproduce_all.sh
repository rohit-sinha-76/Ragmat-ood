#!/bin/bash
# reproduce_all.sh -- Reproduce all key results from RAGMat-OOD paper
#
# Requirements:
#   - conda with 'ragmat' environment (see environment.yml)
#   - JARVIS-DFT data (downloaded automatically on first run)
#   - ~50 GB free disk space (for graph cache)
#   - 64 GB RAM recommended (graph loading is RAM-intensive)
#
# Expected total runtime:
#   - First run (builds graph cache): ~2-4 hours
#   - Subsequent runs (graphs cached): ~40 min CPU / ~10 min GPU

set -e

CONDA_BASE="${HOME}/miniforge3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ragmat

echo "======================================================"
echo " RAGMat-OOD Result Reproduction"
echo " $(date)"
echo "======================================================"

echo ""
echo "=== Step 1: Bootstrap CIs on Tier-0 predictions ==="
echo "    Expected runtime: ~1 min"
python scripts/run_bootstrap_cis.py
echo "    Done. Output: final_result/bootstrap_cis_report.md"

echo ""
echo "=== Step 2: Gating analysis (CGCNN + Mahalanobis OOD) ==="
echo "    Expected runtime: 20-40 min CPU / 5-10 min GPU"
python scripts/run_gating_analysis.py
echo "    Done. Output: final_result/gating_final_report.md"

echo ""
echo "=== Step 3: Physical interpretability ==="
echo "    Expected runtime: <1 min"
python scripts/run_interpretability.py
echo "    Done. Output: final_result/interpretability_report.md"

echo ""
echo "======================================================"
echo " Reproduction complete."
echo " Key outputs:"
echo "   final_result/bootstrap_cis_report.md    (Table 2 in paper)"
echo "   final_result/gating_final_report.md     (Tables 3-4 in paper)"
echo "   final_result/interpretability_report.md (Table 5 in paper)"
echo "======================================================"
