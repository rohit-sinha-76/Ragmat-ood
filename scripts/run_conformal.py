import json
import logging
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ragmat.encoders.cgcnn import CGCNNEncoder
from ragmat.uncertainty.conformal import ConformalPredictor
from ragmat.encoders.graph_builder import _ELEMENTS

# ZSNI helpers (inlined — the original run_imputation_rescue.py was a diagnostic
# scratch script; the minimal logic needed here is reproduced directly).

def find_seen_elements(train_graphs) -> set:
    """Return the set of element indices that appear in at least one training graph."""
    seen = set()
    for g in train_graphs:
        # Node features are one-hot over 92 elements; argmax recovers the element index.
        seen.update(g.x.argmax(dim=1).tolist())
    return seen


def get_element_features() -> dict:
    """Return a dict {element_index: 2D periodic-table coordinate (row, group)}
    for use in ZSNI nearest-neighbour imputation."""
    # Periodic table row/group for elements H(0) -> U(91)
    # Source: standard periodic table positions
    PT = [
        (1,1),(1,18),                                                           # H, He
        (2,1),(2,2),(2,13),(2,14),(2,15),(2,16),(2,17),(2,18),                  # Li-Ne
        (3,1),(3,2),(3,13),(3,14),(3,15),(3,16),(3,17),(3,18),                  # Na-Ar
        (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),           # K-Ni
        (4,11),(4,12),(4,13),(4,14),(4,15),(4,16),(4,17),(4,18),                # Cu-Kr
        (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),(5,10),           # Rb-Pd
        (5,11),(5,12),(5,13),(5,14),(5,15),(5,16),(5,17),(5,18),                # Ag-Xe
        (6,1),(6,2),                                                            # Cs, Ba
        (8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),         # La-Gd
        (8,13),(8,14),(8,15),(8,16),(8,17),                                     # Tb-Lu
        (6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),                      # Hf-Au
        (6,12),(6,13),(6,14),(6,15),(6,16),(6,17),(6,18),                       # Hg-Rn
        (7,1),(7,2),                                                            # Fr, Ra
        (9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),(9,12),         # Ac-Cf
    ]
    # Pad to 92 entries if needed
    while len(PT) < 92:
        PT.append((0, 0))
    return {i: np.array(PT[i], dtype=float) for i in range(len(PT))}


def load_cached_graphs(prop: str, split: str, partition: str):
    """Load pre-built graph objects from disk cache."""
    from ragmat.encoders.graph_builder import CrystalGraphBuilder
    import json
    graphs_dir = _PROJECT_ROOT / "data" / "graphs" / prop / split
    cache_file = graphs_dir / f"{partition}_graphs.pt"
    if cache_file.exists():
        return torch.load(cache_file, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"Graph cache not found: {cache_file}. Run run_phase6.py first.")

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def manual_calibrate_and_predict(enc, scaler, val_graphs, test_graphs):
    # evaluate val set
    val_preds, val_trues = [], []
    val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out, _ = enc(batch)
            pred = scaler.inverse_transform(out.cpu().numpy())
            val_preds.append(pred)
            val_trues.append(batch.y.view(-1, 1).cpu().numpy())
    
    val_preds = np.vstack(val_preds).reshape(-1)
    val_trues = np.vstack(val_trues).reshape(-1)
    
    scores = np.abs(val_trues - val_preds)
    n_cal = len(scores)
    level = (1.0 - 0.1) * (1.0 + 1.0 / n_cal)
    half_width = float(np.quantile(scores, min(level, 1.0)))

    # evaluate test set
    test_preds, test_trues = [], []
    test_loader = DataLoader(test_graphs, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out, _ = enc(batch)
            pred = scaler.inverse_transform(out.cpu().numpy())
            test_preds.append(pred)
            test_trues.append(batch.y.view(-1, 1).cpu().numpy())
    
    test_preds = np.vstack(test_preds).reshape(-1)
    test_trues = np.vstack(test_trues).reshape(-1)
    
    lb = test_preds - half_width
    ub = test_preds + half_width
    cov = np.mean((test_trues >= lb) & (test_trues <= ub))
    mae = mean_absolute_error(test_trues, test_preds)
    
    return cov, half_width * 2, mae

def main():
    prop = "formation_energy"
    print(f"\nEvaluating Conformal Calibration for {prop} (element_out)")

    # Load graphs (instantly from disk cache)
    val_graphs = load_cached_graphs(prop, "element_out", "val")
    test_graphs = load_cached_graphs(prop, "element_out", "test")

    # Load Base Model
    model_path = Path("final_result/checkpoints") / f"tier1_{prop}_element_out_base_best.pt"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    enc = CGCNNEncoder().to(device)
    enc.load_state_dict(ckpt["model_state_dict"])
    scaler = pickle.loads(ckpt["y_scaler"])
    enc.eval()

    print("\n--- CGCNN Broken ---")
    cg_cov, cg_width, cg_mae = manual_calibrate_and_predict(enc, scaler, val_graphs, test_graphs)
    print(f"CGCNN - Cov: {cg_cov:.3f}, Width: {cg_width:.3f}, MAE: {cg_mae:.3f}")

    print("\n--- ZSNI (k=2) ---")
    train_graphs = load_cached_graphs(prop, "element_out", "train")
    seen = find_seen_elements(train_graphs)
    missing = [i for i in range(len(_ELEMENTS)) if i not in seen]
    
    feats = get_element_features()
    with torch.no_grad():
        weight = enc.embedding.weight.data
        for m in missing:
            m_feat = feats[m]
            dists = [(np.linalg.norm(m_feat - feats[s]), s) for s in seen]
            dists.sort()
            nearest = [s for _, s in dists[:2]]
            weight[:, m] = torch.stack([weight[:, s] for s in nearest]).mean(dim=0)

    z_cov, z_width, z_mae = manual_calibrate_and_predict(enc, scaler, val_graphs, test_graphs)
    print(f"ZSNI - Cov: {z_cov:.3f}, Width: {z_width:.3f}, MAE: {z_mae:.3f}")

    results = {
        "CGCNN": {"coverage": cg_cov, "width": cg_width, "mae": cg_mae},
        "ZSNI": {"coverage": z_cov, "width": z_width, "mae": z_mae},
    }
    out_path = _PROJECT_ROOT / "final_result" / "conformal_zsni_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
