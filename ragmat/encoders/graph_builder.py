"""Crystal graph builder: pymatgen Structure → PyTorch Geometric Data.

Converts a pymatgen Structure to a graph representation suitable for CGCNN:
- Node features: element one-hot encoding, dim=92 (H to U)
- Edge index: bidirectional edges for all atom pairs within cutoff_radius
- Edge attributes: Gaussian-smeared interatomic distances, dim=n_gaussian_basis
- Label y: target property value as a scalar tensor
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Elements H (Z=1) to U (Z=92) — one-hot encoding basis
_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
]
_ELEMENT_INDEX = {el: i for i, el in enumerate(_ELEMENTS)}
NODE_DIM = len(_ELEMENTS)  # 92


class CrystalGraphBuilder:
    """Convert pymatgen Structures to PyG Data objects for CGCNN.

    Args:
        cutoff_radius: Radial cutoff for neighbour search in Ångströms.
        n_gaussian_basis: Number of Gaussian basis functions for edge features.
        gaussian_min: Minimum of Gaussian centres.
        gaussian_max: Maximum of Gaussian centres.
    """

    def __init__(
        self,
        cutoff_radius: float = 8.0,
        n_gaussian_basis: int = 40,
        gaussian_min: float = 0.0,
        gaussian_max: float = 8.0,
    ) -> None:
        self.cutoff_radius = cutoff_radius
        self.n_gaussian_basis = n_gaussian_basis
        self._centers = torch.linspace(gaussian_min, gaussian_max, n_gaussian_basis)
        self._width = (gaussian_max - gaussian_min) / n_gaussian_basis

    # ── Public API ──────────────────────────────────────────────────────────

    def structure_to_graph(
        self,
        structure: Structure,
        y: float,
        material_id: str,
    ) -> Data:
        """Convert a single Structure to a PyG Data object.

        Args:
            structure: pymatgen Structure.
            y: Target property value.
            material_id: Material identifier (stored as ``data.material_id``).

        Returns:
            ``torch_geometric.data.Data`` with fields:
            ``x`` (node features), ``edge_index``, ``edge_attr``, ``y``,
            ``material_id``.
        """
        # ── Node features: element one-hot (N_atoms, 92) ────────────────
        x = torch.zeros(len(structure), NODE_DIM, dtype=torch.float32)
        for i, site in enumerate(structure):
            sym = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
            idx = _ELEMENT_INDEX.get(sym, 0)
            x[i, idx] = 1.0

        # ── Edges: all pairs within cutoff ──────────────────────────────
        all_distances = structure.get_all_neighbors(self.cutoff_radius, include_index=True)
        src_list, dst_list, dist_list = [], [], []
        for src_idx, neighbors in enumerate(all_distances):
            for neighbor in neighbors:
                dst_idx = neighbor[2]
                dist = neighbor[1]
                # Bidirectional: add both directions
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                dist_list.append(dist)

        if len(src_list) == 0:
            # Isolated atom fallback: self-loop with zero distance
            src_list = list(range(len(structure)))
            dst_list = list(range(len(structure)))
            dist_list = [0.0] * len(structure)

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )  # (2, E)

        # ── Edge attributes: Gaussian expansion of distances ─────────────
        dists_tensor = torch.tensor(dist_list, dtype=torch.float32)  # (E,)
        edge_attr = self._gaussian_smear(dists_tensor)  # (E, n_gaussian_basis)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([[y]], dtype=torch.float32),
            material_id=material_id,
            num_nodes=len(structure),
        )

    def build_dataset(
        self,
        structures: list[Structure],
        targets: list[float],
        ids: list[str],
    ) -> list[Data]:
        """Convert a list of structures to a list of PyG Data objects.

        Args:
            structures: List of pymatgen Structures.
            targets: Corresponding target property values.
            ids: Corresponding material IDs.

        Returns:
            List of ``Data`` objects (same order as inputs).
        """
        dataset = []
        n_failed = 0
        for structure, y, mid in zip(structures, targets, ids):
            try:
                data = self.structure_to_graph(structure, y, mid)
                dataset.append(data)
            except Exception as exc:
                logger.warning("Graph build failed for %s: %s", mid, exc)
                n_failed += 1

        logger.info(
            "Built %d graphs (%d failed) from %d structures",
            len(dataset),
            n_failed,
            len(structures),
        )
        return dataset

    # ── Private helpers ──────────────────────────────────────────────────────

    def _gaussian_smear(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smearing to a vector of distances.

        Args:
            distances: Shape ``(E,)`` — interatomic distances.

        Returns:
            Shape ``(E, n_gaussian_basis)`` — Gaussian-expanded features.
        """
        # distances: (E,) → (E, 1); centers: (G,) → (1, G)
        diff = distances.unsqueeze(1) - self._centers.unsqueeze(0)
        return torch.exp(-(diff**2) / (self._width**2))
