"""
cgts.py
-------
Baseline 1: CGTS — Graph Transformer + SVDD for CAN bus anomaly detection.
(Zhou et al., Cybersecurity 2025)

Pipeline:
  1. READ bit-flip preprocessing  → node feature vectors per CAN ID
  2. Directed attributed graph construction (window = 100 messages)
  3. ID filter module (whitelist of known IDs)
  4. Graph Transformer (graph-level, node + edge attention + Laplacian PE)
  5. SVDD dual-judgment on G_h (node) and G_e (edge) features

Requires: torch_geometric
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import get_laplacian, to_dense_adj
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("[WARN] torch_geometric not found. CGTS will be unavailable.")


# ── READ: Bit-flip preprocessing ─────────────────────────────────────────────

def compute_bit_flip_magnitude(payload_bytes: np.ndarray) -> np.ndarray:
    """
    Compute bit flip magnitude array M_i = ceil(log10(B_i + 1e-6))
    for each bit in the payload.

    Args:
        payload_bytes : (N, 8) array of byte values [0, 255]

    Returns:
        flip_magnitude : (N, 64) float32 array
    """
    N = len(payload_bytes)
    bits = np.unpackbits(payload_bytes.astype(np.uint8), axis=1)  # (N, 64)

    # Bit flip rate per bit position
    flip_matrix = np.abs(np.diff(bits.astype(np.float32), axis=0))  # (N-1, 64)
    flip_matrix = np.vstack([flip_matrix[:1], flip_matrix])         # (N, 64) — pad first row

    # Magnitude array
    B = flip_matrix.clip(min=1e-6)
    M = np.ceil(np.log10(B)).astype(np.float32)
    return M


def read_segment_features(
    arb_id_ints: np.ndarray,
    payload_bytes: np.ndarray,
    max_blocks: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Apply READ algorithm to segment each CAN ID's data field into
    signal blocks using bit flip magnitude boundaries.

    Returns a dict: {arb_id: feature_vector (max_blocks,)} for each unique ID.
    """
    unique_ids = np.unique(arb_id_ints)
    features_per_id = {}
    all_block_counts = []

    for uid in unique_ids:
        mask = arb_id_ints == uid
        pb   = payload_bytes[mask]  # rows for this ID

        if len(pb) < 2:
            features_per_id[uid] = np.zeros(1, dtype=np.float32)
            all_block_counts.append(1)
            continue

        M = compute_bit_flip_magnitude(pb)   # (N, 64)
        mean_M = M.mean(axis=0)              # (64,) — average magnitude per bit

        # Detect signal boundaries: positions where magnitude is higher than neighbors
        blocks = []
        current_block = [mean_M[0]]
        for i in range(1, 64):
            if mean_M[i] > mean_M[i - 1]:
                blocks.append(np.mean(current_block))
                current_block = [mean_M[i]]
            else:
                current_block.append(mean_M[i])
        blocks.append(np.mean(current_block))

        feat = np.array(blocks, dtype=np.float32)
        features_per_id[uid] = feat
        all_block_counts.append(len(feat))

    # Pad all feature vectors to the same length
    if max_blocks is None:
        max_blocks = max(all_block_counts) if all_block_counts else 1

    for uid in features_per_id:
        feat = features_per_id[uid]
        if len(feat) < max_blocks:
            features_per_id[uid] = np.pad(feat, (0, max_blocks - len(feat)))
        else:
            features_per_id[uid] = feat[:max_blocks]

    return features_per_id, max_blocks


# ── Graph construction ────────────────────────────────────────────────────────

def build_message_graph(
    arb_id_ints: np.ndarray,
    node_features: Dict[int, np.ndarray],
    feature_dim: int,
) -> "Data":
    """
    Build a directed attributed graph from a window of CAN messages.

    Nodes  = unique CAN IDs in the window
    Edges  = temporal ordering (message i → message i+1 if different IDs)
    Node attributes = READ feature vectors
    Edge attributes = out-degree of source node (scalar)

    Args:
        arb_id_ints  : (N,) sequence of CAN ID integers in the window
        node_features: dict {arb_id: feature_vector}
        feature_dim  : length of each feature vector

    Returns:
        torch_geometric Data object
    """
    if not HAS_PYGEOMETRIC:
        raise RuntimeError("torch_geometric required for CGTS")

    unique_ids = sorted(set(arb_id_ints.tolist()))
    id_to_idx  = {uid: i for i, uid in enumerate(unique_ids)}
    n_nodes    = len(unique_ids)

    # Node features
    x = np.zeros((n_nodes, feature_dim), dtype=np.float32)
    for uid, idx in id_to_idx.items():
        if uid in node_features:
            x[idx] = node_features[uid]

    # Edges: consecutive message pairs
    src_list, dst_list, edge_attr_list = [], [], []
    out_degree = np.zeros(n_nodes, dtype=np.float32)

    for t in range(len(arb_id_ints) - 1):
        s = id_to_idx[arb_id_ints[t]]
        d = id_to_idx[arb_id_ints[t + 1]]
        if s != d:
            src_list.append(s)
            dst_list.append(d)
            out_degree[s] += 1

    # Normalize out-degree as edge attribute
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(
            [out_degree[s] for s in src_list], dtype=torch.float32
        ).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    return Data(
        x          = torch.from_numpy(x),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        num_nodes  = n_nodes,
    )


# ── Laplacian Positional Encoding ─────────────────────────────────────────────

def laplacian_pe(data: "Data", pe_dim: int = 8) -> torch.Tensor:
    """
    Compute Laplacian positional encoding for graph nodes.
    Returns (n_nodes, pe_dim) float tensor.
    """
    n = data.num_nodes
    if n <= 1 or data.edge_index.shape[1] == 0:
        return torch.zeros(n, pe_dim)

    edge_index, edge_weight = get_laplacian(
        data.edge_index, normalization="sym", num_nodes=n
    )
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]  # (n, n)
    try:
        _, eigvecs = torch.linalg.eigh(L)
        pe = eigvecs[:, 1:pe_dim + 1]
        if pe.shape[1] < pe_dim:
            pe = F.pad(pe, (0, pe_dim - pe.shape[1]))
    except Exception:
        pe = torch.zeros(n, pe_dim)
    return pe.float()


# ── Graph Transformer Layer ───────────────────────────────────────────────────

class GTLayer(nn.Module):
    """
    Single Graph Transformer layer with node + edge attention.
    (Dwivedi & Bresson, 2021 — adapted for CGTS)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        # Node projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Edge projections
        self.W_E = nn.Linear(1, d_model, bias=False)

        # Output projections
        self.O_h = nn.Linear(d_model, d_model)
        self.O_e = nn.Linear(d_model, d_model)

        # FFN
        self.ffn_h = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.ffn_e = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )

        self.norm1_h = nn.LayerNorm(d_model)
        self.norm2_h = nn.LayerNorm(d_model)
        self.norm1_e = nn.LayerNorm(d_model)
        self.norm2_e = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,          # (n, d_model) node features
        e: torch.Tensor,          # (m, d_model) edge features
        edge_index: torch.Tensor, # (2, m)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, m = h.shape[0], edge_index.shape[1]

        if m == 0:
            h = self.norm1_h(h + self.O_h(h))
            h = self.norm2_h(h + self.ffn_h(h))
            return h, e

        src, dst = edge_index[0], edge_index[1]

        Q = self.W_Q(h).view(n, self.n_heads, self.d_k)  # (n, H, dk)
        K = self.W_K(h).view(n, self.n_heads, self.d_k)
        V = self.W_V(h).view(n, self.n_heads, self.d_k)

        # Attention scores with edge bias
        attn = (Q[src] * K[dst]).sum(-1) / math.sqrt(self.d_k)  # (m, H)
        e_h  = e.view(m, self.n_heads, self.d_k)
        attn = attn + e_h.sum(-1) / math.sqrt(self.d_k)
        # Softmax per destination node
        attn_exp = attn.exp()                                     # (m, H)
        attn_sum = torch.zeros(n, self.n_heads, device=h.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_norm = attn_exp / (attn_sum[dst] + 1e-6)            # (m, H)

        # Aggregate
        agg = attn_norm.unsqueeze(-1) * V[src]                   # (m, H, dk)
        h_new = torch.zeros(n, self.n_heads, self.d_k, device=h.device)
        h_new.index_add_(0, dst, agg)
        h_new = h_new.view(n, self.d_model)
        h_new = self.norm1_h(h + self.O_h(h_new))
        h_new = self.norm2_h(h_new + self.ffn_h(h_new))

        # Update edge features
        e_new = self.norm1_e(e + self.O_e(e))
        e_new = self.norm2_e(e_new + self.ffn_e(e_new))

        return h_new, e_new


# ── Graph Transformer Encoder ─────────────────────────────────────────────────

class GraphTransformerEncoder(nn.Module):
    """
    Stacked GT layers → graph-level features G_h (node-pooled) + G_e (edge-pooled).
    """

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        pe_dim: int = 8,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim + pe_dim, d_model)
        self.edge_proj = nn.Linear(1, d_model)
        self.layers    = nn.ModuleList([GTLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, data: "Data") -> Tuple[torch.Tensor, torch.Tensor]:
        pe  = laplacian_pe(data, pe_dim=8)
        h   = torch.cat([data.x, pe.to(data.x.device)], dim=-1)
        h   = self.node_proj(h)                         # (n, d_model)

        edge_attr = data.edge_attr if data.edge_attr is not None else torch.zeros(data.edge_index.shape[1], 1)
        e   = self.edge_proj(edge_attr.to(h.device))    # (m, d_model)

        for layer in self.layers:
            h, e = layer(h, e, data.edge_index.to(h.device))

        G_h = h.mean(dim=0, keepdim=True)   # (1, d_model) — node-level graph repr
        G_e = e.mean(dim=0, keepdim=True) if e.shape[0] > 0 else torch.zeros(1, h.shape[-1], device=h.device)
        return G_h, G_e                     # both: (1, d_model)


# ── Full CGTS Model ───────────────────────────────────────────────────────────

class CGTS(nn.Module):
    """
    Full CGTS model.

    Training:
        Call forward(graphs) on normal-only graphs → SVDD loss.
        After init, call init_svdd_center(normal_graphs) to set c.

    Inference:
        Call anomaly_score(graphs) → dual-judgment score
        (anomaly if BOTH G_h and G_e scores exceed threshold 0).
    """

    def __init__(
        self,
        node_feat_dim: int = 16,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        nu: float = 0.01,
    ):
        super().__init__()
        self.gt   = GraphTransformerEncoder(node_feat_dim, d_model, n_layers, n_heads)
        self.nu   = nu

        self.register_buffer("c_h", torch.zeros(d_model))
        self.register_buffer("c_e", torch.zeros(d_model))
        self.R_h  = nn.Parameter(torch.tensor(0.0))
        self.R_e  = nn.Parameter(torch.tensor(0.0))

    def forward(self, graphs: List["Data"]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graphs : list of torch_geometric Data objects (one per window)

        Returns:
            loss     : SVDD loss over all graphs
            scores_h : (B,) node-level anomaly scores
            scores_e : (B,) edge-level anomaly scores
        """
        G_hs, G_es = [], []
        for g in graphs:
            G_h, G_e = self.gt(g)
            G_hs.append(G_h)
            G_es.append(G_e)

        G_hs = torch.cat(G_hs, dim=0)   # (B, d_model)
        G_es = torch.cat(G_es, dim=0)

        dist_h = ((G_hs - self.c_h) ** 2).sum(dim=1)
        dist_e = ((G_es - self.c_e) ** 2).sum(dim=1)

        loss_h = self.R_h ** 2 + (1 / self.nu) * F.relu(dist_h - self.R_h ** 2).mean()
        loss_e = self.R_e ** 2 + (1 / self.nu) * F.relu(dist_e - self.R_e ** 2).mean()
        loss   = loss_h + loss_e

        return loss, dist_h.detach(), dist_e.detach()

    @torch.no_grad()
    def init_svdd_center(self, graphs: List["Data"], device: str = "cpu"):
        self.eval()
        G_hs, G_es = [], []
        for g in graphs:
            G_h, G_e = self.gt(g.to(device))
            G_hs.append(G_h)
            G_es.append(G_e)
        G_hs = torch.cat(G_hs)
        G_es = torch.cat(G_es)
        c_h = G_hs.mean(0); c_h[(c_h.abs() < 1e-6)] = 1e-6
        c_e = G_es.mean(0); c_e[(c_e.abs() < 1e-6)] = 1e-6
        self.c_h.copy_(c_h)
        self.c_e.copy_(c_e)
        dists_h = ((G_hs - self.c_h) ** 2).sum(1)
        dists_e = ((G_es - self.c_e) ** 2).sum(1)
        self.R_h.data = torch.quantile(dists_h, 0.95).sqrt()
        self.R_e.data = torch.quantile(dists_e, 0.95).sqrt()

    @torch.no_grad()
    def anomaly_score(self, graphs: List["Data"]) -> torch.Tensor:
        """
        Dual-judgment: anomaly score = max(dist_h, dist_e) — threshold at 0
        for each graph feature independently; flag if BOTH exceed threshold.
        Returns (B,) float scores (higher = more anomalous).
        """
        self.eval()
        _, dist_h, dist_e = self.forward(graphs)
        # Dual: score is positive only if both exceed their radii
        s_h = dist_h - self.R_h ** 2
        s_e = dist_e - self.R_e ** 2
        return torch.min(s_h, s_e)   # anomaly if > 0


# ── ID Filter ─────────────────────────────────────────────────────────────────

class IDFilter:
    """
    Whitelist of normal CAN IDs seen during training.
    Unknown IDs (not in whitelist) are immediately flagged as anomalies.
    """

    def __init__(self):
        self.known_ids: Set[int] = set()

    def fit(self, arb_id_ints: np.ndarray):
        self.known_ids = set(arb_id_ints.tolist())

    def flag_unknown(self, arb_id_ints: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True = unknown (anomaly)."""
        return np.array([uid not in self.known_ids for uid in arb_id_ints])
