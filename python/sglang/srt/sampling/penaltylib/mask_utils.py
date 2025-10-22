from __future__ import annotations

from typing import List, Optional

import torch


def _flatten_row_indices(blocked_ids: List[Optional[torch.Tensor]], device) -> Optional[torch.Tensor]:
    """Return a 1-D tensor of row indices aligned with concatenated ids.

    For each row i that has a non-empty ids tensor, repeat i for len(ids) times.
    Returns None if there is no valid id across the batch.
    """
    rows: List[torch.Tensor] = []
    for i, ids in enumerate(blocked_ids):
        if ids is None:
            continue
        # Ensure tensor and on correct device
        if not torch.is_tensor(ids):
            continue
        if ids.numel() == 0:
            continue
        rows.append(torch.full((ids.numel(),), i, device=device, dtype=torch.long))
    if not rows:
        return None
    return torch.cat(rows)


def apply_blocked_ids_mask_inplace(
    tensor: torch.Tensor,
    blocked_ids: Optional[List[Optional[torch.Tensor]]],
    *,
    fill_value: float,
    dense_threshold: int = 512,
) -> None:
    """Apply a per-row block mask to `tensor` in-place efficiently.

    Uses a per-row heuristic:
    - rows with nnz <= dense_threshold use one batched sparse scatter via index_put_
    - rows with nnz > dense_threshold build a per-row boolean mask and masked_fill_

    Args:
        tensor: [B, V] tensor (logits or probs)
        blocked_ids: list of length B where each entry is 1-D LongTensor of ids
            to be filled with `fill_value` for that row, or None/empty for no-op.
        fill_value: value to write at blocked positions (e.g., -inf or 0.0)
        dense_threshold: threshold for switching a row to dense mask path.
    """
    if not blocked_ids:
        return

    B, V = tensor.shape
    device = tensor.device

    # Separate rows into sparse and dense by their nnz
    sparse_pairs: List[torch.Tensor] = []  # stacked (row_idx, col_ids)
    sparse_ids: List[torch.Tensor] = []
    dense_rows: List[int] = []
    dense_cols: List[torch.Tensor] = []

    for i, ids in enumerate(blocked_ids):
        if ids is None or (torch.is_tensor(ids) and ids.numel() == 0):
            continue
        if not torch.is_tensor(ids):
            continue
        t = ids.to(device=device, dtype=torch.long)
        nnz = int(t.numel())
        if nnz <= dense_threshold:
            sparse_pairs.append(torch.full((nnz,), i, device=device, dtype=torch.long))
            sparse_ids.append(t)
        else:
            dense_rows.append(i)
            dense_cols.append(t)

    # Apply sparse rows in one batched op
    if sparse_pairs:
        row_idx = torch.cat(sparse_pairs)
        col_idx = torch.cat(sparse_ids)
        tensor.index_put_((row_idx, col_idx), torch.as_tensor(fill_value, device=device))

    # Apply dense rows with per-row boolean masks (few rows expected)
    for i, cols in zip(dense_rows, dense_cols):
        mask_row = torch.zeros((V,), device=device, dtype=torch.bool)
        mask_row.index_put_(
            (cols,), torch.ones_like(cols, dtype=torch.bool, device=device)
        )
        tensor[i].masked_fill_(mask_row, fill_value)
