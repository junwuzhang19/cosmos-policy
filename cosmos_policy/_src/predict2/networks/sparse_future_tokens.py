# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical token pruning helpers for sparse future-image inference.

The DiT normally represents video tokens as ``[B, T, H, W, D]``.  During
inference we keep every token in non-selected latent frames and retain only a
caller-provided set of spatial tokens in selected future-image frames.  The
helpers preserve the original flattened token order so positional embeddings
can be gathered with exactly the same indices.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseTokenLayout:
    """Index mapping between a dense video grid and its physically kept tokens."""

    keep_indices: torch.Tensor
    frame_indices: torch.Tensor
    dense_shape: tuple[int, int, int]
    sparse_frames: tuple[int, ...]

    @property
    def full_token_count(self) -> int:
        t, h, w = self.dense_shape
        return t * h * w

    @property
    def kept_token_count(self) -> int:
        return int(self.keep_indices.numel())

    @property
    def dropped_token_count(self) -> int:
        return self.full_token_count - self.kept_token_count


def canonicalize_sparse_frame_indices(
    frame_to_spatial_indices: Mapping[int, Sequence[int]] | None,
) -> dict[int, tuple[int, ...]] | None:
    """Copy and normalize a user-facing sparse-frame mapping."""

    if frame_to_spatial_indices is None:
        return None
    result: dict[int, tuple[int, ...]] = {}
    for frame, indices in frame_to_spatial_indices.items():
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise TypeError(f"Frame indices must be integers, got {frame!r}")
        normalized = tuple(int(index) for index in indices)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Frame {frame} contains duplicate spatial token indices")
        result[frame] = tuple(sorted(normalized))
    return result


def build_sparse_token_layout(
    *,
    num_frames: int,
    height: int,
    width: int,
    frame_to_spatial_indices: Mapping[int, Sequence[int]],
    device: torch.device | str | None = None,
) -> SparseTokenLayout:
    """Build sorted dense-token indices for independent per-frame selections."""

    if min(num_frames, height, width) <= 0:
        raise ValueError("num_frames, height, and width must all be positive")

    canonical = canonicalize_sparse_frame_indices(frame_to_spatial_indices)
    assert canonical is not None
    spatial_size = height * width
    invalid_frames = [frame for frame in canonical if frame < 0 or frame >= num_frames]
    if invalid_frames:
        raise ValueError(f"Sparse frame indices out of range: {invalid_frames}")

    keep: list[int] = []
    for frame in range(num_frames):
        spatial_indices = canonical.get(frame)
        if spatial_indices is None:
            spatial_indices = tuple(range(spatial_size))
        invalid_spatial = [index for index in spatial_indices if index < 0 or index >= spatial_size]
        if invalid_spatial:
            raise ValueError(
                f"Frame {frame} spatial indices out of range for {height}x{width}: {invalid_spatial}"
            )
        frame_offset = frame * spatial_size
        keep.extend(frame_offset + index for index in spatial_indices)

    if not keep:
        raise ValueError("Sparse selection cannot remove every token")

    keep_indices = torch.tensor(keep, dtype=torch.long, device=device)
    frame_indices = torch.div(keep_indices, spatial_size, rounding_mode="floor")
    return SparseTokenLayout(
        keep_indices=keep_indices,
        frame_indices=frame_indices,
        dense_shape=(num_frames, height, width),
        sparse_frames=tuple(sorted(canonical)),
    )


def gather_video_tokens(x_b_t_h_w_d: torch.Tensor, layout: SparseTokenLayout) -> torch.Tensor:
    """Gather dense video tokens into ``[B, kept_tokens, 1, 1, D]``."""

    batch, num_frames, height, width, channels = x_b_t_h_w_d.shape
    if (num_frames, height, width) != layout.dense_shape:
        raise ValueError(
            f"Tensor grid {(num_frames, height, width)} does not match layout {layout.dense_shape}"
        )
    flat = x_b_t_h_w_d.reshape(batch, layout.full_token_count, channels)
    return flat.index_select(1, layout.keep_indices).reshape(batch, layout.kept_token_count, 1, 1, channels)


def gather_frame_conditioning(x_b_t_d: torch.Tensor | None, layout: SparseTokenLayout) -> torch.Tensor | None:
    """Repeat per-frame conditioning for each retained spatial token."""

    if x_b_t_d is None:
        return None
    if x_b_t_d.shape[1] != layout.dense_shape[0]:
        raise ValueError(
            f"Conditioning has {x_b_t_d.shape[1]} frames but layout has {layout.dense_shape[0]}"
        )
    return x_b_t_d.index_select(1, layout.frame_indices)


def gather_rope_tokens(rope_l_1_1_d: torch.Tensor | None, layout: SparseTokenLayout) -> torch.Tensor | None:
    """Keep the exact original 3-D RoPE coordinates for retained tokens."""

    if rope_l_1_1_d is None:
        return None
    if rope_l_1_1_d.shape[0] != layout.full_token_count:
        raise ValueError(
            f"RoPE has {rope_l_1_1_d.shape[0]} tokens but layout has {layout.full_token_count}"
        )
    return rope_l_1_1_d.index_select(0, layout.keep_indices)


def scatter_video_tokens(x_b_l_1_1_d: torch.Tensor, layout: SparseTokenLayout) -> torch.Tensor:
    """Scatter retained outputs back to the original grid, filling dropped tokens with zero."""

    batch, kept_tokens, one_h, one_w, channels = x_b_l_1_1_d.shape
    if (kept_tokens, one_h, one_w) != (layout.kept_token_count, 1, 1):
        raise ValueError(
            "Sparse tensor must have shape "
            f"[B, {layout.kept_token_count}, 1, 1, D], got {tuple(x_b_l_1_1_d.shape)}"
        )
    dense = x_b_l_1_1_d.new_zeros((batch, layout.full_token_count, channels))
    dense.index_copy_(1, layout.keep_indices, x_b_l_1_1_d.reshape(batch, kept_tokens, channels))
    num_frames, height, width = layout.dense_shape
    return dense.reshape(batch, num_frames, height, width, channels)
