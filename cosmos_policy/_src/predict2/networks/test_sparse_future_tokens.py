# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cosmos_policy._src.predict2.networks.sparse_future_tokens import (
    build_sparse_token_layout,
    canonicalize_sparse_frame_indices,
    gather_frame_conditioning,
    gather_rope_tokens,
    gather_video_tokens,
    scatter_video_tokens,
)


def test_layout_keeps_non_selected_frames_dense_and_views_independent() -> None:
    layout = build_sparse_token_layout(
        num_frames=4,
        height=2,
        width=3,
        frame_to_spatial_indices={1: [0, 5], 3: [1]},
    )

    assert layout.keep_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 19]
    assert layout.frame_indices.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3]
    assert layout.full_token_count == 24
    assert layout.kept_token_count == 15
    assert layout.dropped_token_count == 9
    assert layout.sparse_frames == (1, 3)


def test_gather_and_scatter_preserve_retained_values_and_zero_dropped_values() -> None:
    dense = torch.arange(2 * 3 * 2 * 2 * 2).reshape(2, 3, 2, 2, 2)
    layout = build_sparse_token_layout(
        num_frames=3,
        height=2,
        width=2,
        frame_to_spatial_indices={1: [0, 3]},
    )

    sparse = gather_video_tokens(dense, layout)
    restored = scatter_video_tokens(sparse, layout)
    flat_dense = dense.reshape(2, -1, 2)
    flat_restored = restored.reshape(2, -1, 2)

    torch.testing.assert_close(flat_restored.index_select(1, layout.keep_indices), sparse[:, :, 0, 0])
    dropped = torch.tensor(sorted(set(range(layout.full_token_count)) - set(layout.keep_indices.tolist())))
    torch.testing.assert_close(flat_restored.index_select(1, dropped), torch.zeros((2, len(dropped), 2), dtype=dense.dtype))
    torch.testing.assert_close(flat_dense.index_select(1, layout.keep_indices), sparse[:, :, 0, 0])


def test_conditioning_and_rope_follow_original_token_coordinates() -> None:
    layout = build_sparse_token_layout(
        num_frames=3,
        height=2,
        width=2,
        frame_to_spatial_indices={1: [1], 2: [0, 3]},
    )
    conditioning = torch.tensor([[[10.0], [20.0], [30.0]]])
    rope = torch.arange(12).reshape(12, 1, 1, 1)

    assert gather_frame_conditioning(conditioning, layout).flatten().tolist() == [10, 10, 10, 10, 20, 30, 30]
    assert gather_rope_tokens(rope, layout).flatten().tolist() == layout.keep_indices.tolist()


def test_full_selection_is_dense_identity() -> None:
    dense = torch.randn(1, 2, 3, 2, 4)
    all_spatial = list(range(6))
    layout = build_sparse_token_layout(
        num_frames=2,
        height=3,
        width=2,
        frame_to_spatial_indices={0: all_spatial, 1: all_spatial},
    )

    restored = scatter_video_tokens(gather_video_tokens(dense, layout), layout)
    torch.testing.assert_close(restored, dense)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({3: [0]}, "frame indices"),
        ({1: [4]}, "spatial indices"),
        ({1: [0, 0]}, "duplicate"),
    ],
)
def test_invalid_indices_fail_loudly(mapping: dict[int, list[int]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_sparse_token_layout(
            num_frames=3,
            height=2,
            width=2,
            frame_to_spatial_indices=mapping,
        )


def test_canonicalization_copies_and_sorts_input() -> None:
    source = {7: [5, 1]}
    canonical = canonicalize_sparse_frame_indices(source)
    source[7].append(3)

    assert canonical == {7: (1, 5)}
