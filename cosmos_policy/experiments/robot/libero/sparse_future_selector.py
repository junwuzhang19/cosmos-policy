# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent-view sparse future-token selectors for LIBERO evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def random_spatial_indices(*, grid_size: int, budget: int, seed: int) -> list[int]:
    if not 0 <= budget <= grid_size * grid_size:
        raise ValueError("budget must fit within one view's spatial token grid")
    return torch.randperm(grid_size * grid_size, generator=torch.Generator().manual_seed(seed))[:budget].tolist()


def _box_gap(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    dx = max(b["x0"] - a["x1"], a["x0"] - b["x1"], 0.0)
    dy = max(b["y0"] - a["y1"], a["y0"] - b["y1"], 0.0)
    return float(np.hypot(dx, dy))


def heuristic_spatial_indices(
    boxes: Sequence[Mapping[str, object]],
    *,
    grid_size: int,
    budget: int,
    seed: int,
    sigma_scale: float = 0.25,
    temperature: float = 0.7,
    top_p: float = 0.9,
    reachable_distance: float = 0.25,
    gripper_weight: float = 2.0,
) -> list[int]:
    """Sample one view with bbox-relative Gaussians, temperature, and top-p."""

    if not boxes:
        raise ValueError("heuristic selector requires at least one box")
    if not 0 < temperature or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0,1]")
    grippers = [box for box in boxes if box["role"] == "gripper"]
    if not grippers:
        raise ValueError("heuristic selector requires a gripper box")

    axis = (torch.arange(grid_size, dtype=torch.float64) + 0.5) / grid_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    probability = torch.zeros((grid_size, grid_size), dtype=torch.float64)
    for box in boxes:
        role = str(box["role"])
        normalized = {key: float(box[key]) for key in ("x0", "y0", "x1", "y1")}
        if role != "gripper" and min(_box_gap(gripper, normalized) for gripper in grippers) > reachable_distance:
            continue
        width = normalized["x1"] - normalized["x0"]
        height = normalized["y1"] - normalized["y0"]
        center_x = (normalized["x0"] + normalized["x1"]) / 2
        center_y = (normalized["y0"] + normalized["y1"]) / 2
        sigma_x = max(sigma_scale * width, 0.5 / grid_size)
        sigma_y = max(sigma_scale * height, 0.5 / grid_size)
        squared = ((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2
        probability += (gripper_weight if role == "gripper" else 1.0) * torch.exp(-0.5 * squared)
    probability /= probability.sum()
    probability = torch.softmax(torch.log(probability.flatten()) / temperature, dim=0).reshape_as(probability)

    values, order = torch.sort(probability.flatten(), descending=True)
    if top_p < 1:
        keep_count = (
            int(
                torch.searchsorted(
                    torch.cumsum(values, dim=0),
                    torch.tensor(top_p, dtype=values.dtype),
                    right=False,
                )
            )
            + 1
        )
    else:
        keep_count = values.numel()
    keep_count = max(keep_count, budget)
    support = order[:keep_count]
    weights = probability.flatten().index_select(0, support)
    sampled = torch.multinomial(
        weights,
        budget,
        replacement=False,
        generator=torch.Generator().manual_seed(seed),
    )
    return sorted(support.index_select(0, sampled).tolist())


def set_model_sparse_future_tokens(
    model,
    *,
    selector: str,
    budget_per_view: int,
    seed: int,
    boxes_by_view: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    wrist_frame: int = 6,
    agent_frame: int = 7,
    grid_size: int = 14,
) -> dict[str, object]:
    """Configure dense/random/heuristic future frames on a loaded policy."""

    if selector == "full":
        model.net.set_sparse_future_token_indices(None)
        return {"selector": selector, "indices": None}
    if selector == "random":
        wrist = random_spatial_indices(grid_size=grid_size, budget=budget_per_view, seed=seed)
        agent = random_spatial_indices(grid_size=grid_size, budget=budget_per_view, seed=seed + 1)
    elif selector == "heuristic":
        if boxes_by_view is None:
            raise ValueError("heuristic selector requires independent boxes_by_view")
        wrist = heuristic_spatial_indices(
            boxes_by_view["wrist"], grid_size=grid_size, budget=budget_per_view, seed=seed
        )
        agent = heuristic_spatial_indices(
            boxes_by_view["agent"], grid_size=grid_size, budget=budget_per_view, seed=seed + 1
        )
    else:
        raise ValueError(f"Unsupported sparse selector: {selector}")
    indices = {wrist_frame: wrist, agent_frame: agent}
    model.net.set_sparse_future_token_indices(indices)
    return {"selector": selector, "indices": indices}
