# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent-view sparse future-token selectors for LIBERO evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class _TaskSpec:
    family: str
    target_instance: str | None = None
    target_body_contains: str | None = None
    destination_instance: str | None = None
    destination_site: str | None = None


_GOAL_TASKS = {
    "open_the_middle_drawer_of_the_cabinet": _TaskSpec(
        "contact", target_body_contains="wooden_cabinet_1_cabinet_middle"
    ),
    "open_the_top_drawer_and_put_the_bowl_inside": _TaskSpec(
        "open_then_place",
        target_instance="akita_black_bowl_1",
        destination_site="wooden_cabinet_1_top_region",
    ),
    "push_the_plate_to_the_front_of_the_stove": _TaskSpec("push", target_instance="plate_1"),
    "put_the_bowl_on_the_plate": _TaskSpec(
        "pick_place", target_instance="akita_black_bowl_1", destination_instance="plate_1"
    ),
    "put_the_bowl_on_the_stove": _TaskSpec(
        "pick_place", target_instance="akita_black_bowl_1", destination_site="flat_stove_1_cook_region"
    ),
    "put_the_bowl_on_top_of_the_cabinet": _TaskSpec(
        "pick_place", target_instance="akita_black_bowl_1", destination_site="wooden_cabinet_1_top_side"
    ),
    "put_the_cream_cheese_in_the_bowl": _TaskSpec(
        "pick_place", target_instance="cream_cheese_1", destination_instance="akita_black_bowl_1"
    ),
    "put_the_wine_bottle_on_the_rack": _TaskSpec(
        "pick_place", target_instance="wine_bottle_1", destination_site="wine_rack_1_top_region"
    ),
    "put_the_wine_bottle_on_top_of_the_cabinet": _TaskSpec(
        "pick_place", target_instance="wine_bottle_1", destination_site="wooden_cabinet_1_top_side"
    ),
    "turn_on_the_stove": _TaskSpec("contact", target_body_contains="flat_stove_1_button"),
}


def _geom_ids_for_instance(env, instance: str) -> list[int]:
    return [int(value) for value in env.env.model.instances_to_ids[instance]["geom"]]


def _geom_ids_for_body_substring(env, substring: str) -> list[int]:
    return [
        geom_id
        for geom_id in range(env.sim.model.ngeom)
        if substring in (env.sim.model.body_id2name(env.sim.model.geom_bodyid[geom_id]) or "")
    ]


def _gripper_geom_ids(env) -> list[int]:
    mapping = env.env.model.instances_to_ids.get("PandaGripper0")
    if mapping is not None:
        return [int(value) for value in mapping["geom"]]
    return [
        geom_id for geom_id in range(env.sim.model.ngeom) if "gripper0_" in (env.sim.model.geom_id2name(geom_id) or "")
    ]


def _bbox(mask: np.ndarray, *, flip_y: bool) -> dict[str, float] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    height, width = mask.shape
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    result = {"x0": x0 / width, "y0": y0 / height, "x1": x1 / width, "y1": y1 / height}
    if flip_y:
        result["y0"], result["y1"] = 1 - result["y1"], 1 - result["y0"]
    return {key: float(value) for key, value in result.items()}


def _site_mask(env, camera: str, site: str, image_size: int) -> np.ndarray:
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

    site_id = env.sim.model.site_name2id(site)
    center = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    rotation = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    half_size = np.maximum(np.asarray(env.sim.model.site_size[site_id]), 0.015)
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = center + (signs * half_size) @ rotation.T
    transform = get_camera_transform_matrix(env.sim, camera, image_size, image_size)
    pixels = np.asarray(
        project_points_from_world_to_camera(corners, transform, image_size, image_size), dtype=np.float64
    )
    finite = np.isfinite(pixels).all(axis=1)
    mask = np.zeros((image_size, image_size), dtype=bool)
    if not finite.any():
        return mask
    y0, x0 = np.floor(pixels[finite].min(axis=0)).astype(int)
    y1, x1 = np.ceil(pixels[finite].max(axis=0)).astype(int) + 1
    mask[max(0, y0) : min(image_size, y1), max(0, x0) : min(image_size, x1)] = True
    return mask


def _target_is_touching_gripper(env, target_geom_ids: Sequence[int]) -> bool:
    target = set(target_geom_ids)
    gripper = set(_gripper_geom_ids(env))
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & target and pair & gripper:
            return True
    return False


def _render_segmentation(env, camera: str, image_size: int) -> np.ndarray:
    """Render geom IDs without robosuite's NumPy-2 uint8 overflow."""

    context = env.sim._render_context_offscreen
    camera_id = env.sim.model.camera_name2id(camera)
    context.render(width=image_size, height=image_size, camera_id=camera_id, segmentation=True)
    encoded = context.read_pixels(image_size, image_size, segmentation=False).astype(np.int32)
    packed = encoded[..., 0] + encoded[..., 1] * (2**8) + encoded[..., 2] * (2**16)
    packed[packed >= context.scn.ngeom + 1] = 0
    seg_ids = np.full((context.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
    for index in range(context.scn.ngeom):
        geom = context.scn.geoms[index]
        if geom.segid != -1:
            seg_ids[geom.segid + 1] = (geom.objtype, geom.objid)
    return seg_ids[packed]


def libero_goal_oracle_boxes(
    env,
    task_description: str,
    *,
    image_size: int = 224,
    flip_images: bool = True,
) -> dict[str, list[dict[str, object]]]:
    """Ground active atomic roles independently in both LIBERO views."""

    task = task_description.strip().lower().replace(" ", "_")
    if task not in _GOAL_TASKS:
        raise ValueError(f"No oracle sparse-grounding spec for task: {task_description}")
    spec = _GOAL_TASKS[task]
    target_ids = (
        _geom_ids_for_instance(env, spec.target_instance)
        if spec.target_instance
        else _geom_ids_for_body_substring(env, str(spec.target_body_contains))
    )
    held = bool(spec.target_instance and _target_is_touching_gripper(env, target_ids))
    active_destination = spec.family in {"pick_place", "open_then_place"} and held

    output = {}
    for camera, view in (("robot0_eye_in_hand", "wrist"), ("agentview", "agent")):
        segmentation = _render_segmentation(env, camera, image_size)[..., 1]
        role_masks: list[tuple[str, np.ndarray]] = [
            ("gripper", np.isin(segmentation, _gripper_geom_ids(env))),
            ("target", np.isin(segmentation, target_ids)),
        ]
        if active_destination and spec.destination_instance:
            role_masks.append(
                ("destination", np.isin(segmentation, _geom_ids_for_instance(env, spec.destination_instance)))
            )
        elif active_destination and spec.destination_site:
            role_masks.append(("destination", _site_mask(env, camera, spec.destination_site, image_size)))
        boxes = []
        for role, mask in role_masks:
            box = _bbox(mask, flip_y=flip_images)
            if box is not None:
                boxes.append({"role": role, **box})
        if not any(box["role"] == "gripper" for box in boxes):
            raise ValueError(f"Oracle gripper is not visible in {view} view")
        output[view] = boxes
    return output


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
    budgets_by_view: Mapping[str, int] | None = None,
    boxes_by_view: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    wrist_frame: int = 6,
    agent_frame: int = 7,
    grid_size: int = 14,
) -> dict[str, object]:
    """Configure dense/random/heuristic future frames on a loaded policy."""

    budgets = {"wrist": budget_per_view, "agent": budget_per_view}
    if budgets_by_view is not None:
        if set(budgets_by_view) != set(budgets):
            raise ValueError("budgets_by_view must contain exactly wrist and agent")
        budgets = {view: int(value) for view, value in budgets_by_view.items()}
    if any(not 0 <= value <= grid_size * grid_size for value in budgets.values()):
        raise ValueError("each sparse future budget must fit its view-local grid")
    if selector == "full":
        model.net.set_sparse_future_token_indices(None)
        return {"selector": selector, "indices": None, "budgets_by_view": budgets}
    if selector == "random":
        wrist = random_spatial_indices(grid_size=grid_size, budget=budgets["wrist"], seed=seed)
        agent = random_spatial_indices(grid_size=grid_size, budget=budgets["agent"], seed=seed + 1)
    elif selector == "heuristic":
        if boxes_by_view is None:
            raise ValueError("heuristic selector requires independent boxes_by_view")
        wrist = heuristic_spatial_indices(
            boxes_by_view["wrist"], grid_size=grid_size, budget=budgets["wrist"], seed=seed
        )
        agent = heuristic_spatial_indices(
            boxes_by_view["agent"], grid_size=grid_size, budget=budgets["agent"], seed=seed + 1
        )
    else:
        raise ValueError(f"Unsupported sparse selector: {selector}")
    indices = {wrist_frame: wrist, agent_frame: agent}
    model.net.set_sparse_future_token_indices(indices)
    return {"selector": selector, "indices": indices, "budgets_by_view": budgets}
