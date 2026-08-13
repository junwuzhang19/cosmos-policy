from types import SimpleNamespace

import numpy as np

from cosmos_policy.experiments.robot.libero.sparse_future_selector import (
    _bbox,
    _render_segmentation,
    heuristic_spatial_indices,
    random_spatial_indices,
    set_model_sparse_future_tokens,
    wrist_aperture_prior_mask,
    wrist_gripper_exclusion_mask,
)

BOXES = {
    "wrist": [{"role": "gripper", "x0": 0.1, "y0": 0.4, "x1": 0.8, "y1": 0.6}],
    "agent": [
        {"role": "gripper", "x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2},
        {"role": "target", "x0": 0.2, "y0": 0.2, "x1": 0.3, "y1": 0.3},
    ],
}


class _Net:
    def set_sparse_future_token_indices(self, indices):
        self.indices = indices


class _Model:
    net = _Net()


def test_random_views_use_independent_draws():
    model = _Model()
    metadata = set_model_sparse_future_tokens(
        model,
        selector="random",
        budget_per_view=8,
        seed=3,
    )

    assert len(metadata["indices"][6]) == 8
    assert len(metadata["indices"][7]) == 8
    assert metadata["indices"][6] != metadata["indices"][7]


def test_views_accept_asymmetric_budgets():
    model = _Model()
    metadata = set_model_sparse_future_tokens(
        model,
        selector="random",
        budget_per_view=8,
        budgets_by_view={"agent": 8, "wrist": 32},
        seed=3,
    )

    assert len(metadata["indices"][6]) == 32
    assert len(metadata["indices"][7]) == 8
    assert metadata["budgets_by_view"] == {"agent": 8, "wrist": 32}


def test_asymmetric_budgets_require_both_views():
    model = _Model()

    try:
        set_model_sparse_future_tokens(
            model,
            selector="random",
            budget_per_view=8,
            budgets_by_view={"agent": 8},
            seed=3,
        )
    except ValueError as error:
        assert "exactly wrist and agent" in str(error)
    else:
        raise AssertionError("missing view budget must be rejected")


def test_heuristic_views_are_sampled_separately_and_reproducibly():
    model = _Model()
    first = set_model_sparse_future_tokens(
        model,
        selector="heuristic",
        budget_per_view=8,
        seed=9,
        boxes_by_view=BOXES,
    )
    second = set_model_sparse_future_tokens(
        model,
        selector="heuristic",
        budget_per_view=8,
        seed=9,
        boxes_by_view=BOXES,
    )

    assert first == second
    assert first["indices"][6] != first["indices"][7]
    assert set(first["indices"][6]).isdisjoint(
        first["wrist_excluded_spatial_indices"]
    )


def test_temperature_top_p_sampler_beats_uniform_near_gripper_over_seeds():
    box = BOXES["wrist"]
    roi = set(
        np.flatnonzero(wrist_aperture_prior_mask(14).numpy().reshape(-1)).tolist()
    )
    heuristic_hits = 0
    random_hits = 0
    for seed in range(100):
        heuristic_hits += len(
            set(
                heuristic_spatial_indices(
                    box, grid_size=14, budget=8, seed=seed, view="wrist"
                )
            )
            & roi
        )
        random_hits += len(
            set(
                random_spatial_indices(
                    grid_size=14,
                    budget=8,
                    seed=seed,
                    excluded=wrist_gripper_exclusion_mask(14),
                )
            )
            & roi
        )

    assert heuristic_hits > random_hits * 2


def test_full_restores_dense_path():
    model = _Model()
    set_model_sparse_future_tokens(model, selector="full", budget_per_view=8, seed=0)

    assert model.net.indices is None


def test_bbox_flip_matches_policy_image_vertical_flip():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 2:5] = True

    assert _bbox(mask, flip_y=True) == {"x0": 0.2, "y0": 0.7, "x1": 0.5, "y1": 0.9}


def test_segmentation_decode_casts_uint8_before_bit_packing():
    context = SimpleNamespace(
        scn=SimpleNamespace(ngeom=1, geoms=[SimpleNamespace(segid=0, objtype=5, objid=7)]),
        render=lambda **_: None,
        read_pixels=lambda *_args, **_kwargs: np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.uint8),
    )
    model = SimpleNamespace(camera_name2id=lambda _name: 3)
    env = SimpleNamespace(sim=SimpleNamespace(_render_context_offscreen=context, model=model))

    result = _render_segmentation(env, "agentview", 2)

    assert result.tolist() == [[[5, 7], [-1, -1]]]


def test_wrist_template_scales_user_7x7_layout_to_cosmos_14x14_grid():
    exclusion = wrist_gripper_exclusion_mask(14)
    aperture = wrist_aperture_prior_mask(14)

    assert int(exclusion.sum()) == 44
    assert int(aperture.sum()) == 32
    assert exclusion[12:14].all()
    assert aperture[8:10, 2:12].all()
    assert aperture[10:12, 4:10].all()
