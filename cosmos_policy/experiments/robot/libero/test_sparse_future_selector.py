from cosmos_policy.experiments.robot.libero.sparse_future_selector import (
    heuristic_spatial_indices,
    random_spatial_indices,
    set_model_sparse_future_tokens,
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


def test_temperature_top_p_sampler_beats_uniform_near_gripper_over_seeds():
    box = BOXES["wrist"]
    roi = {row * 14 + col for row in range(5, 9) for col in range(1, 12)}
    heuristic_hits = 0
    random_hits = 0
    for seed in range(100):
        heuristic_hits += len(
            set(heuristic_spatial_indices(box, grid_size=14, budget=8, seed=seed)) & roi
        )
        random_hits += len(set(random_spatial_indices(grid_size=14, budget=8, seed=seed)) & roi)

    assert heuristic_hits > random_hits * 2


def test_full_restores_dense_path():
    model = _Model()
    set_model_sparse_future_tokens(model, selector="full", budget_per_view=8, seed=0)

    assert model.net.indices is None
