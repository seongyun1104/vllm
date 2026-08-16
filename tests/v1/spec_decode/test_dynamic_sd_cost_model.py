# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cost-model-derived Dynamic SD schedules."""

from types import SimpleNamespace

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.dynamic.cost_model import (
    CostModelParams,
    _optimal_k,
    derive_dynamic_sd_schedule,
)
from vllm.v1.spec_decode.dynamic.utils import (
    build_dynamic_sd_schedule_lookup,
    validate_and_normalize_dynamic_sd_schedule,
)

_PARAMS = CostModelParams(
    acceptance=0.6,
    verify_fixed_base=1.0,
    verify_fixed_per_ctx=2.5e-4,
    draft_marginal_base=0.2,
    draft_marginal_per_bs=0.02,
)
_BS_TIERS = [(1, 16), (17, 64)]
_CTX_GRID = [4000, 8000, 16000, 32000]


def test_derived_schedule_passes_validator_and_builds_lookup():
    schedule = derive_dynamic_sd_schedule(
        _PARAMS, _BS_TIERS, _CTX_GRID, max_k=5, capture_budget=4
    )
    # Every entry is a 5-item (bs_lo, bs_hi, ctx_lo, ctx_hi, K) grid cell.
    assert all(len(e) == 5 for e in schedule)
    # The producer output is consumed by the real validator unchanged.
    validate_and_normalize_dynamic_sd_schedule(schedule)
    lookup = build_dynamic_sd_schedule_lookup(
        schedule, vllm_max_batch_size=64, vllm_num_speculative_tokens=5
    )
    assert len(lookup.ctx_boundaries) >= 1


def test_kstar_non_decreasing_in_context():
    ctxs = [1, 2000, 8000, 32000, 100000]
    for bs in (16, 64):
        ks = [_optimal_k(_PARAMS, bs, c, 5) for c in ctxs]
        assert ks == sorted(ks), f"K* not non-decreasing in ctx (bs={bs}): {ks}"


def test_kstar_non_increasing_in_batch():
    for ctx in (1, 8000, 32000):
        ks = [_optimal_k(_PARAMS, bs, ctx, 5) for bs in (1, 8, 32, 128)]
        assert ks == sorted(ks, reverse=True), (
            f"K* not non-increasing in batch (ctx={ctx}): {ks}"
        )


def test_boundaries_are_derived_not_hardcoded():
    # A cost model with no ctx dependence must collapse to a single ctx bucket:
    # boundaries only appear where K* actually changes.
    flat = CostModelParams(
        acceptance=0.6,
        verify_fixed_base=1.0,
        verify_fixed_per_ctx=0.0,
        draft_marginal_base=0.2,
        draft_marginal_per_bs=0.02,
    )
    schedule = derive_dynamic_sd_schedule(
        flat, _BS_TIERS, _CTX_GRID, max_k=5
    )
    ctx_ranges = {(e[2], e[3]) for e in schedule}
    assert len(ctx_ranges) == 1, ctx_ranges

    # With a ctx-dependent model, more than one bucket emerges from the grid.
    schedule = derive_dynamic_sd_schedule(
        _PARAMS, _BS_TIERS, _CTX_GRID, max_k=5
    )
    assert len({(e[2], e[3]) for e in schedule}) > 1


def test_capture_budget_clamps_distinct_k():
    for budget in (1, 2, 3):
        schedule = derive_dynamic_sd_schedule(
            _PARAMS, _BS_TIERS, _CTX_GRID, max_k=5, capture_budget=budget
        )
        assert len({e[4] for e in schedule}) <= budget


@pytest.mark.parametrize(
    "kwargs",
    [
        {"acceptance": 0.0},
        {"acceptance": 1.0},
        {"verify_fixed_base": 0.0},
        {"draft_marginal_base": -1.0},
        {"verify_fixed_per_ctx": -1.0},
    ],
)
def test_cost_model_params_rejects_bad_values(kwargs):
    base = dict(
        acceptance=0.6,
        verify_fixed_base=1.0,
        verify_fixed_per_ctx=2.5e-4,
        draft_marginal_base=0.2,
        draft_marginal_per_bs=0.02,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        CostModelParams(**base)


def test_derive_rejects_non_rectangular_or_bad_tiers():
    with pytest.raises(ValueError):
        derive_dynamic_sd_schedule(
            _PARAMS, [(2, 16)], _CTX_GRID, max_k=5  # first tier must start at 1
        )
    with pytest.raises(ValueError):
        derive_dynamic_sd_schedule(
            _PARAMS, [(1, 16), (10, 64)], _CTX_GRID, max_k=5  # overlap
        )


def _derive_self(**overrides):
    base = dict(
        spec_schedule_mode="derive",
        speculative_token_schedule=None,
        spec_cost_model={
            "acceptance": 0.6,
            "bs_tiers": [[1, 16], [17, 64]],
            "ctx_grid": [4000, 8000, 16000, 32000],
            "capture_budget": 4,
        },
        num_speculative_tokens=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_derive_mode_populates_schedule():
    schedule = SpeculativeConfig._derive_schedule(_derive_self())
    assert all(len(e) == 5 for e in schedule)
    validate_and_normalize_dynamic_sd_schedule(schedule)


def test_derive_mode_forward_compat_acceptance_dict():
    obj = _derive_self(
        spec_cost_model={
            "acceptance": {"prior": 0.6, "self_correct_after_n_steps": 0},
            "bs_tiers": [[1, 64]],
            "ctx_grid": [4000, 32000],
        }
    )
    schedule = SpeculativeConfig._derive_schedule(obj)
    assert all(len(e) == 5 for e in schedule)


def test_derive_mode_rejects_manual_schedule():
    obj = _derive_self(speculative_token_schedule=[(1, 64, 3)])
    with pytest.raises(ValueError, match="do not also set it manually"):
        SpeculativeConfig._derive_schedule(obj)


def test_derive_mode_requires_cost_model():
    obj = _derive_self(spec_cost_model=None)
    with pytest.raises(ValueError, match="requires spec_cost_model"):
        SpeculativeConfig._derive_schedule(obj)


def test_derive_mode_rejects_unknown_key():
    obj = _derive_self(
        spec_cost_model={
            "acceptance": 0.6,
            "bs_tiers": [[1, 64]],
            "ctx_grid": [4000, 32000],
            "bogus": 1,
        }
    )
    with pytest.raises(ValueError, match="unknown keys"):
        SpeculativeConfig._derive_schedule(obj)
