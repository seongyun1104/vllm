# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-model producer for Dynamic SD schedules.

Emits the same rectangular ``(bs_lo, bs_hi, ctx_lo, ctx_hi, K)`` schedule a user
would hand-author, but derives every K and every context boundary from a stated
cost model instead of typed constants. The output feeds
``validate_and_normalize_dynamic_sd_schedule`` unchanged.

The model is intentionally minimal:

    E_accept(K) = sum_{i=0}^{K} acceptance**i          # expected emitted tokens
    T_step(K)   = F(ctx) + K * M(bs)
    K*(bs, ctx) = argmax_{0 <= K <= max_k} E_accept(K) / T_step(K)

``F(ctx)`` (fixed per-step verify cost, dominated by attention over the KV cache)
grows with context, so the per-K marginal cost shrinks relative to the step and
``K*`` is non-decreasing in ctx. ``M(bs)`` (per-speculative-token cost) grows with
batch, so ``K*`` is non-increasing in batch -- the usual DSD behavior. The
coefficients are design-time references; the magnitude of any resulting speedup
is workload-gated and must be calibrated on hardware before it is claimed.
"""

from dataclasses import dataclass

from .utils import _CTX_INFINITY, DynamicSDSchedule


@dataclass
class CostModelParams:
    """Coefficients for the Dynamic SD cost model.

    Args:
        acceptance: Draft-token acceptance prior in ``(0, 1)``. Use a
            conservative floor: overestimating raises K and pays the ctx tax the
            schedule exists to avoid, while underestimating only forgoes some
            uplift.
        verify_fixed_base: Fixed per-step verify cost at ``ctx -> 0``.
        verify_fixed_per_ctx: Growth of the fixed verify cost per context token.
        draft_marginal_base: Per-speculative-token cost at ``batch -> 0``.
        draft_marginal_per_bs: Growth of the per-token cost per unit batch size.
    """

    acceptance: float
    verify_fixed_base: float
    verify_fixed_per_ctx: float
    draft_marginal_base: float
    draft_marginal_per_bs: float

    def __post_init__(self) -> None:
        if not 0.0 < self.acceptance < 1.0:
            raise ValueError(
                f"acceptance must be in (0, 1), got {self.acceptance}."
            )
        if self.verify_fixed_base <= 0.0 or self.draft_marginal_base <= 0.0:
            raise ValueError(
                "verify_fixed_base and draft_marginal_base must be > 0."
            )
        if self.verify_fixed_per_ctx < 0.0 or self.draft_marginal_per_bs < 0.0:
            raise ValueError(
                "verify_fixed_per_ctx and draft_marginal_per_bs must be >= 0."
            )


def _optimal_k(params: CostModelParams, bs: int, ctx: int, max_k: int) -> int:
    """argmax over ``K in [0, max_k]`` of ``E_accept(K) / T_step(K)``."""
    fixed = params.verify_fixed_base + params.verify_fixed_per_ctx * ctx
    marginal = params.draft_marginal_base + params.draft_marginal_per_bs * bs
    best_k = 0
    best_score = -1.0
    emitted = 0.0
    weight = 1.0  # acceptance**i
    for k in range(max_k + 1):
        emitted += weight
        score = emitted / (fixed + k * marginal)
        if score > best_score:
            best_score = score
            best_k = k
        weight *= params.acceptance
    return best_k


def _coarsen_to_budget(
    dense: list[list[int]], capture_budget: int
) -> list[list[int]]:
    """Reduce the number of distinct K values to ``capture_budget`` by merging
    the closest adjacent pair (keeping the lower, conservative value) until the
    distinct-K set -- one CUDA graph per value -- fits the budget."""
    while True:
        distinct = sorted({k for row in dense for k in row})
        if len(distinct) <= capture_budget:
            return dense
        gaps = [
            (distinct[i + 1] - distinct[i], i) for i in range(len(distinct) - 1)
        ]
        _, i = min(gaps)
        lo, hi = distinct[i], distinct[i + 1]
        dense = [[lo if k == hi else k for k in row] for row in dense]


def derive_dynamic_sd_schedule(
    cost_params: CostModelParams,
    bs_tiers: list[tuple[int, int]],
    ctx_grid: list[int],
    max_k: int,
    capture_budget: int | None = None,
) -> DynamicSDSchedule:
    """Derive a rectangular Dynamic SD schedule from a cost model.

    Args:
        cost_params: Cost-model coefficients (see :class:`CostModelParams`).
        bs_tiers: Non-overlapping ``(bs_lo, bs_hi)`` batch-size ranges; the first
            must start at 1.
        ctx_grid: Ascending candidate context-boundary points (each > 0). K* is
            evaluated per tier at every grid point; boundaries are placed only
            where some tier's K* changes.
        max_k: Argmax ceiling; equals ``num_speculative_tokens``.
        capture_budget: Optional ceiling on the number of distinct K values
            (CUDA graphs). Adjacent K values are merged until it is met.

    Returns:
        A rectangular list of ``(bs_lo, bs_hi, ctx_lo, ctx_hi, K)`` entries that
        ``validate_and_normalize_dynamic_sd_schedule`` accepts.
    """
    if not bs_tiers:
        raise ValueError("bs_tiers must be non-empty.")
    if max_k <= 0:
        raise ValueError("max_k must be > 0.")
    if capture_budget is not None and capture_budget < 1:
        raise ValueError("capture_budget must be >= 1 when set.")

    tiers = sorted(bs_tiers)
    if tiers[0][0] != 1:
        raise ValueError("The first batch-size tier must start at 1.")
    prev_end = 0
    for bs_lo, bs_hi in tiers:
        if bs_lo <= prev_end or bs_lo > bs_hi:
            raise ValueError("bs_tiers must be sorted and non-overlapping.")
        prev_end = bs_hi

    grid = sorted(set(ctx_grid))
    if not grid or grid[0] <= 0:
        raise ValueError("ctx_grid must be non-empty with positive points.")

    # K*[tier][j] at each grid point; batch enters via the tier's high end
    # (highest batch -> lowest K -> conservative for the whole tier).
    kstar = [
        [_optimal_k(cost_params, bs_hi, ctx, max_k) for ctx in grid]
        for _, bs_hi in tiers
    ]

    # A context boundary is needed wherever any tier's K* changes.
    boundary_points = [
        grid[j]
        for j in range(1, len(grid))
        if any(kstar[t][j] != kstar[t][j - 1] for t in range(len(tiers)))
    ]

    ctx_lo_list = [1] + boundary_points
    ctx_hi_list = [p - 1 for p in boundary_points] + [_CTX_INFINITY]
    # Grid index whose K* represents each bucket (its lower edge).
    bucket_grid_idx = [0] + [grid.index(p) for p in boundary_points]

    dense = [
        [kstar[t][j] for j in bucket_grid_idx] for t in range(len(tiers))
    ]
    if capture_budget is not None:
        dense = _coarsen_to_budget(dense, capture_budget)

    schedule: DynamicSDSchedule = []
    for t, (bs_lo, bs_hi) in enumerate(tiers):
        for c, (ctx_lo, ctx_hi) in enumerate(zip(ctx_lo_list, ctx_hi_list)):
            schedule.append((bs_lo, bs_hi, ctx_lo, ctx_hi, dense[t][c]))
    return schedule
