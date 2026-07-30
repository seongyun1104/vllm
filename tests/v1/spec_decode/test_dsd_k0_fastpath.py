# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the K=0 fast-path predicate."""

import pytest

from vllm.v1.spec_decode.utils import should_skip_drafter_forward


class _SharingDrafter:
    HAS_CROSS_MODEL_KV_SHARING = True


class _NonSharingDrafter:
    HAS_CROSS_MODEL_KV_SHARING = False


class _LegacyDrafter:
    """A drafter that predates the capability flag (e.g. ngram)."""


@pytest.mark.parametrize(
    "drafter, num_spec_tokens_to_schedule, data_parallel_size, expected",
    [
        (_SharingDrafter(), 0, 1, True),
        (_SharingDrafter(), 1, 1, False),
        (_SharingDrafter(), 3, 1, False),
        (_SharingDrafter(), 0, 2, False),
        (_SharingDrafter(), 0, 4, False),
        (_NonSharingDrafter(), 0, 1, False),
        (_NonSharingDrafter(), 3, 1, False),
        (_LegacyDrafter(), 0, 1, False),
        (_LegacyDrafter(), 3, 1, False),
    ],
)
def test_predicate_matrix(
    drafter: object,
    num_spec_tokens_to_schedule: int,
    data_parallel_size: int,
    expected: bool,
) -> None:
    """Parametrized coverage of should_skip_drafter_forward across drafter
    class (shared-KV, non-shared, legacy without flag), K, and DP size.
    """
    result = should_skip_drafter_forward(
        num_spec_tokens_to_schedule,
        data_parallel_size,
        drafter,
    )
    assert result is expected
