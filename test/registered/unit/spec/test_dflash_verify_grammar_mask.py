"""CPU regression tests for the stale request-level grammar mask in verify.

``apply_dflash_verify_logits_adjustments`` runs on the ``[bs * draft_token_num,
vocab]`` verify logits. ``sampling_info.grammar_mask`` at that point belongs to
the previous extend/decode step: it has one row per REQUEST and is applied to a
zero-valued buffer that is then broadcast over the whole verify block. Two
consequences, both reproduced here:

* a request that finished its grammar, or a batch that was filtered down,
  leaves an all-masked (or differently-sized) row behind -- every verify logit
  becomes -inf, or the mask's row count no longer matches the batch;
* even a same-sized, still-live mask is the wrong state: each tree position
  has its own grammar state, and the DFlash/DSpark verify callers already build
  and apply that per-position mask themselves (build_grammar_vocab_mask), so
  broadcasting the per-request mask masks a position from its predecessor.

The per-position mask is applied by the caller AFTER these adjustments, so the
helper clearing the request-level row cannot drop it.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 4
BS = 2
DRAFT_TOKEN_NUM = 2


class _MaskingBackend:
    """A grammar backend that only turns masked positions into -inf."""

    @staticmethod
    def apply_vocab_mask(*, logits, vocab_mask):
        logits.masked_fill_(vocab_mask.to(dtype=torch.bool), float("-inf"))


def _sampling_info(**overrides):
    info = SamplingBatchInfo(
        temperatures=torch.ones(BS, 1),
        top_ps=torch.ones(BS),
        top_ks=torch.ones(BS, dtype=torch.int32),
        min_ps=torch.zeros(BS),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        device="cpu",
    )
    for name, value in overrides.items():
        setattr(info, name, value)
    return info


def _stale_mask(rows=BS, *, mask_all=True):
    """The leftover request-level mask: one row per (previous) request."""
    vocab_mask = torch.ones(rows, VOCAB) if mask_all else torch.zeros(rows, VOCAB)
    return GrammarMask(_MaskingBackend(), vocab_mask.to(dtype=torch.bool))


def _adjust(logits, sampling_info):
    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=DRAFT_TOKEN_NUM,
    )


class TestDFlashVerifyStaleGrammarMask(CustomTestCase):
    def test_unconstrained_verify_is_not_poisoned_by_a_stale_mask(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.tensor(
                [[0.0, 1.0, 4.0, 2.0]] * (BS * DRAFT_TOKEN_NUM), dtype=dtype
            )
            expected = logits.clone()
            sampling_info = _sampling_info(grammar_mask=_stale_mask())

            _adjust(logits, sampling_info)

            torch.testing.assert_close(logits, expected)
            self.assertEqual(
                logits.argmax(dim=-1).tolist(), [2] * (BS * DRAFT_TOKEN_NUM)
            )
            self.assertIsNone(sampling_info.grammar_mask)

    def test_no_new_mask_after_grammar_transition(self):
        """All grammars finished -> the caller builds no new mask, so the only
        thing that could be applied here is the previous step's."""
        logits = torch.arange(BS * DRAFT_TOKEN_NUM * VOCAB, dtype=torch.float32).view(
            BS * DRAFT_TOKEN_NUM, VOCAB
        )
        expected = logits.clone()
        sampling_info = _sampling_info(
            grammars=[None, None], grammar_mask=_stale_mask()
        )

        _adjust(logits, sampling_info)

        torch.testing.assert_close(logits, expected)
        self.assertIsNone(sampling_info.grammar_mask)

    def test_filtered_batch_row_count_mismatch_survives(self):
        """The batch was filtered down after the mask was built, so the mask has
        more rows than the verify batch: applying it here would not even fit."""
        logits = torch.zeros((BS * DRAFT_TOKEN_NUM, VOCAB), dtype=torch.float32)
        sampling_info = _sampling_info(grammar_mask=_stale_mask(rows=BS + 1))

        _adjust(logits, sampling_info)  # must not raise

        torch.testing.assert_close(logits, torch.zeros_like(logits))
        self.assertIsNone(sampling_info.grammar_mask)

    def test_per_position_verify_mask_still_wins(self):
        """The caller's fresh per-position mask, applied after the adjustments,
        is the only grammar state that reaches the verify logits."""
        logits = torch.tensor([[0.0, 1.0, 4.0, 2.0]] * (BS * DRAFT_TOKEN_NUM))
        sampling_info = _sampling_info(grammar_mask=_stale_mask())
        # One row per verify position: position i may only emit token i.
        per_position = GrammarMask(
            _MaskingBackend(),
            torch.tensor(
                [
                    [False, True, True, True],
                    [True, False, True, True],
                    [True, True, False, True],
                    [True, True, True, False],
                ]
            ),
        )

        _adjust(logits, sampling_info)
        per_position.apply(logits)

        self.assertEqual(logits.argmax(dim=-1).tolist(), [0, 1, 2, 3])
        self.assertEqual(torch.isneginf(logits).sum().item(), 3 * VOCAB)

    def test_penalties_and_logit_bias_survive_a_stale_mask(self):
        """The additive-penalty and logit-bias paths must not be diverted by the
        stale mask -- and logit_bias still lands after the penalizer, which is
        what the float32 accumulation buffer rounds differently."""
        additive = torch.tensor([[-1.0, 0.0, -2.0, 0.0], [0.0, -1.0, 0.0, -2.0]])
        logit_bias = torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
        expected = torch.tensor(
            [
                [0.0, 4.0, -1.0, 1.0],
                [0.0, 4.0, -1.0, 1.0],
                [1.0, 0.0, 4.0, -1.0],
                [1.0, 0.0, 4.0, -1.0],
            ]
        )
        for has_stale_mask in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(stale=has_stale_mask, dtype=dtype):
                    logits = torch.ones((BS * DRAFT_TOKEN_NUM, VOCAB), dtype=dtype)
                    sampling_info = _sampling_info(
                        acc_additive_penalties=additive.to(dtype),
                        logit_bias=logit_bias.to(dtype),
                        grammar_mask=_stale_mask() if has_stale_mask else None,
                    )

                    _adjust(logits, sampling_info)

                    torch.testing.assert_close(logits, expected.to(dtype))
                    self.assertIsNone(sampling_info.grammar_mask)

    def test_live_penalizer_fallback_keeps_bias_ordering(self):
        """A live (non-overlap) penalizer still takes the dense path, and the
        ordering there is unchanged: penalizer, then (cleared) grammar, then
        logit_bias -- and the stale mask contributes nothing."""
        penalty = torch.tensor([[-1.0, 0.0, -2.0, 0.0], [0.0, -1.0, 0.0, -2.0]])
        sampling_info = _sampling_info(
            penalizer_orchestrator=SimpleNamespace(
                is_required=True, apply=lambda x: x.add_(penalty)
            ),
            logit_bias=torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]),
            grammar_mask=_stale_mask(),
        )
        logits = torch.ones((BS * DRAFT_TOKEN_NUM, VOCAB))

        _adjust(logits, sampling_info)

        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [0.0, 4.0, -1.0, 1.0],
                    [0.0, 4.0, -1.0, 1.0],
                    [1.0, 0.0, 4.0, -1.0],
                    [1.0, 0.0, 4.0, -1.0],
                ]
            ),
        )
        self.assertIsNone(sampling_info.grammar_mask)

    def test_custom_logit_processor_still_applies_before_the_rest(self):
        """A custom processor runs first, on the unadjusted logits, whether or
        not a stale mask is present; penalties and bias are added on top."""

        class _Shift(torch.nn.Module):
            def __call__(self, logits, params):
                return logits - 100.0

        additive = torch.tensor([[-1.0, 0.0, -2.0, 0.0], [0.0, -1.0, 0.0, -2.0]])
        for has_stale_mask in (False, True):
            with self.subTest(stale=has_stale_mask):
                logits = torch.zeros((BS * DRAFT_TOKEN_NUM, VOCAB))
                sampling_info = _sampling_info(
                    has_custom_logit_processor=True,
                    custom_params=[None] * BS,
                    custom_logit_processor={
                        1: (
                            _Shift(),
                            torch.ones(BS, dtype=torch.bool),
                        )
                    },
                    acc_additive_penalties=additive,
                    grammar_mask=_stale_mask() if has_stale_mask else None,
                )

                _adjust(logits, sampling_info)

                torch.testing.assert_close(
                    logits,
                    torch.tensor(
                        [
                            [-101.0, -100.0, -102.0, -100.0],
                            [-101.0, -100.0, -102.0, -100.0],
                            [-100.0, -101.0, -100.0, -102.0],
                            [-100.0, -101.0, -100.0, -102.0],
                        ]
                    ),
                )
                self.assertIsNone(sampling_info.grammar_mask)


if __name__ == "__main__":
    unittest.main()
