"""CPU regression tests for the DFlash compact-draft host seq-lens bound.

The compact (``--speculative-draft-window-size``) draft path fills the planning
``seq_lens_cpu`` from ``_fill_compact_seq_lens_cpu_bound`` ->
``_compute_compact_draft_seq_lens_host``, and attention backends consume it as a
host mirror of the DEVICE kv length (the same contract the non-compact path
upholds with ``seq_lens_cpu.copy_(...); seq_lens_cpu.add_(block_size)``).

The draft's device kv length is the windowed prefix PLUS the verify block:
``_rebuild_compact_draft_cache`` writes the suffix window at ``[0,
draft_prefix_len)`` and the ``block_size`` verify slots immediately after it. A
bound that stops at the prefix plans a page table one block short of the rows
the draft attends. These tests pin the bound against the real device arithmetic
(``_compute_compact_draft_seq_lens`` plus the block) rather than against a
hand-rolled formula, at the shipped profile's page/block geometry (page 64,
T8, window 2048).

The host mirror must also stay sync-free: the sync-free sources are the
overlap-scheduled host lengths, not a D2H of the device tensor, so the tests
pass a sentinel for the device lengths and fail loudly if a host source is
ever ignored.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The profile this path ships on: window 2048, page 64, 8 draft tokens (T8).
WINDOW = 2048
PAGE = 64
BLOCK = 8


class _DeviceOnly:
    """Stand-in for the device compact prefix lengths: any read is a sync."""

    def __getattr__(self, name):
        raise AssertionError(
            f"seq_lens_cpu planning fell back to a device read ({name})"
        )


def _worker(window=WINDOW, page_size=PAGE, block_size=BLOCK):
    # __init__ needs a real model; the two bound helpers only touch these three.
    w = object.__new__(DFlashWorkerV2)
    w.draft_window_size = window
    w.page_size = page_size
    w.block_size = block_size
    return w


def _device_kv_lens(w, lens):
    """What the draft actually holds on device: exact compact prefix + block."""
    seq = torch.tensor(lens, dtype=torch.int64)
    fake_self = SimpleNamespace(
        device=seq.device, draft_window_size=w.draft_window_size, page_size=w.page_size
    )
    prefix = DFlashWorkerV2._compute_compact_draft_seq_lens(fake_self, seq)
    return (prefix.to(torch.int64) + w.block_size).tolist()


def _host_bound(w, lens):
    out = torch.zeros(len(lens), dtype=torch.int32)
    w._compute_compact_draft_seq_lens_host(
        torch.tensor(lens, dtype=torch.int64), out=out
    )
    return out.tolist()


class TestCompactDraftSeqLensHostBound(CustomTestCase):
    def test_bound_covers_device_kv_length_page64_t8(self):
        w = _worker()
        lens = [
            1,
            7,
            8,
            PAGE - 1,
            PAGE,
            PAGE + 1,
            WINDOW - 1,
            WINDOW,
            WINDOW + 1,
            WINDOW + PAGE - 1,
            WINDOW + PAGE,
            WINDOW + PAGE + 1,
            3 * WINDOW,
            28110,
        ]
        # Every phase of the page-alignment sawtooth at the window edge.
        lens += [WINDOW + PAGE + phase for phase in range(0, PAGE, 7)]
        lens += [WINDOW + phase for phase in range(0, PAGE, 7)]
        for got, L in zip(_host_bound(w, lens), lens):
            device = _device_kv_lens(w, [L])[0]
            self.assertGreaterEqual(
                got,
                device,
                f"host bound {got} < device kv length {device} at len={L} "
                f"(window={WINDOW}, page={PAGE}, block={BLOCK})",
            )

    def test_page_phase_edges_use_the_monotonic_envelope(self):
        """An over-estimated host length must not under-shoot the shorter true
        device length: the exact page-align arithmetic is a sawtooth, so the
        envelope (not the formula) has to be what is mirrored."""
        w = _worker(page_size=PAGE)
        # The exact compact length rises one per token until the visible start
        # reaches the next page boundary, then drops back to the window.
        true_len = WINDOW + 2 * PAGE - 1
        reserved = true_len + 1  # overlap bound the host actually carries
        self.assertLess(
            _device_kv_lens(w, [reserved])[0],
            _device_kv_lens(w, [true_len])[0],
            "expected the sawtooth counter-example to hold at this geometry",
        )
        self.assertGreaterEqual(
            _host_bound(w, [reserved])[0], _device_kv_lens(w, [true_len])[0]
        )

    def test_short_prefixes_pass_through_plus_the_block(self):
        w = _worker(page_size=1)
        self.assertEqual(
            _host_bound(w, [1, 1024, WINDOW - 1]),
            [x + BLOCK for x in (1, 1024, WINDOW - 1)],
        )

    def test_saturated_prefixes_clamp_then_add_the_block(self):
        """Clamp first: adding the block to the clamp ceiling and clamping to it
        would come out block short for any prefix cropped below the ceiling."""
        self.assertEqual(
            _host_bound(_worker(page_size=1), [3 * WINDOW]), [WINDOW + BLOCK]
        )
        self.assertEqual(
            _host_bound(_worker(page_size=PAGE), [3 * WINDOW]), [WINDOW + PAGE + BLOCK]
        )

    def test_paged_backend_still_gets_page_headroom(self):
        w = _worker(page_size=PAGE, block_size=1)
        self.assertEqual(_host_bound(w, [100_000]), [WINDOW + PAGE + 1])


class TestFillCompactSeqLensCpuBound(CustomTestCase):
    def _fill(
        self, w, *, batch_seq_lens_cpu=None, nxt_kv_lens_cpu=None, device_lens, size
    ):
        out = torch.zeros(size, dtype=torch.int32)
        w._fill_compact_seq_lens_cpu_bound(
            batch_seq_lens_cpu=batch_seq_lens_cpu,
            nxt_kv_lens_cpu=nxt_kv_lens_cpu,
            draft_prefix_lens=device_lens,
            out=out,
        )
        return out.tolist()

    def _check_host_source(self, *, kwname):
        # Unpaged: the envelope is the exact device value plus the block.
        exact_self = _worker(page_size=1)
        host = torch.tensor(
            [1, PAGE + 1, WINDOW - 1, WINDOW, 3 * WINDOW], dtype=torch.int64
        )
        got = self._fill(
            exact_self, device_lens=_DeviceOnly(), size=host.numel(), **{kwname: host}
        )
        self.assertEqual(got, _device_kv_lens(exact_self, host.tolist()))
        # Paged: the host mirror may exceed the device value (that is the point
        # of the envelope), but never falls under it.
        paged_self = _worker()
        got = self._fill(
            paged_self, device_lens=_DeviceOnly(), size=host.numel(), **{kwname: host}
        )
        for bound, L in zip(got, host.tolist()):
            self.assertGreaterEqual(bound, _device_kv_lens(paged_self, [L])[0])

    def test_batch_host_lens_cover_the_block_without_a_device_read(self):
        self._check_host_source(kwname="batch_seq_lens_cpu")

    def test_next_kv_host_lens_cover_the_block_without_a_device_read(self):
        self._check_host_source(kwname="nxt_kv_lens_cpu")

    def test_blocking_fallback_also_covers_the_block(self):
        """With no host source the legacy D2H of the compact prefix is the only
        value available, and it is prefix-only, so the block is added there too."""
        w = _worker()
        prefix = torch.tensor([1, WINDOW, 3 * WINDOW], dtype=torch.int32)
        got = self._fill(w, device_lens=prefix, size=3)
        self.assertEqual(got, [p + BLOCK for p in prefix.tolist()])


if __name__ == "__main__":
    unittest.main()
