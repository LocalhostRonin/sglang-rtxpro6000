from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.pool_host.mamba import (
    _SLOT_ALIGN_BYTES,
    MambaPoolHost,
    _align_slot_sibling_strides,
    _SlotSiblingHostSpec,
)
from sglang.srt.mem_cache.storage.mmap import alloc_mmap

# Real PLE geometry of Qwen3.8-Flash-Next: one short-conv layer of
# (hidden_size * hc_count, (ple_conv_kernel_size - 1) * ngram_size) and an int64
# ngram context of (ngram_size - 1) tokens. The ngram sibling is the one that
# used to be 16 bytes/slot.
SHORT_CONV_SHAPE = (10240, 9)
NGRAM_CONTEXT_LEN = 2
NUM_MAMBA_LAYERS = 36
TEMPORAL_ELEM, CONV_ELEM = 786432, 30720


def _specs(short_conv_shape=SHORT_CONV_SHAPE, ngram_len=NGRAM_CONTEXT_LEN):
    """The two PLE slot siblings, built the way MambaPool.cache_pool allocates them."""
    slots = 4
    conv_state = torch.zeros((1, slots + 1) + short_conv_shape, dtype=torch.bfloat16)
    ngram_context = torch.zeros((slots + 1, ngram_len), dtype=torch.int64)
    return [
        _SlotSiblingHostSpec(
            name="ple_short_conv",
            # slot_axis == 1: the layer dim already leads the conv state.
            device_layers=conv_state,
            slot_axis=1,
        ),
        _SlotSiblingHostSpec(
            name="ple_ngram",
            # slot_axis == 0: a bare (slots, context_len) tensor, unsqueezed.
            device_layers=ngram_context.unsqueeze(0),
            slot_axis=0,
        ),
    ]


def _host_allocate(spec, size):
    """init_kv_buffer's page_first sibling branch."""
    assert spec.bytes_per_slot % spec.dtype.itemsize == 0
    return alloc_mmap((size, spec.bytes_per_slot // spec.dtype.itemsize), spec.dtype)


def _pool_stub(layout="page_first", num_slots=4):
    """A MambaPoolHost carrying only what is_stride_page_aligned / host sizing read."""
    pool = object.__new__(MambaPoolHost)
    pool.layout = layout
    pool.num_mamba_layers = NUM_MAMBA_LAYERS
    pool.temporal_state_elem_size = TEMPORAL_ELEM
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.bfloat16
    pool.conv_state_elem_sizes = [CONV_ELEM]
    pool.temporal_buffer = alloc_mmap(
        (num_slots, NUM_MAMBA_LAYERS, 1, 48, 128, 128), torch.bfloat16
    )
    pool.conv_buffer = [
        alloc_mmap((num_slots, NUM_MAMBA_LAYERS, 1) + SHORT_CONV_SHAPE, torch.bfloat16)
    ]
    pool.slot_sibling_specs = _specs()
    for spec in pool.slot_sibling_specs:
        spec.host_buffer = _host_allocate(spec, num_slots)
    return pool


class TestMambaHostSlotStride(unittest.TestCase):
    def test_sub_page_sibling_is_padded_to_the_page_size(self):
        """The ngram context is 16 bytes/slot: one sub-page stride is enough to
        push the entire hybrid pool onto bounce buffers."""
        short_conv, ngram = _specs()
        self.assertEqual(short_conv.item_bytes, 184320)  # already page aligned
        self.assertEqual(ngram.item_bytes, 16)

        _align_slot_sibling_strides([short_conv, ngram])

        # Nothing to pad stays tight, so the common case costs nothing.
        self.assertEqual(short_conv.bytes_per_slot, 184320)
        self.assertEqual(ngram.bytes_per_slot, _SLOT_ALIGN_BYTES)
        for spec in (short_conv, ngram):
            self.assertEqual(spec.bytes_per_slot % _SLOT_ALIGN_BYTES, 0)
            self.assertGreaterEqual(spec.bytes_per_slot, spec.item_bytes)

    def test_padding_is_idempotent_and_never_shrinks_a_slot(self):
        specs = _specs()
        _align_slot_sibling_strides(specs)
        before = [spec.bytes_per_slot for spec in specs]
        _align_slot_sibling_strides(specs)
        self.assertEqual([spec.bytes_per_slot for spec in specs], before)

    def test_zero_copy_predicate_rejects_then_accepts_the_sibling(self):
        """This is the predicate HiCacheNixl._hybrid_pool_supports_zero_copy
        calls before it decides to bounce."""
        unpadded = _pool_stub()
        self.assertFalse(unpadded.is_stride_page_aligned(_SLOT_ALIGN_BYTES))

        padded = _pool_stub()
        _align_slot_sibling_strides(padded.slot_sibling_specs)
        for spec in padded.slot_sibling_specs:
            spec.host_buffer = _host_allocate(spec, 4)
        self.assertTrue(padded.is_stride_page_aligned(_SLOT_ALIGN_BYTES))

    def test_layer_first_layout_still_reports_unaligned(self):
        """Layer-first strides are driven by `size`, so the predicate must keep
        declining; only page_first offers page-strided slots."""
        pool = _pool_stub(layout="layer_first")
        _align_slot_sibling_strides(pool.slot_sibling_specs)
        self.assertFalse(pool.is_stride_page_aligned(_SLOT_ALIGN_BYTES))

    def test_padded_slots_round_trip_without_reading_the_pad(self):
        specs = _specs()
        _align_slot_sibling_strides(specs)
        for spec in specs:
            spec.host_buffer = _host_allocate(spec, 4)
            host_slot = spec.host_slot(2)
            self.assertEqual(host_slot.shape, spec.device_layers[:, 2].shape)

            device_slot = spec.device_layers[:, 2]
            device_slot.copy_(
                torch.arange(device_slot.numel(), dtype=torch.float32)
                .to(spec.dtype)
                .view_as(device_slot)
            )
            want = device_slot.clone()

            host_slot.copy_(device_slot)  # backup_from_device_all_layer
            self.assertTrue(torch.equal(host_slot, want))

            device_slot.zero_()
            device_slot.copy_(host_slot)  # load_slot_siblings_to_device
            self.assertTrue(torch.equal(device_slot, want))

            # The flat page carries the pad, but the item bytes stay in front.
            flat = spec.host_buffer[2].view(torch.uint8).reshape(-1)
            self.assertEqual(flat.numel(), spec.bytes_per_slot)
            self.assertTrue(
                torch.equal(
                    flat[: spec.item_bytes],
                    want.contiguous().view(torch.uint8).reshape(-1),
                )
            )
            if spec.bytes_per_slot > spec.item_bytes:
                self.assertFalse(flat[spec.item_bytes :].any())

    def test_size_per_token_stays_a_whole_number_of_pages(self):
        """size_per_token is the flat page stride the storage layer copies, so a
        fractional page would desynchronise page pointers from slot strides."""
        pool = _pool_stub()
        unpadded = pool.get_size_per_token()
        _align_slot_sibling_strides(pool.slot_sibling_specs)
        padded = pool.get_size_per_token()

        self.assertEqual(
            unpadded,
            (CONV_ELEM * 2 + TEMPORAL_ELEM * 2) * NUM_MAMBA_LAYERS + 184320 + 16,
        )
        # 184320 is already aligned, so only the 16-byte sibling moves.
        self.assertEqual(padded - unpadded, _SLOT_ALIGN_BYTES - 16)
        self.assertEqual(padded % _SLOT_ALIGN_BYTES, 0)
        # The slot strides have to add up to exactly what the pool advertises.
        components = (CONV_ELEM * 2 + TEMPORAL_ELEM * 2) * NUM_MAMBA_LAYERS
        components += sum(spec.bytes_per_slot for spec in pool.slot_sibling_specs)
        self.assertEqual(padded, components)

    def test_sibling_page_pointers_stay_page_aligned(self):
        """get_page_buffer_meta emits base + index * bytes_per_slot, so the
        padded stride has to keep every page pointer on a page boundary."""
        pool = _pool_stub(num_slots=8)
        _align_slot_sibling_strides(pool.slot_sibling_specs)
        for spec in pool.slot_sibling_specs:
            spec.host_buffer = _host_allocate(spec, 8)
            base = spec.host_buffer.data_ptr()
            self.assertEqual(base % _SLOT_ALIGN_BYTES, 0)
            for index in range(8):
                ptr = base + index * spec.bytes_per_slot
                self.assertEqual(ptr % _SLOT_ALIGN_BYTES, 0)

    def test_hybrid_key_components_are_not_dropped(self):
        """_get_hybrid_key_multiplier keys off the component inventory, so the
        fix must align the sibling rather than remove it."""
        pool = _pool_stub()
        names = pool.get_storage_component_names()
        self.assertIn("ple_ngram", names)
        self.assertIn("ple_short_conv", names)
        self.assertEqual(len(names), len(pool.get_hybrid_pool_buffer()))


if __name__ == "__main__":
    unittest.main()
