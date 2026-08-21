import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from datasets import Dataset

from e2e_common.data import DatasetMixSourcePreset
from e2e_common import lazy_datasets as lazy_module
from e2e_common.lazy_datasets import (
    _IndexedMixedRawStream,
    _LazyPresetIterableDataset,
    _PermutedRawDatasetView,
    _build_permutation_indices,
    _iter_random_source_indices,
    _iter_raw_records_for_worker,
    _load_indexed_raw_mix,
    build_mixed_lazy_dataset,
    dataset_length_or_none,
)


def _message_preset(alias: str, path: str) -> DatasetMixSourcePreset:
    return DatasetMixSourcePreset(
        alias=alias,
        path=path,
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="edgerazor_messages",
    )


def _raw_dataset(source: str, count: int) -> Dataset:
    return Dataset.from_dict(
        {
            "source": [source for _ in range(count)],
            "row_id": list(range(count)),
            "messages": [
                [
                    {"role": "user", "content": f"question_{source}_{idx}"},
                    {"role": "assistant", "content": f"answer_{source}_{idx}"},
                ]
                for idx in range(count)
            ],
        }
    )


def _raw_signature(raw_dataset, limit: int = 32):
    result = []
    for idx in range(min(int(limit), int(len(raw_dataset)))):
        row = raw_dataset[idx]
        result.append((str(row["source"]), int(row["row_id"])))
    return result


def _record_signature(records):
    return [(str(row["source"]), int(row["row_id"])) for row in records]


class _IdentityLazyDataset(_LazyPresetIterableDataset):
    def _encode_record(self, record):
        return record


class _RawStream:
    def __init__(self, count: int):
        self.count = int(count)

    def __iter__(self):
        for idx in range(self.count):
            yield {"id": idx}


class _NoTransformRawDataset:
    def __init__(self, raw):
        self.raw = raw

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        return self.raw[int(index)]

    def shuffle(self, *args, **kwargs):
        raise AssertionError("shuffle must not be called")

    def map(self, *args, **kwargs):
        raise AssertionError("map must not be called")

    def add_column(self, *args, **kwargs):
        raise AssertionError("add_column must not be called")

    def select(self, *args, **kwargs):
        raise AssertionError("select must not be called")

    def flatten_indices(self, *args, **kwargs):
        raise AssertionError("flatten_indices must not be called")


class RemainingLoraDataRegressionCleanupTests(unittest.TestCase):
    def test_permutation_indices_are_deterministic_and_int32_for_normal_dataset(self):
        first = _build_permutation_indices(128, seed=31)
        second = _build_permutation_indices(128, seed=31)
        other = _build_permutation_indices(128, seed=32)

        self.assertEqual(first.dtype, np.int32)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertNotEqual(first.tolist(), other.tolist())
        self.assertEqual(sorted(first.tolist()), list(range(128)))
        self.assertFalse(first.flags.writeable)

    def test_indexable_worker_shard_has_no_duplicates_and_covers_all_records(self):
        raw = [{"id": idx} for idx in range(32)]
        preset = _message_preset("tiny", "tiny/path")
        dataset = _IdentityLazyDataset(
            raw,
            tokenizer=None,
            max_seq_len=8,
            task="messages",
            preset=preset,
            seed=0,
        )

        with mock.patch.object(
            lazy_module,
            "get_worker_info",
            return_value=SimpleNamespace(id=0, num_workers=2),
        ):
            worker0 = [int(row["id"]) for row in dataset]
        with mock.patch.object(
            lazy_module,
            "get_worker_info",
            return_value=SimpleNamespace(id=1, num_workers=2),
        ):
            worker1 = [int(row["id"]) for row in dataset]

        self.assertEqual(set(worker0).intersection(worker1), set())
        self.assertEqual(set(worker0).union(worker1), set(range(32)))
        self.assertEqual(len(worker0) + len(worker1), 32)

    def test_non_indexable_stream_worker_shard_has_no_duplicates_and_covers_all_records(self):
        raw = _RawStream(32)
        with mock.patch.object(
            lazy_module,
            "get_worker_info",
            return_value=SimpleNamespace(id=0, num_workers=2),
        ):
            worker0 = [int(row["id"]) for row in _iter_raw_records_for_worker(raw)]
        with mock.patch.object(
            lazy_module,
            "get_worker_info",
            return_value=SimpleNamespace(id=1, num_workers=2),
        ):
            worker1 = [int(row["id"]) for row in _iter_raw_records_for_worker(raw)]

        self.assertEqual(set(worker0).intersection(worker1), set())
        self.assertEqual(set(worker0).union(worker1), set(range(32)))
        self.assertEqual(len(worker0) + len(worker1), 32)

    def test_mixed_lazy_dataset_length_remains_unknown(self):
        preset = _message_preset("tiny", "tiny/path")
        dataset = _LazyPresetIterableDataset(
            [{"id": idx} for idx in range(8)],
            tokenizer=None,
            max_seq_len=8,
            task="messages",
            preset=preset,
            seed=0,
        )

        self.assertIsNone(dataset_length_or_none(dataset))

    def test_single_source_indexed_loader_does_not_call_hf_shuffle(self):
        raw = _raw_dataset("A", 32)
        presets = {"tiny_a": _message_preset("tiny_a", "tiny/a")}

        def load_raw(preset):
            self.assertEqual(str(preset.alias), "tiny_a")
            return raw, None

        with mock.patch.dict(lazy_module.DATASET_MIX_SOURCE_PRESETS, presets, clear=False):
            with mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw):
                with mock.patch.object(
                    raw,
                    "shuffle",
                    side_effect=AssertionError("shuffle must not be called"),
                ):
                    _spec, _stats, view, _presets, kind = _load_indexed_raw_mix(
                        "tiny_a=1.0",
                        seed=31,
                    )

        self.assertEqual(kind, "tiny_a")
        self.assertIsInstance(view, _PermutedRawDatasetView)
        self.assertEqual(len(view), 32)
        expected = [int(idx) for idx in _build_permutation_indices(32, seed=31)[:16]]
        actual = [int(view[idx]["row_id"]) for idx in range(16)]
        self.assertEqual(actual, expected)

    def test_multi_source_indexed_loader_does_not_call_hf_transforms(self):
        raw_a = _NoTransformRawDataset(_raw_dataset("A", 48))
        raw_b = _NoTransformRawDataset(_raw_dataset("B", 24))
        presets = {
            "tiny_a": _message_preset("tiny_a", "tiny/a"),
            "tiny_b": _message_preset("tiny_b", "tiny/b"),
        }

        def load_raw(preset):
            if str(preset.alias) == "tiny_a":
                return raw_a, None
            if str(preset.alias) == "tiny_b":
                return raw_b, None
            raise AssertionError(f"unexpected preset {preset.alias}")

        with mock.patch.dict(lazy_module.DATASET_MIX_SOURCE_PRESETS, presets, clear=False):
            with mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw):
                cache = {}
                _spec, _stats, stream, _presets, kind = _load_indexed_raw_mix(
                    "tiny_a=0.7,tiny_b=0.3",
                    seed=31,
                    raw_dataset_cache=cache,
                )

        self.assertEqual(kind, "mix")
        self.assertIsInstance(stream, _IndexedMixedRawStream)
        self.assertEqual(
            set(cache.keys()),
            {("tiny_a", "tiny/a", None, "train"), ("tiny_b", "tiny/b", None, "train")},
        )
        self.assertIs(cache[("tiny_a", "tiny/a", None, "train")], raw_a)
        self.assertIs(cache[("tiny_b", "tiny/b", None, "train")], raw_b)
        sample = next(iter(stream))
        self.assertNotIn("__vaellm_source_idx", sample)

    def test_mixed_stream_is_deterministic_and_resets_per_iterator(self):
        raw_a = _raw_dataset("A", 48)
        raw_b = _raw_dataset("B", 24)
        stream31 = _IndexedMixedRawStream(
            (raw_a, raw_b),
            (
                _build_permutation_indices(len(raw_a), seed=31),
                _build_permutation_indices(len(raw_b), seed=31),
            ),
            (0.7, 0.3),
            seed=31,
        )
        stream32 = _IndexedMixedRawStream(
            (raw_a, raw_b),
            (
                _build_permutation_indices(len(raw_a), seed=32),
                _build_permutation_indices(len(raw_b), seed=32),
            ),
            (0.7, 0.3),
            seed=32,
        )

        first = list(itertools.islice(iter(stream31), 64))
        second = list(itertools.islice(iter(stream31), 64))
        seed32 = list(itertools.islice(iter(stream32), 64))
        self.assertEqual(_record_signature(first), _record_signature(second))
        self.assertNotEqual(_record_signature(first), _record_signature(seed32))

    def test_source_chooser_ratio_and_seed_are_repeatable(self):
        first = list(
            itertools.islice(
                _iter_random_source_indices(
                    num_sources=2,
                    probabilities=(0.7, 0.3),
                    seed=31,
                ),
                20000,
            )
        )
        second = list(
            itertools.islice(
                _iter_random_source_indices(
                    num_sources=2,
                    probabilities=(0.7, 0.3),
                    seed=31,
                ),
                20000,
            )
        )
        other = list(
            itertools.islice(
                _iter_random_source_indices(
                    num_sources=2,
                    probabilities=(0.7, 0.3),
                    seed=32,
                ),
                20000,
            )
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        ratio_a = first.count(0) / len(first)
        self.assertGreater(ratio_a, 0.68)
        self.assertLess(ratio_a, 0.72)

    def test_indexed_stream_all_exhausted_wraps_only_after_source_permutation_done(self):
        raw_a = _raw_dataset("A", 5)
        raw_b = _raw_dataset("B", 3)
        stream = _IndexedMixedRawStream(
            (raw_a, raw_b),
            (
                _build_permutation_indices(len(raw_a), seed=31),
                _build_permutation_indices(len(raw_b), seed=31),
            ),
            (0.8, 0.2),
            seed=31,
        )

        rows = list(iter(stream))
        signature = _record_signature(rows)
        self.assertGreaterEqual(len(signature), 8)
        self.assertEqual(
            {row_id for source, row_id in signature if source == "A"},
            set(range(5)),
        )
        self.assertEqual(
            {row_id for source, row_id in signature if source == "B"},
            set(range(3)),
        )

        expected_sizes = {"A": 5, "B": 3}
        seen = {"A": set(), "B": set()}
        for source, row_id in signature:
            if row_id in seen[source]:
                self.assertEqual(seen[source], set(range(expected_sizes[source])))
            seen[source].add(row_id)

    def test_indexed_stream_worker_shard_matches_baseline_stride(self):
        raw_a = _raw_dataset("A", 12)
        raw_b = _raw_dataset("B", 8)
        stream = _IndexedMixedRawStream(
            (raw_a, raw_b),
            (
                _build_permutation_indices(len(raw_a), seed=31),
                _build_permutation_indices(len(raw_b), seed=31),
            ),
            (0.6, 0.4),
            seed=31,
        )

        baseline = _record_signature(stream.iter_worker(worker_id=0, num_workers=1))
        worker0 = _record_signature(stream.iter_worker(worker_id=0, num_workers=2))
        worker1 = _record_signature(stream.iter_worker(worker_id=1, num_workers=2))
        self.assertEqual(worker0, baseline[0::2])
        self.assertEqual(worker1, baseline[1::2])

    def test_mixed_lazy_dataset_rejects_multiple_text_formats(self):
        raw_a = _raw_dataset("A", 8)
        raw_b = _raw_dataset("B", 8)
        preset_a = _message_preset("tiny_a", "tiny/a")
        preset_b = DatasetMixSourcePreset(
            alias="tiny_other",
            path="tiny/other",
            config=None,
            train_split="train",
            eval_split=None,
            text_field="messages",
            text_format="plain",
        )
        presets = {"tiny_a": preset_a, "tiny_other": preset_b}

        def load_raw(preset):
            if str(preset.alias) == "tiny_a":
                return raw_a, None
            if str(preset.alias) == "tiny_other":
                return raw_b, None
            raise AssertionError(f"unexpected preset {preset.alias}")

        with mock.patch.dict(lazy_module.DATASET_MIX_SOURCE_PRESETS, presets, clear=False):
            with mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw):
                with self.assertRaisesRegex(ValueError, "multiple text_format"):
                    build_mixed_lazy_dataset(
                        "tiny_a=0.5,tiny_other=0.5",
                        task="messages",
                        tokenizer=object(),
                        max_seq_len=8,
                        seed=0,
                    )

    def test_lazy_stats_do_not_fake_packed_rows(self):
        raw_a = _raw_dataset("A", 8)
        raw_b = _raw_dataset("B", 8)
        presets = {
            "tiny_a": _message_preset("tiny_a", "tiny/a"),
            "tiny_b": _message_preset("tiny_b", "tiny/b"),
        }

        def load_raw(preset):
            if str(preset.alias) == "tiny_a":
                return raw_a, None
            if str(preset.alias) == "tiny_b":
                return raw_b, None
            raise AssertionError(f"unexpected preset {preset.alias}")

        with mock.patch.dict(lazy_module.DATASET_MIX_SOURCE_PRESETS, presets, clear=False):
            with mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw):
                _spec, source_stats, _raw, _presets, _kind = _load_indexed_raw_mix(
                    "tiny_a=0.5,tiny_b=0.5",
                    seed=0,
                )
                for source_info in source_stats:
                    self.assertNotIn("packed_rows", source_info)

                _spec, _source_stats, dataset, _is_iterable = build_mixed_lazy_dataset(
                    "tiny_a=0.5,tiny_b=0.5",
                    task="messages",
                    tokenizer=object(),
                    max_seq_len=8,
                    seed=0,
                )

        self.assertIsNone(dataset_length_or_none(dataset))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
