import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from datasets import Dataset

from e2e_common.data import DatasetMixSourcePreset
from e2e_common import lazy_datasets as lazy_module
from e2e_common.lazy_datasets import (
    _LazyPresetIterableDataset,
    _iter_raw_records_for_worker,
    _load_interleaved_raw_mix,
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


class _IdentityLazyDataset(_LazyPresetIterableDataset):
    def _encode_record(self, record):
        return record


class _RawStream:
    def __init__(self, count: int):
        self.count = int(count)

    def __iter__(self):
        for idx in range(self.count):
            yield {"id": idx}


class RemainingLoraDataRegressionCleanupTests(unittest.TestCase):
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

    def test_two_source_raw_cache_matches_no_cache_for_each_seed(self):
        raw_a = _raw_dataset("A", 48)
        raw_b = _raw_dataset("B", 48)
        presets = {
            "tiny_a": _message_preset("tiny_a", "tiny/a"),
            "tiny_b": _message_preset("tiny_b", "tiny/b"),
        }
        load_calls = []

        def load_raw(preset):
            load_calls.append(str(preset.alias))
            if str(preset.alias) == "tiny_a":
                return raw_a, None
            if str(preset.alias) == "tiny_b":
                return raw_b, None
            raise AssertionError(f"unexpected preset {preset.alias}")

        with mock.patch.dict(lazy_module.DATASET_MIX_SOURCE_PRESETS, presets, clear=False):
            with mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw):
                cache = {}
                _spec31, _stats31, cached31, _presets31, _kind31 = _load_interleaved_raw_mix(
                    "tiny_a=0.7,tiny_b=0.3",
                    seed=31,
                    raw_dataset_cache=cache,
                )
                _spec32, _stats32, cached32, _presets32, _kind32 = _load_interleaved_raw_mix(
                    "tiny_a=0.7,tiny_b=0.3",
                    seed=32,
                    raw_dataset_cache=cache,
                )

        self.assertEqual(load_calls, ["tiny_a", "tiny_b"])
        self.assertEqual(
            set(cache.keys()),
            {("tiny_a", "tiny/a", None, "train"), ("tiny_b", "tiny/b", None, "train")},
        )
        self.assertIs(cache[("tiny_a", "tiny/a", None, "train")], raw_a)
        self.assertIs(cache[("tiny_b", "tiny/b", None, "train")], raw_b)
        self.assertNotIn("__vaellm_source_idx", list(cached31.column_names))

        with mock.patch.dict(lazy_module.DATASET_MIX_SOURCE_PRESETS, presets, clear=False):
            with mock.patch.object(lazy_module, "_load_preset_raw_datasets", side_effect=load_raw):
                _spec31, _stats31, plain31, _presets31, _kind31 = _load_interleaved_raw_mix(
                    "tiny_a=0.7,tiny_b=0.3",
                    seed=31,
                )
                _spec32, _stats32, plain32, _presets32, _kind32 = _load_interleaved_raw_mix(
                    "tiny_a=0.7,tiny_b=0.3",
                    seed=32,
                )

        cached31_signature = _raw_signature(cached31)
        cached32_signature = _raw_signature(cached32)
        plain31_signature = _raw_signature(plain31)
        plain32_signature = _raw_signature(plain32)
        self.assertEqual(cached31_signature, plain31_signature)
        self.assertEqual(cached32_signature, plain32_signature)
        self.assertNotEqual(cached31_signature, cached32_signature)

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
                _spec, source_stats, _raw, _presets, _kind = _load_interleaved_raw_mix(
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
