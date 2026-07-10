#!/usr/bin/env python3
"""Download EdgeRazor-style distill jsonl from HuggingFace (no EdgeRazor repo required)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2e_common.data import VAELLM_EDGERAZOR_DATA_DIR

GENERAL_JSONL_NAMES = (
    "ii_7M_instruct.jsonl",
    "ii_gen_1.4M_instruct.jsonl",
    "tulu_0.6M_instruct.jsonl",
    "am_1.4M_instruct.jsonl",
)

from contextlib import contextmanager

_INFINITY_REPO = "BAAI/Infinity-Instruct"
_INFINITY_ROLE_MAP = {"human": "user", "gpt": "assistant"}
_AM_REPO = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
_AM_FILES = ("am_0.5M.jsonl", "am_0.9M.jsonl")
_INFINITY_ACCESS_CHECKED = False


def _resolve_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


@contextmanager
def _official_hf_hub_endpoint():
    saved_endpoint = os.environ.pop("HF_ENDPOINT", None)
    try:
        yield
    finally:
        if saved_endpoint is not None:
            os.environ["HF_ENDPOINT"] = saved_endpoint


def _infinity_access_error(config_name: Optional[str], exc: Exception) -> RuntimeError:
    from huggingface_hub.errors import GatedRepoError

    if isinstance(exc, GatedRepoError):
        return RuntimeError(
            f"无法访问 gated 数据集 {_INFINITY_REPO}（config={config_name}）。\n"
            f"HF token 可能无效，或该账号尚未获批访问权限。\n"
            f"1) 打开 https://huggingface.co/datasets/{_INFINITY_REPO} ，"
            f"用配置 token 的同一账号登录并申请访问；\n"
            f"2) 若 token 无效，在 https://huggingface.co/settings/tokens 重新生成，"
            f"并更新 scripts/download_distill_dataset.sh 中的 HF_TOKEN。"
        )
    return RuntimeError(
        f"无法加载 gated 数据集 {_INFINITY_REPO}（config={config_name}）。\n"
        f"1) 在 https://huggingface.co/datasets/{_INFINITY_REPO} 申请访问；\n"
        f"2) 配置有效 HF token 并更新 scripts/download_distill_dataset.sh。"
    )


def _ensure_infinity_access() -> None:
    global _INFINITY_ACCESS_CHECKED
    if _INFINITY_ACCESS_CHECKED:
        return

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    token = _resolve_hf_token()
    if not token or token == "hf_xxx":
        raise RuntimeError(
            f"{_INFINITY_REPO} 为 gated 数据集，请先在 scripts/download_distill_dataset.sh "
            f"配置有效 HF_TOKEN，并在 https://huggingface.co/datasets/{_INFINITY_REPO} 申请访问。"
        )

    with _official_hf_hub_endpoint():
        api = HfApi()
        try:
            api.whoami(token=token)
        except Exception as exc:
            raise RuntimeError(
                "HF_TOKEN 无效或已过期。请在 https://huggingface.co/settings/tokens "
                "重新生成 token，并更新 scripts/download_distill_dataset.sh。"
            ) from exc
        try:
            hf_hub_download(
                _INFINITY_REPO,
                "7M/train-00000-of-00075.parquet",
                repo_type="dataset",
                token=token,
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                f"HF token 有效，但账号尚未获得 {_INFINITY_REPO} 的访问权限。\n"
                f"请用同一账号打开 https://huggingface.co/datasets/{_INFINITY_REPO} "
                f"填写申请表单，等待审批通过后再重试。"
            ) from exc

    _INFINITY_ACCESS_CHECKED = True


def _load_hf_dataset(repo_id: str, config_name: Optional[str] = None, *, split: str = "train"):
    from datasets import load_dataset

    token = _resolve_hf_token()
    kwargs = {"split": split}
    if token is not None:
        kwargs["token"] = token
    if config_name is None:
        return load_dataset(repo_id, **kwargs)
    return load_dataset(repo_id, config_name, **kwargs)


def _load_infinity_dataset(config_name: str, *, split: str = "train"):
    from datasets import load_dataset

    _ensure_infinity_access()
    token = _resolve_hf_token()
    with _official_hf_hub_endpoint():
        try:
            return load_dataset(_INFINITY_REPO, config_name, split=split, token=token)
        except Exception as exc:
            raise _infinity_access_error(config_name, exc) from exc


def _convert_infinity_conversations(conversations: object) -> Optional[List[Dict[str, str]]]:
    if not conversations or not isinstance(conversations, list):
        return None
    messages: List[Dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        role = _INFINITY_ROLE_MAP.get(str(turn.get("from", "")).strip(), str(turn.get("from", "")).strip())
        content = turn.get("value", "")
        messages.append({"role": role, "content": str(content)})
    return messages if messages else None


def _clean_am_messages(messages: object) -> Optional[List[Dict[str, str]]]:
    if not messages or not isinstance(messages, list):
        return None
    clean: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        info = msg.get("info")
        if role == "assistant":
            content = ""
            if isinstance(info, dict):
                content = str(info.get("answer_content") or "")
        else:
            content = str(msg.get("content", ""))
        clean.append({"role": role, "content": content})
    return clean if clean else None


def _write_jsonl_record(handle, record: Dict[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _report_saved(path: Path, count: int) -> None:
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"Saved {count} samples -> {path} ({size_mb:.1f} MB)")


def generate_ii_7M_instruct(*, output_dir: Path, max_samples: Optional[int]) -> int:
    from tqdm import tqdm

    output_path = output_dir / "ii_7M_instruct.jsonl"
    print(f"Loading {_INFINITY_REPO} (7M) ...")
    dataset = _load_infinity_dataset("7M")
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(dataset, desc="ii_7M_instruct"):
            messages = _convert_infinity_conversations(example.get("conversations"))
            if messages is None:
                continue
            _write_jsonl_record(handle, {"messages": messages})
            count += 1
            if max_samples is not None and count >= int(max_samples):
                break
    _report_saved(output_path, count)
    return count


def generate_ii_gen_instruct(*, output_dir: Path, max_samples: Optional[int]) -> int:
    from tqdm import tqdm

    output_path = output_dir / "ii_gen_1.4M_instruct.jsonl"
    print(f"Loading {_INFINITY_REPO} (Gen) ...")
    dataset = _load_infinity_dataset("Gen")
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(dataset, desc="ii_gen_1.4M_instruct"):
            messages = _convert_infinity_conversations(example.get("conversations"))
            if messages is None:
                continue
            _write_jsonl_record(handle, {"messages": messages})
            count += 1
            if max_samples is not None and count >= int(max_samples):
                break
    _report_saved(output_path, count)
    return count


def generate_tulu_instruct(*, output_dir: Path, max_samples: Optional[int]) -> int:
    from tqdm import tqdm

    output_path = output_dir / "tulu_0.6M_instruct.jsonl"
    print("Loading allenai/tulu-v3.1-mix-preview-4096-OLMoE ...")
    dataset = _load_hf_dataset("allenai/tulu-v3.1-mix-preview-4096-OLMoE")
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(dataset, desc="tulu_0.6M_instruct"):
            messages = example.get("messages")
            if not isinstance(messages, list):
                continue
            _write_jsonl_record(handle, {"messages": messages})
            count += 1
            if max_samples is not None and count >= int(max_samples):
                break
    _report_saved(output_path, count)
    return count


def generate_am_instruct(*, output_dir: Path, max_samples: Optional[int]) -> int:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    output_path = output_dir / "am_1.4M_instruct.jsonl"
    count = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for filename in _AM_FILES:
            print(f"Downloading {_AM_REPO}/{filename} ...")
            download_kwargs = {"repo_type": "dataset"}
            token = _resolve_hf_token()
            if token is not None:
                download_kwargs["token"] = token
            local_path = hf_hub_download(_AM_REPO, filename, **download_kwargs)
            with open(local_path, "r", encoding="utf-8") as source:
                for line in tqdm(source, desc=f"am:{filename}"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    messages = _clean_am_messages(row.get("messages"))
                    if messages is None:
                        skipped += 1
                        continue
                    _write_jsonl_record(handle, {"messages": messages})
                    count += 1
                    if max_samples is not None and count >= int(max_samples):
                        break
            if max_samples is not None and count >= int(max_samples):
                break
    _report_saved(output_path, count)
    if skipped:
        print(f"Skipped {skipped} invalid AM rows.")
    return count


_GENERATORS = {
    "ii_7M_instruct.jsonl": generate_ii_7M_instruct,
    "ii_gen_1.4M_instruct.jsonl": generate_ii_gen_instruct,
    "tulu_0.6M_instruct.jsonl": generate_tulu_instruct,
    "am_1.4M_instruct.jsonl": generate_am_instruct,
}


def _resolve_targets(selected: Sequence[str]) -> List[str]:
    if not selected or selected == ("all",) or list(selected) == ["all"]:
        return list(GENERAL_JSONL_NAMES)
    unknown = [name for name in selected if name not in _GENERATORS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. Valid: {list(GENERAL_JSONL_NAMES)} or 'all'."
        )
    return list(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EdgeRazor-style general distill jsonl from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / VAELLM_EDGERAZOR_DATA_DIR),
        help="Directory for output jsonl files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=f"One or more of: all {' '.join(GENERAL_JSONL_NAMES)}",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional per-dataset sample cap for smoke tests.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip datasets whose output jsonl already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if output jsonl already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.skip_existing) and bool(args.force):
        raise ValueError("--skip_existing and --force are mutually exclusive.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _resolve_targets(list(args.datasets))

    print("============================================")
    print(" VAELLM EdgeRazor general jsonl preparation")
    print(f" Output: {output_dir}")
    print(f" Datasets: {', '.join(targets)}")
    if args.max_samples is not None:
        print(f" max_samples: {int(args.max_samples)} (per dataset)")
    print("============================================")

    for name in targets:
        output_path = output_dir / name
        if output_path.exists() and bool(args.skip_existing) and not bool(args.force):
            print(f"Skip existing: {output_path}")
            continue
        if output_path.exists() and not bool(args.force):
            print(f"Skip existing (pass --force to regenerate): {output_path}")
            continue
        generator = _GENERATORS[name]
        generator(output_dir=output_dir, max_samples=args.max_samples)

    print("Done.")


if __name__ == "__main__":
    main()
