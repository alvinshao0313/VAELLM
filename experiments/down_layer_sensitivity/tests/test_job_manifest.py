from __future__ import annotations

import json

import pytest

from experiments.down_layer_sensitivity.worker import validate_manifest


def _job(**overrides):
    job = {
        "job_id": "restore_L17",
        "restore_layers": [17],
        "mode": "formal",
        "lm_limit": None,
    }
    job.update(overrides)
    return job


def _manifest(*, jobs=None, **overrides):
    manifest = {
        "worker_id": 0,
        "physical_gpu_id": "0",
        "mode": "formal",
        "write_weight_metrics": True,
        "jobs": list(jobs) if jobs is not None else [_job()],
    }
    manifest.update(overrides)
    return manifest


def _validate(manifest, *, worker_id=0, physical_gpu_id="0"):
    validate_manifest(manifest, worker_id=worker_id, physical_gpu_id=physical_gpu_id)


def test_valid_baseline_manifest():
    _validate(
        _manifest(
            jobs=[
                {
                    "job_id": "compressed_baseline_worker00",
                    "restore_layers": [],
                    "mode": "formal",
                    "lm_limit": None,
                }
            ]
        )
    )


def test_valid_single_restore_manifest():
    _validate(_manifest(jobs=[_job(job_id="restore_L17", restore_layers=[17])]))


def test_valid_all_down_original_manifest():
    _validate(
        _manifest(
            jobs=[
                {
                    "job_id": "all_down_original",
                    "restore_layers": list(range(36)),
                    "mode": "formal",
                    "lm_limit": None,
                }
            ]
        )
    )


def test_duplicate_job_id_raises():
    with pytest.raises(ValueError, match="duplicate job_id"):
        _validate(
            _manifest(
                jobs=[
                    _job(job_id="restore_L17", restore_layers=[17]),
                    _job(job_id="restore_L17", restore_layers=[18]),
                ]
            )
        )


@pytest.mark.parametrize("layer", [-1, 36, 100])
def test_restore_layer_outside_range_raises(layer):
    with pytest.raises(ValueError, match="outside 0..35"):
        _validate(_manifest(jobs=[_job(restore_layers=[layer])]))


def test_duplicate_layer_in_restore_list_raises():
    with pytest.raises(ValueError, match="duplicate layer"):
        _validate(_manifest(jobs=[_job(restore_layers=[17, 17])]))


@pytest.mark.parametrize("mode", ["prod", "debug", "", None, "Formal"])
def test_mode_not_smoke_or_formal_raises(mode):
    with pytest.raises(ValueError, match="mode"):
        _validate(_manifest(mode=mode, jobs=[_job(mode=mode)]))


def test_formal_job_with_lm_limit_not_none_raises():
    with pytest.raises(ValueError, match="lm_limit"):
        _validate(_manifest(jobs=[_job(mode="formal", lm_limit=2)]))


@pytest.mark.parametrize("lm_limit", [None, 1, 0, 3])
def test_smoke_job_with_lm_limit_not_2_raises(lm_limit):
    with pytest.raises(ValueError, match="lm_limit"):
        _validate(
            _manifest(
                mode="smoke",
                jobs=[_job(job_id="compressed_baseline_worker00", restore_layers=[], mode="smoke", lm_limit=lm_limit)],
            )
        )


def test_worker_id_mismatch_raises():
    with pytest.raises(ValueError, match="worker_id"):
        _validate(_manifest(), worker_id=1, physical_gpu_id="0")


def test_physical_gpu_id_mismatch_raises():
    with pytest.raises(ValueError, match="physical_gpu_id"):
        _validate(_manifest(), worker_id=0, physical_gpu_id="1")


EXPECTED_FORMAL_JOB_IDS = {
    1: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            "all_down_original",
            *[f"restore_L{i:02d}" for i in range(36)],
        ],
    },
    2: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            *[f"restore_L{i:02d}" for i in range(0, 36, 2)],
        ],
        1: [
            "compressed_baseline_worker01",
            "all_down_original",
            *[f"restore_L{i:02d}" for i in range(1, 36, 2)],
        ],
    },
    4: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            *[f"restore_L{i:02d}" for i in (2, 6, 10, 14, 18, 22, 26, 30, 34)],
        ],
        1: [
            "compressed_baseline_worker01",
            "all_down_original",
            *[f"restore_L{i:02d}" for i in (3, 7, 11, 15, 19, 23, 27, 31, 35)],
        ],
        2: [
            "compressed_baseline_worker02",
            *[f"restore_L{i:02d}" for i in (0, 4, 8, 12, 16, 20, 24, 28, 32)],
        ],
        3: [
            "compressed_baseline_worker03",
            *[f"restore_L{i:02d}" for i in (1, 5, 9, 13, 17, 21, 25, 29, 33)],
        ],
    },
    8: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            *[f"restore_L{i:02d}" for i in (6, 14, 22, 30)],
        ],
        1: [
            "compressed_baseline_worker01",
            "all_down_original",
            *[f"restore_L{i:02d}" for i in (7, 15, 23, 31)],
        ],
        2: [
            "compressed_baseline_worker02",
            *[f"restore_L{i:02d}" for i in (0, 8, 16, 24, 32)],
        ],
        3: [
            "compressed_baseline_worker03",
            *[f"restore_L{i:02d}" for i in (1, 9, 17, 25, 33)],
        ],
        4: [
            "compressed_baseline_worker04",
            *[f"restore_L{i:02d}" for i in (2, 10, 18, 26, 34)],
        ],
        5: [
            "compressed_baseline_worker05",
            *[f"restore_L{i:02d}" for i in (3, 11, 19, 27, 35)],
        ],
        6: [
            "compressed_baseline_worker06",
            *[f"restore_L{i:02d}" for i in (4, 12, 20, 28)],
        ],
        7: [
            "compressed_baseline_worker07",
            *[f"restore_L{i:02d}" for i in (5, 13, 21, 29)],
        ],
    },
}


def _job_ids(manifest: dict) -> list[str]:
    return [job["job_id"] for job in manifest["jobs"]]


def _all_job_ids(manifests: list[dict]) -> list[str]:
    ids = []
    for manifest in manifests:
        ids.extend(_job_ids(manifest))
    return ids


@pytest.mark.parametrize("num_gpus", [1, 2, 4, 8])
def test_formal_phase1_manifest_allocation_is_deterministic(num_gpus):
    from experiments.down_layer_sensitivity.run import build_phase1_manifests

    selected_gpus = [str(i) for i in range(num_gpus)]
    manifests = build_phase1_manifests(selected_gpus=selected_gpus, mode="formal")
    expected = EXPECTED_FORMAL_JOB_IDS[num_gpus]

    assert len(manifests) == num_gpus
    all_ids = _all_job_ids(manifests)
    assert len(all_ids) == 38 + num_gpus
    assert len(set(all_ids)) == 38 + num_gpus
    assert all_ids.count("all_down_original") == 1
    for layer in range(36):
        assert all_ids.count(f"restore_L{layer:02d}") == 1

    for worker_id, manifest in enumerate(manifests):
        assert manifest["worker_id"] == worker_id
        assert manifest["physical_gpu_id"] == selected_gpus[worker_id]
        assert manifest["mode"] == "formal"
        assert manifest["write_weight_metrics"] is (worker_id == 0)
        job_ids = _job_ids(manifest)
        assert job_ids == expected[worker_id]
        assert job_ids[0] == f"compressed_baseline_worker{worker_id:02d}"
        if worker_id == 0:
            assert job_ids[1] == "compressed_baseline_worker00_repeat"
        scientific_prefix = "restore_"
        for index, job_id in enumerate(job_ids):
            if job_id.startswith(scientific_prefix) or job_id == "all_down_original":
                assert index > 0
                if worker_id == 0:
                    assert index > 1
        for job in manifest["jobs"]:
            assert job["mode"] == "formal"
            assert job["lm_limit"] is None
            if job["job_id"] == "all_down_original":
                assert job["restore_layers"] == list(range(36))
            elif job["job_id"].startswith("restore_L"):
                layer = int(job["job_id"].split("L", 1)[1])
                assert job["restore_layers"] == [layer]
            else:
                assert job["restore_layers"] == []
        validate_manifest(
            manifest,
            worker_id=worker_id,
            physical_gpu_id=selected_gpus[worker_id],
        )

    again = build_phase1_manifests(selected_gpus=selected_gpus, mode="formal")
    assert json.dumps(manifests, ensure_ascii=False) == json.dumps(again, ensure_ascii=False)


def test_smoke_phase1_manifest_is_exactly_four_jobs_on_one_gpu():
    from experiments.down_layer_sensitivity.run import build_phase1_manifests

    manifests = build_phase1_manifests(selected_gpus=["0"], mode="smoke")
    assert len(manifests) == 1
    manifest = manifests[0]
    assert _job_ids(manifest) == [
        "compressed_baseline_worker00",
        "compressed_baseline_worker00_repeat",
        "restore_L00",
        "all_down_original",
    ]
    assert manifest["worker_id"] == 0
    assert manifest["physical_gpu_id"] == "0"
    assert manifest["mode"] == "smoke"
    assert manifest["write_weight_metrics"] is True
    jobs = manifest["jobs"]
    assert jobs[0]["restore_layers"] == []
    assert jobs[1]["restore_layers"] == []
    assert jobs[2]["restore_layers"] == [0]
    assert jobs[3]["restore_layers"] == list(range(36))
    for job in jobs:
        assert job["mode"] == "smoke"
        assert job["lm_limit"] == 2
    validate_manifest(manifest, worker_id=0, physical_gpu_id="0")


def test_smoke_rejects_multiple_gpus():
    from experiments.down_layer_sensitivity.run import build_phase1_manifests

    with pytest.raises(ValueError, match="smoke"):
        build_phase1_manifests(selected_gpus=["0", "1"], mode="smoke")


def test_launch_phase_workers_writes_manifests_and_fixed_command(tmp_path, monkeypatch):
    from experiments.down_layer_sensitivity import run as run_mod

    captured = []

    class _Proc:
        def wait(self):
            return 0

    def _fake_popen(cmd, env=None, cwd=None):
        captured.append({"cmd": list(cmd), "env": dict(env), "cwd": cwd})
        return _Proc()

    monkeypatch.setattr(run_mod.subprocess, "Popen", _fake_popen)

    selected_gpus = ["3", "5"]
    manifests = run_mod.build_phase1_manifests(selected_gpus=selected_gpus, mode="formal")
    phase_dir = tmp_path / "phase1"
    run_mod.launch_phase_workers(
        checkpoint_dir="/ckpt",
        phase_dir=str(phase_dir),
        selected_gpus=selected_gpus,
        manifests=manifests,
    )

    assert len(captured) == 2
    repo_root = run_mod.REPO_ROOT
    for worker_id, gpu in enumerate(selected_gpus):
        manifest_path = phase_dir / "manifests" / f"worker_{worker_id:02d}.json"
        assert manifest_path.is_file()
        with open(manifest_path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert on_disk["write_weight_metrics"] is (worker_id == 0)
        assert on_disk["jobs"][0]["job_id"] == f"compressed_baseline_worker{worker_id:02d}"
        record = captured[worker_id]
        assert record["cwd"] == repo_root
        assert record["env"]["CUDA_VISIBLE_DEVICES"] == gpu
        assert record["cmd"] == [
            run_mod.sys.executable,
            "experiments/down_layer_sensitivity/worker.py",
            "--checkpoint_dir",
            "/ckpt",
            "--manifest_path",
            str(manifest_path),
            "--jobs_dir",
            str(phase_dir / "jobs"),
            "--worker_meta_path",
            str(phase_dir / "worker_logs" / f"worker_{worker_id:02d}_meta.json"),
            "--worker_id",
            str(worker_id),
            "--physical_gpu_id",
            gpu,
        ]


def test_launch_phase_workers_failed_exit_writes_failed_workers(tmp_path, monkeypatch):
    from experiments.down_layer_sensitivity import run as run_mod

    class _Proc:
        def __init__(self, code):
            self._code = code

        def wait(self):
            return self._code

    codes = [0, 1]

    def _fake_popen(cmd, env=None, cwd=None):
        return _Proc(codes.pop(0))

    monkeypatch.setattr(run_mod.subprocess, "Popen", _fake_popen)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_config_path = run_dir / "run_config.json"
    run_config_path.write_text(json.dumps({"status": "running"}) + "\n", encoding="utf-8")
    selected_gpus = ["0", "1"]
    manifests = run_mod.build_phase1_manifests(selected_gpus=selected_gpus, mode="formal")

    with pytest.raises(SystemExit) as exc:
        run_mod.launch_phase_workers(
            checkpoint_dir="/ckpt",
            phase_dir=str(run_dir / "phase1"),
            selected_gpus=selected_gpus,
            manifests=manifests,
        )
    assert exc.value.code == 1
    with open(run_config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["status"] == "failed"
    assert config["failed_workers"] == [1]


PHASE2_RANKED = list(range(35, -1, -1))
PHASE2_RANDOM8_LAYERS = {
    31: [0, 1, 4, 7, 9, 21, 25, 30],
    32: [0, 4, 9, 13, 15, 19, 23, 31],
    33: [10, 11, 14, 17, 19, 20, 28, 30],
    34: [1, 11, 12, 14, 22, 28, 33, 35],
    35: [8, 9, 13, 21, 31, 33, 34, 35],
}
PHASE2_TOP_LAYERS = {
    "top2": [35, 34],
    "top4": [35, 34, 33, 32],
    "top8": [35, 34, 33, 32, 31, 30, 29, 28],
    "top12": [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24],
}
EXPECTED_PHASE2_JOB_IDS = {
    1: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            "top2",
            "top4",
            "top8",
            "top12",
            "random8_seed31",
            "random8_seed32",
            "random8_seed33",
            "random8_seed34",
            "random8_seed35",
        ],
    },
    2: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            "top4",
            "top12",
            "random8_seed32",
            "random8_seed34",
        ],
        1: [
            "compressed_baseline_worker01",
            "top2",
            "top8",
            "random8_seed31",
            "random8_seed33",
            "random8_seed35",
        ],
    },
    4: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            "top12",
            "random8_seed34",
        ],
        1: [
            "compressed_baseline_worker01",
            "top2",
            "random8_seed31",
            "random8_seed35",
        ],
        2: [
            "compressed_baseline_worker02",
            "top4",
            "random8_seed32",
        ],
        3: [
            "compressed_baseline_worker03",
            "top8",
            "random8_seed33",
        ],
    },
    8: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            "random8_seed34",
        ],
        1: ["compressed_baseline_worker01", "top2", "random8_seed35"],
        2: ["compressed_baseline_worker02", "top4"],
        3: ["compressed_baseline_worker03", "top8"],
        4: ["compressed_baseline_worker04", "top12"],
        5: ["compressed_baseline_worker05", "random8_seed31"],
        6: ["compressed_baseline_worker06", "random8_seed32"],
        7: ["compressed_baseline_worker07", "random8_seed33"],
    },
    9: {
        0: [
            "compressed_baseline_worker00",
            "compressed_baseline_worker00_repeat",
            "random8_seed35",
        ],
        1: ["compressed_baseline_worker01", "top2"],
        2: ["compressed_baseline_worker02", "top4"],
        3: ["compressed_baseline_worker03", "top8"],
        4: ["compressed_baseline_worker04", "top12"],
        5: ["compressed_baseline_worker05", "random8_seed31"],
        6: ["compressed_baseline_worker06", "random8_seed32"],
        7: ["compressed_baseline_worker07", "random8_seed33"],
        8: ["compressed_baseline_worker08", "random8_seed34"],
    },
}


def _phase2_restore_layers(job_id: str) -> list[int]:
    if job_id.startswith("compressed_baseline_"):
        return []
    if job_id in PHASE2_TOP_LAYERS:
        return list(PHASE2_TOP_LAYERS[job_id])
    if job_id.startswith("random8_seed"):
        seed = int(job_id.removeprefix("random8_seed"))
        return list(PHASE2_RANDOM8_LAYERS[seed])
    raise AssertionError(f"unexpected phase-2 job_id {job_id}")


@pytest.mark.parametrize("num_gpus", [1, 2, 4, 8, 9])
def test_formal_phase2_manifest_allocation_is_deterministic(num_gpus):
    from experiments.down_layer_sensitivity.run import build_phase2_manifests
    from experiments.down_layer_sensitivity.worker import validate_manifest

    selected_gpus = [str(i) for i in range(num_gpus)]
    manifests = build_phase2_manifests(
        selected_gpus=selected_gpus,
        ranked_layers=PHASE2_RANKED,
    )
    expected = EXPECTED_PHASE2_JOB_IDS[num_gpus]
    w2 = num_gpus
    assert len(manifests) == w2
    all_ids = _all_job_ids(manifests)
    assert len(all_ids) == 9 + w2 + 1
    assert len(set(all_ids)) == 9 + w2 + 1
    assert "top1" not in all_ids
    assert all_ids.count("top2") == 1
    assert all_ids.count("top12") == 1
    for seed in (31, 32, 33, 34, 35):
        assert all_ids.count(f"random8_seed{seed}") == 1

    for worker_id, manifest in enumerate(manifests):
        assert manifest["worker_id"] == worker_id
        assert manifest["physical_gpu_id"] == selected_gpus[worker_id]
        assert manifest["mode"] == "formal"
        assert manifest["write_weight_metrics"] is False
        job_ids = _job_ids(manifest)
        assert job_ids == expected[worker_id]
        assert job_ids[0] == f"compressed_baseline_worker{worker_id:02d}"
        if worker_id == 0:
            assert job_ids[1] == "compressed_baseline_worker00_repeat"
        for index, job_id in enumerate(job_ids):
            if not job_id.startswith("compressed_baseline_"):
                assert index > 0
                if worker_id == 0:
                    assert index > 1
        for job in manifest["jobs"]:
            assert job["mode"] == "formal"
            assert job["lm_limit"] is None
            assert job["restore_layers"] == _phase2_restore_layers(job["job_id"])
        validate_manifest(
            manifest,
            worker_id=worker_id,
            physical_gpu_id=selected_gpus[worker_id],
        )

    again = build_phase2_manifests(
        selected_gpus=selected_gpus,
        ranked_layers=PHASE2_RANKED,
    )
    assert json.dumps(manifests, ensure_ascii=False) == json.dumps(again, ensure_ascii=False)


def test_phase2_uses_first_w2_gpus_when_more_than_nine_selected():
    from experiments.down_layer_sensitivity.run import build_phase2_manifests

    selected_gpus = [str(i) for i in range(10)]
    manifests = build_phase2_manifests(
        selected_gpus=selected_gpus,
        ranked_layers=PHASE2_RANKED,
    )
    assert len(manifests) == 9
    assert [m["physical_gpu_id"] for m in manifests] == [str(i) for i in range(9)]
    assert _all_job_ids(manifests) == _all_job_ids(
        build_phase2_manifests(
            selected_gpus=selected_gpus[:9],
            ranked_layers=PHASE2_RANKED,
        )
    )
    assert "9" not in [m["physical_gpu_id"] for m in manifests]


def test_phase2_rejects_ranked_layers_that_are_not_permutation_of_0_35():
    from experiments.down_layer_sensitivity.run import build_phase2_manifests

    with pytest.raises(ValueError, match="ranked"):
        build_phase2_manifests(
            selected_gpus=["0"],
            ranked_layers=list(range(35)),
        )
    with pytest.raises(ValueError, match="ranked"):
        build_phase2_manifests(
            selected_gpus=["0"],
            ranked_layers=list(range(36)) + [0],
        )
    with pytest.raises(ValueError, match="ranked"):
        build_phase2_manifests(
            selected_gpus=["0"],
            ranked_layers=[0] * 36,
        )


def test_phase2_rejects_empty_selected_gpus():
    from experiments.down_layer_sensitivity.run import build_phase2_manifests

    with pytest.raises(ValueError, match="selected_gpus"):
        build_phase2_manifests(selected_gpus=[], ranked_layers=list(range(36)))


def test_formal_main_phase2_launches_on_first_w2_gpus(tmp_path, monkeypatch):
    from experiments.down_layer_sensitivity import run as run_mod
    import experiments.down_layer_sensitivity.summarize as summarize_mod

    launches = []

    def fake_launch(*, checkpoint_dir, phase_dir, selected_gpus, manifests):
        launches.append(
            {
                "phase_dir": phase_dir,
                "selected_gpus": list(selected_gpus),
                "n_manifests": len(manifests),
                "write_weight_metrics": [m["write_weight_metrics"] for m in manifests],
            }
        )

    monkeypatch.setattr(run_mod, "launch_phase_workers", fake_launch)
    monkeypatch.setattr(summarize_mod, "summarize_phase1", lambda **kwargs: list(PHASE2_RANKED))
    monkeypatch.setattr(summarize_mod, "summarize_final", lambda **kwargs: None)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    run_mod.main(
        [
            "--checkpoint_dir",
            "/ckpt",
            "--output_dir",
            str(output_dir),
            "--gpus",
            "0,1,2,3,4,5,6,7,8,9",
            "--mode",
            "formal",
        ]
    )
    assert len(launches) == 2
    assert launches[0]["selected_gpus"] == [str(i) for i in range(10)]
    assert launches[1]["selected_gpus"] == [str(i) for i in range(9)]
    assert launches[1]["n_manifests"] == 9
    assert launches[1]["write_weight_metrics"] == [False] * 9
    assert launches[1]["phase_dir"].endswith("/phase2")
    run_dir = next(output_dir.iterdir())
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["phase2_worker_count"] == 9
    assert config["status"] == "completed"


def test_formal_main_sets_failed_when_phase1_summary_raises(tmp_path, monkeypatch):
    from experiments.down_layer_sensitivity import run as run_mod
    import experiments.down_layer_sensitivity.summarize as summarize_mod

    monkeypatch.setattr(run_mod, "launch_phase_workers", lambda **kwargs: None)

    def _raise(**kwargs):
        raise ValueError("A_all_down_original <= A_compressed")

    monkeypatch.setattr(summarize_mod, "summarize_phase1", _raise)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="A_all_down_original <= A_compressed"):
        run_mod.main(
            [
                "--checkpoint_dir",
                "/ckpt",
                "--output_dir",
                str(output_dir),
                "--gpus",
                "0,1",
                "--mode",
                "formal",
            ]
        )
    run_dir = next(output_dir.iterdir())
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "failed"
