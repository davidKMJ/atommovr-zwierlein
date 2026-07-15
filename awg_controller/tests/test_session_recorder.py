"""Tests for SessionRecorder stage dumps and Camera hook integration."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.errormodels import ZeroNoise
from awg_controller.src.camera import GaussianCameraConfig, OfflineArrayCamera
from awg_controller.src.session_recorder import (
    GifOptions,
    SessionRecorder,
    moves_to_records,
)
from atommovr.utils.Move import Move


class TestSessionRecorder:
    def test_disabled_is_noop(self, tmp_path):
        rec = SessionRecorder(tmp_path, enabled=False)
        assert rec.run_dir is None
        rec.begin_round(0)
        assert rec.save_stage("acquire", frame=np.zeros((4, 4), dtype=np.uint8)) is None
        rec.log_round(atoms=1)
        assert list(tmp_path.iterdir()) == []

    def test_save_stage_and_log_round(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "run_test",
            meta={"grid": [4, 4]},
            gif=GifOptions(enabled=False),
        )
        assert (rec.run_dir / "meta.json").is_file()
        meta = json.loads((rec.run_dir / "meta.json").read_text())
        assert meta["grid"] == [4, 4]

        frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
        occ = np.eye(4, dtype=int)
        rec.begin_round(0)
        stage = rec.save_stage("acquire", frame=frame, occupancy=occ)
        assert stage is not None
        assert stage.name == "round_00_acquire"
        assert (stage / "frame.npy").is_file()
        assert (stage / "occupancy.npy").is_file()
        assert np.array_equal(np.load(stage / "occupancy.npy"), occ)

        rec.log_round(
            atoms=4,
            filled=2,
            need=4,
            n_moves=1,
            moves=[{"fr": 0, "fc": 0, "tr": 1, "tc": 1}],
        )
        lines = (rec.run_dir / "rounds.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["round"] == 0
        assert row["n_moves"] == 1
        assert row["moves"][0]["tr"] == 1

    def test_moves_to_records(self):
        batches = [[Move(0, 1, 2, 3), Move(1, 1, 1, 2)]]
        assert moves_to_records(batches) == [
            {"fr": 0, "fc": 1, "tr": 2, "tc": 3},
            {"fr": 1, "fc": 1, "tr": 1, "tc": 2},
        ]
        assert moves_to_records([]) == []

    def test_gif_options_written_to_meta(self, tmp_path):
        gif = GifOptions(
            enabled=True,
            sources=("occupancy",),
            stages=("detect",),
            duration_s=0.25,
            max_side=64,
            auto_write=False,
        )
        rec = SessionRecorder(
            tmp_path, run_dir=tmp_path / "gif_meta", gif=gif
        )
        meta = json.loads((rec.run_dir / "meta.json").read_text())
        assert meta["gif"]["sources"] == ["occupancy"]
        assert meta["gif"]["duration_s"] == 0.25
        assert meta["gif"]["auto_write"] is False

    def test_write_gifs_from_detect_stages(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "gif_run",
            gif=GifOptions(
                enabled=True,
                sources=("frame", "occupancy"),
                stages=("detect",),
                duration_s=0.2,
                max_side=128,
                occupancy_cell_px=8,
                auto_write=False,
            ),
        )
        for r in range(3):
            rec.begin_round(r)
            # acquire should be ignored by default stages filter
            rec.save_stage(
                "acquire",
                frame=np.full((32, 32), r * 40, dtype=np.uint8),
                occupancy=np.eye(4, dtype=int),
            )
            rec.save_stage(
                "detect",
                frame=np.full((32, 32), 50 + r * 40, dtype=np.uint8),
                occupancy=(np.arange(16).reshape(4, 4) > r).astype(int),
            )

        written = rec.finalize()
        assert "frame" in written
        assert "occupancy" in written
        assert (rec.run_dir / "frames.gif").is_file()
        assert (rec.run_dir / "occupancy.gif").is_file()
        assert len(rec._gif_frames["frame"]) == 3
        assert len(rec._gif_frames["occupancy"]) == 3

    def test_gif_disabled_skips_files(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "no_gif",
            gif=GifOptions(enabled=False),
        )
        rec.begin_round(0)
        rec.save_stage(
            "detect",
            frame=np.zeros((8, 8), dtype=np.uint8),
            occupancy=np.ones((2, 2), dtype=int),
        )
        assert rec.finalize() == {}
        assert not (rec.run_dir / "frames.gif").exists()
        assert not (rec.run_dir / "occupancy.gif").exists()


class TestCameraRecorderHook:
    def test_sync_writes_acquire_and_detect_stages(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "cam_run",
            gif=GifOptions(enabled=False),
        )
        occ0 = np.ones((4, 4), dtype=int)
        cam = OfflineArrayCamera(
            (4, 4),
            image_generator=GaussianCameraConfig(
                image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
            ),
            initial_occupancy=occ0,
            seed=0,
        )
        array = AtomArray((4, 4), n_species=1, error_model=ZeroNoise(seed=0))
        rec.begin_round(0)
        cam.sync(array, recorder=rec)

        acquire_dir = rec.run_dir / "round_00_acquire"
        detect_dir = rec.run_dir / "round_00_detect"
        assert acquire_dir.is_dir()
        assert detect_dir.is_dir()
        assert (acquire_dir / "frame.npy").is_file()
        assert (detect_dir / "occupancy.npy").is_file()
        det = np.load(detect_dir / "occupancy.npy")
        assert det.shape == (4, 4)
        assert int(det.sum()) == 16

    def test_no_recorder_leaves_tmp_empty(self, tmp_path):
        cam = OfflineArrayCamera(
            (4, 4),
            image_generator=GaussianCameraConfig(
                image_shape=(128, 128), min_spacing_px=16.0, noise_level=0.0
            ),
            initial_occupancy=np.ones((4, 4), dtype=int),
            seed=0,
        )
        array = AtomArray((4, 4), n_species=1, error_model=ZeroNoise(seed=0))
        cam.sync(array)
        assert list(tmp_path.iterdir()) == []


class TestControllerOwnsRecorder:
    def test_controller_passes_recorder_into_sync(self, tmp_path):
        from atommovr.utils.core import PhysicalParams
        from awg_controller.scripts.atommover_controller import (
            HardwareConfig,
            SoftwareConfig,
            atommovrController,
        )
        from awg_controller.src.awg_control import AODSettings

        rows, cols = 6, 5
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "ctrl_run",
            meta={"grid": [rows, cols]},
            gif=GifOptions(
                enabled=True,
                sources=("frame", "occupancy"),
                stages=("detect",),
                duration_s=0.15,
                max_side=128,
                auto_write=False,
            ),
        )
        cam = OfflineArrayCamera(
            (rows, cols),
            image_generator=GaussianCameraConfig(
                image_shape=(200, 200), min_spacing_px=20.0, noise_level=0.0
            ),
            physical_params=PhysicalParams(loading_prob=0.9, spacing=5e-6),
            seed=3,
        )
        sw = SoftwareConfig(
            max_rounds=2,
            algorithm_name="Hungarian",
            physical_params=PhysicalParams(
                loading_prob=0.9, spacing=5e-6, middle_size=[2, 2]
            ),
            error_model=ZeroNoise(seed=3),
            aod_settings=AODSettings(grid_rows=rows, grid_cols=cols),
        )
        with atommovrController(
            sw, HardwareConfig(), camera=cam, recorder=rec
        ) as ctrl:
            assert ctrl.recorder is rec
            assert not hasattr(cam, "recorder") or getattr(cam, "recorder", None) is None
            ok = ctrl.run()

        assert (rec.run_dir / "meta.json").is_file()
        assert (rec.run_dir / "rounds.jsonl").is_file()
        assert (rec.run_dir / "round_00_acquire" / "frame.npy").is_file()
        assert (rec.run_dir / "round_00_detect" / "occupancy.npy").is_file()
        # controller.run() calls recorder.finalize() in finally
        assert (rec.run_dir / "frames.gif").is_file()
        assert (rec.run_dir / "occupancy.gif").is_file()
        lines = [
            ln
            for ln in (rec.run_dir / "rounds.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) >= 1
        assert ok in (True, False)

    def test_offline_camera_shim_reexports(self):
        from awg_controller.src import offline_camera as shim
        from awg_controller.src import camera as cam_mod

        assert shim.OfflineArrayCamera is cam_mod.OfflineArrayCamera
        assert shim.GaussianCameraConfig is cam_mod.GaussianCameraConfig
