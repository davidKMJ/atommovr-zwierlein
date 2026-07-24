"""Tests for SessionRecorder stage dumps and Camera hook integration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.errormodels import ZeroNoise
from awg_controller.src.awg_control import AWGBatch, RFRamp
from awg_controller.src.camera import GaussianCameraConfig, OfflineArrayCamera
from awg_controller.src.session_recorder import (
    GifOptions,
    SessionRecorder,
    SpectrogramOptions,
    VisualizationOptions,
    moves_to_records,
)
from atommovr.utils.Move import Move


def _two_tone_round(f_end_ch0=75e6, f_end_ch1=65e6, duration_s=5e-6):
    """A single ``AWGBatch`` moving one tone per channel, for spectrogram tests."""
    ramps = [
        RFRamp(
            channel=0,
            core=0,
            f_start=70e6,
            f_end=f_end_ch0,
            amplitude_pct=40.0,
            tone_index=0,
        ),
        RFRamp(
            channel=1,
            core=0,
            f_start=60e6,
            f_end=f_end_ch1,
            amplitude_pct=40.0,
            tone_index=0,
        ),
    ]
    return [AWGBatch(ramps=ramps, total_duration_s=duration_s)]


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
        rec = SessionRecorder(tmp_path, run_dir=tmp_path / "gif_meta", gif=gif)
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


class TestSpectrogram:
    def test_disabled_by_default_is_noop(self, tmp_path):
        rec = SessionRecorder(tmp_path, run_dir=tmp_path / "spec_off")
        rec.begin_round(0)
        assert rec.save_spectrogram(_two_tone_round(), sample_rate_hz=20e6) is None
        assert not (rec.run_dir / "round_00_spectrogram").exists()

    def test_empty_batches_is_noop(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_empty",
            spectrogram=SpectrogramOptions(enabled=True),
        )
        rec.begin_round(0)
        assert rec.save_spectrogram([], sample_rate_hz=20e6) is None

    def test_missing_sample_rate_raises(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_no_rate",
            spectrogram=SpectrogramOptions(enabled=True),
        )
        rec.begin_round(0)
        with pytest.raises(ValueError):
            rec.save_spectrogram(_two_tone_round())

    def test_writes_png_and_waveforms(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_on",
            spectrogram=SpectrogramOptions(
                enabled=True, nperseg=32, channel_labels={0: "V/row", 1: "H/col"}
            ),
        )
        rec.begin_round(2)
        stage_dir = rec.save_spectrogram(_two_tone_round(), sample_rate_hz=20e6)
        assert stage_dir is not None
        assert stage_dir.name == "round_02_spectrogram"
        assert (stage_dir / "spectrogram.png").is_file()
        wf0 = np.load(stage_dir / "waveform_ch0_with_holding.npy")
        wf1 = np.load(stage_dir / "waveform_ch1_with_holding.npy")
        assert wf0.shape == wf1.shape == (100,)  # 5us * 20MHz
        assert np.max(np.abs(wf0)) <= 0.40 + 1e-9  # amplitude_pct/100 ceiling

    def test_default_sample_rate_from_options(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_default_rate",
            spectrogram=SpectrogramOptions(enabled=True, sample_rate_hz=20e6),
        )
        rec.begin_round(0)
        stage_dir = rec.save_spectrogram(_two_tone_round())
        assert stage_dir is not None
        assert (stage_dir / "spectrogram.png").is_file()

    def test_freq_limits_kwarg_and_options(self, tmp_path):
        """AOD freq_min/freq_max must be accepted from options and kwargs."""
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_freq",
            spectrogram=SpectrogramOptions(
                enabled=True,
                sample_rate_hz=20e6,
                nperseg=32,
                freq_min_hz=50e6,
                freq_max_hz=90e6,
            ),
        )
        rec.begin_round(0)
        assert rec.save_spectrogram(_two_tone_round()) is not None
        # kwargs override options
        assert (
            rec.save_spectrogram(_two_tone_round(), freq_min_hz=60e6, freq_max_hz=80e6)
            is not None
        )

    def test_moves_only_silences_holding_tones(self, tmp_path):
        """_rf_batches_moves_only zeros static holds so save_spectrogram's
        "no holding" waveform is movers-only, while its "with holding"
        waveform keeps everything — both are always written now (three-panel
        spectrogram), not gated by a toggle.
        """
        from awg_controller.src.awg_control import AWGBatch, RFRamp
        from awg_controller.src.session_recorder import _rf_batches_moves_only
        from awg_controller.src.scapp import synthesize_round_waveform

        # Keep tones well below Nyquist (fs=20 MHz) and off integer-fs
        # multiples so holds are visible in the time series.
        batches = [
            AWGBatch(
                ramps=[
                    RFRamp(
                        channel=0,
                        core=0,
                        f_start=3e6,
                        f_end=4e6,
                        amplitude_pct=20.0,
                        tone_index=0,
                    ),
                    RFRamp(
                        channel=0,
                        core=1,
                        f_start=5e6,
                        f_end=5e6,  # hold
                        amplitude_pct=20.0,
                        tone_index=1,
                    ),
                ],
                total_duration_s=5e-6,
            )
        ]
        filtered = _rf_batches_moves_only(batches)
        assert [r.amplitude_pct for r in filtered[0].ramps] == [20.0, 0.0]

        rate = 20e6
        wf_moves = synthesize_round_waveform(filtered, rate, ramp_shape="linear")[0]
        wf_all = synthesize_round_waveform(batches, rate, ramp_shape="linear")[0]
        assert np.sqrt(np.mean(wf_all**2)) > np.sqrt(np.mean(wf_moves**2)) * 1.2

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_moves",
            spectrogram=SpectrogramOptions(
                enabled=True, sample_rate_hz=rate, nperseg=32
            ),
        )
        rec.begin_round(0)
        stage = rec.save_spectrogram(batches)
        assert stage is not None
        assert np.allclose(np.load(stage / "waveform_ch0_no_holding.npy"), wf_moves)
        assert np.allclose(np.load(stage / "waveform_ch0_with_holding.npy"), wf_all)

    def test_regression_three_panels_per_channel_properly_labeled(
        self, tmp_path, monkeypatch
    ):
        """save_spectrogram must render exactly 3 rows per channel — pure
        f(t), moves-only STFT, with-holding STFT — each titled distinctly.
        Row 1 (pure f(t)) must plot *every* tone (moving and non-moving —
        a never-moved row/col is real information, not clutter to hide).
        """
        from matplotlib.figure import Figure

        captured = {}
        orig_savefig = Figure.savefig

        def _capture(self, *args, **kwargs):
            captured["fig"] = self
            return orig_savefig(self, *args, **kwargs)

        monkeypatch.setattr(Figure, "savefig", _capture)

        ramps = [
            RFRamp(
                channel=0,
                core=0,
                f_start=3e6,
                f_end=4e6,
                amplitude_pct=20.0,
                tone_index=0,
            ),
            RFRamp(
                channel=0,
                core=1,
                f_start=5e6,
                f_end=5e6,
                amplitude_pct=20.0,
                tone_index=1,
            ),  # hold
            RFRamp(
                channel=1,
                core=0,
                f_start=2e6,
                f_end=2.5e6,
                amplitude_pct=20.0,
                tone_index=0,
            ),
        ]
        batches = [AWGBatch(ramps=ramps, total_duration_s=5e-6)]
        rate = 20e6
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_three_panel",
            spectrogram=SpectrogramOptions(
                enabled=True, sample_rate_hz=rate, ramp_shape="scurve"
            ),
        )
        rec.begin_round(0)
        stage = rec.save_spectrogram(batches)
        assert stage is not None

        fig = captured["fig"]
        # 2 channels x 3 rows = 6 data axes, plus one shared colorbar axes.
        assert len(fig.axes) == 7

        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles.count("Commanded frequency f(t) (scurve)") == 2
        assert titles.count("AWG output spectrogram (moves only)") == 2
        assert titles.count("AWG output spectrogram (with holding)") == 2

        # Row 0, channel 0 has one moving tone (core 0) and one holding tone
        # (core 1) -- both must be drawn now, styled distinctly (moving
        # tones bolder/opaque, non-moving dim/thin) so neither is hidden.
        ax_f_t_ch0 = fig.axes[0]
        assert len(ax_f_t_ch0.lines) == 2
        alphas = sorted(line.get_alpha() for line in ax_f_t_ch0.lines)
        assert alphas == [0.35, 0.85]

        # Row 0, channel 1 has exactly one (moving) tone.
        ax_f_t_ch1 = fig.axes[1]
        assert len(ax_f_t_ch1.lines) == 1

    def test_include_stft_false_writes_only_f_t_panel_no_sample_rate_needed(
        self, tmp_path
    ):
        """include_stft=False must skip waveform synthesis/STFT entirely --
        no sample_rate_hz required, only spectrogram_f_t.png written, no
        waveform .npy files.
        """
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_no_stft",
            spectrogram=SpectrogramOptions(enabled=True, include_stft=False),
        )
        rec.begin_round(0)
        # Deliberately no sample_rate_hz kwarg and no SpectrogramOptions.sample_rate_hz.
        stage = rec.save_spectrogram(_two_tone_round())
        assert stage is not None
        files = sorted(p.name for p in stage.iterdir())
        assert files == ["spectrogram_f_t.png"]

    def test_include_stft_false_never_calls_waveform_synthesis(
        self, tmp_path, monkeypatch
    ):
        import awg_controller.src.scapp as scapp_mod

        def _boom(*args, **kwargs):
            raise AssertionError("synthesize_round_waveform must not be called")

        monkeypatch.setattr(scapp_mod, "synthesize_round_waveform", _boom)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_no_stft_no_synth",
            spectrogram=SpectrogramOptions(enabled=True, include_stft=False),
        )
        rec.begin_round(0)
        assert rec.save_spectrogram(_two_tone_round()) is not None

    def test_include_stft_true_is_default(self, tmp_path):
        """include_stft defaults to True -- existing combined-figure
        behavior must be unchanged when callers don't set it."""
        opts = SpectrogramOptions()
        assert opts.include_stft is True

    def test_separate_holding_stft_writes_three_files_not_one(self, tmp_path):
        """separate_holding_stft=True must switch output from one combined
        spectrogram.png to three independent files, and vice versa."""
        rate = 20e6
        batches = _two_tone_round(duration_s=5e-6)

        rec_combined = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_combined",
            spectrogram=SpectrogramOptions(
                enabled=True, sample_rate_hz=rate, separate_holding_stft=False
            ),
        )
        rec_combined.begin_round(0)
        stage_combined = rec_combined.save_spectrogram(batches)
        assert (stage_combined / "spectrogram.png").is_file()
        assert not (stage_combined / "spectrogram_f_t.png").exists()
        assert not (stage_combined / "spectrogram_moves_only.png").exists()
        assert not (stage_combined / "spectrogram_with_holding.png").exists()

        rec_separate = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_separate",
            spectrogram=SpectrogramOptions(
                enabled=True, sample_rate_hz=rate, separate_holding_stft=True
            ),
        )
        rec_separate.begin_round(0)
        stage_separate = rec_separate.save_spectrogram(batches)
        assert not (stage_separate / "spectrogram.png").exists()
        assert (stage_separate / "spectrogram_f_t.png").is_file()
        assert (stage_separate / "spectrogram_moves_only.png").is_file()
        assert (stage_separate / "spectrogram_with_holding.png").is_file()

    def test_separate_holding_stft_uses_independent_larger_window(
        self, tmp_path, monkeypatch
    ):
        """The whole point of separate_holding_stft: row 3 (with holding)
        must get a window sized purely from target_df_hz (no shortest-ramp
        time cap), which is larger than row 2's (moves only) time-capped
        window — better frequency resolution for closely-spaced static
        tones, at the cost of time resolution it doesn't need.
        """
        import scipy.signal as sig

        captured_npersegs = []
        orig = sig.spectrogram

        def _capture(*args, **kwargs):
            captured_npersegs.append(kwargs.get("nperseg"))
            return orig(*args, **kwargs)

        monkeypatch.setattr(sig, "spectrogram", _capture)

        rate = 20e6
        batches = _two_tone_round(duration_s=5e-6)
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_sep_window",
            spectrogram=SpectrogramOptions(
                enabled=True, sample_rate_hz=rate, separate_holding_stft=True
            ),
        )
        rec.begin_round(0)
        stage = rec.save_spectrogram(batches)
        assert stage is not None

        # Per channel, _stft is called (moves-only, with-holding) in order.
        assert len(captured_npersegs) == 4
        moves_npersegs = captured_npersegs[0::2]
        holding_npersegs = captured_npersegs[1::2]
        assert moves_npersegs != holding_npersegs
        assert all(h > m for h, m in zip(holding_npersegs, moves_npersegs))

    def test_regression_nperseg_shrinks_for_short_ramps(self):
        """A window sized only from target_df_hz can end up longer than a
        short move itself (a single-site move can be a few µs at typical
        synthesis rates) — the STFT can't then resolve the sweep within one
        window, so the chirp reads as a blocky/discrete patch instead of a
        smooth diagonal line. Auto-selected nperseg must stay well inside
        the shortest nonzero-duration batch in the round.
        """
        from awg_controller.src.session_recorder import (
            _choose_nperseg,
            _min_batch_samples,
        )

        rate = 4 * 118e6  # matches the tutorial notebook's AOD band
        short_batch = _two_tone_round(duration_s=3.01e-6)  # ~1-site move
        min_samples = _min_batch_samples(short_batch, rate)
        max_window = min_samples // 8

        nperseg_uncapped = _choose_nperseg(
            rate, 200_000, nperseg=None, target_df_hz=50e3
        )
        nperseg_capped = _choose_nperseg(
            rate,
            200_000,
            nperseg=None,
            target_df_hz=50e3,
            max_window_samples=max_window,
        )

        assert nperseg_uncapped > min_samples  # old behavior: window > whole ramp
        assert nperseg_capped <= max_window
        assert min_samples / nperseg_capped >= 8  # several windows resolve the sweep

    def test_regression_short_ramp_gets_multiple_stft_windows(self, tmp_path):
        """End-to-end: save_spectrogram's auto-selected nperseg must not
        exceed a short ramp's own sample count.
        """
        from awg_controller.src.session_recorder import (
            _choose_nperseg,
            _min_batch_samples,
        )

        rate = 4 * 118e6
        batches = _two_tone_round(duration_s=3.01e-6)
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_short_ramp",
            spectrogram=SpectrogramOptions(enabled=True, sample_rate_hz=rate),
        )
        rec.begin_round(0)
        stage = rec.save_spectrogram(batches)
        assert stage is not None
        wf = np.load(stage / "waveform_ch0_with_holding.npy")

        min_samples = _min_batch_samples(batches, rate)
        nperseg = _choose_nperseg(
            rate,
            wf.size,
            nperseg=None,
            target_df_hz=50e3,
            max_window_samples=min_samples // 8,
        )
        assert nperseg < min_samples

    def test_regression_time_axis_spans_full_duration(self, tmp_path, monkeypatch):
        """The plotted time axis must cover ``[0, round duration]``, not
        scipy's STFT segment-center times — those crop up to
        ``nperseg / (2 * fs)`` off each end, which silently hides real
        signal near the start/end of the round (worse for large windows
        relative to a short round).
        """
        from matplotlib.figure import Figure

        captured = {}
        orig_savefig = Figure.savefig

        def _capture(self, *args, **kwargs):
            # self.axes[0] is the first data subplot (sharex=True keeps both
            # channels in sync); the colorbar's own axes gets appended after
            # the data subplots, so self.axes[-1] is *not* a data axis.
            captured["xlim"] = self.axes[0].get_xlim()
            return orig_savefig(self, *args, **kwargs)

        monkeypatch.setattr(Figure, "savefig", _capture)

        # In-band tones (well under the 10 MHz Nyquist for rate=20e6) so the
        # draw loop's frequency-band mask isn't empty and both axes get a
        # real set_xlim() call.
        rate = 20e6
        ramps = [
            RFRamp(
                channel=0,
                core=0,
                f_start=3e6,
                f_end=4e6,
                amplitude_pct=40.0,
                tone_index=0,
            ),
            RFRamp(
                channel=1,
                core=0,
                f_start=2e6,
                f_end=3e6,
                amplitude_pct=40.0,
                tone_index=0,
            ),
        ]
        batches = [AWGBatch(ramps=ramps, total_duration_s=5e-6)]
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_time_axis",
            spectrogram=SpectrogramOptions(
                enabled=True, sample_rate_hz=rate, nperseg=64
            ),
        )
        rec.begin_round(0)
        stage = rec.save_spectrogram(batches)
        assert stage is not None
        wf = np.load(stage / "waveform_ch0_with_holding.npy")
        duration_us = wf.size / rate * 1e6

        xlim = captured["xlim"]
        assert xlim[0] == pytest.approx(0.0, abs=1e-9)
        assert xlim[1] == pytest.approx(duration_us, rel=1e-6)

    def test_regression_spectrogram_does_not_switch_mpl_backend(self, tmp_path):
        """save_spectrogram must not call matplotlib.use() (breaks notebook plt.show)."""
        import matplotlib

        before = matplotlib.get_backend()
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "spec_backend",
            spectrogram=SpectrogramOptions(enabled=True, sample_rate_hz=20e6),
        )
        rec.begin_round(0)
        assert rec.save_spectrogram(_two_tone_round()) is not None
        assert matplotlib.get_backend() == before


def _tiny_atom_array(rows=4, cols=4, seed=0):
    arr = AtomArray([rows, cols], n_species=1, error_model=ZeroNoise(seed=seed))
    arr.matrix[:, :, 0] = 1  # fully loaded: deterministic, no loading-probability noise
    return arr


def _one_move_batches(n=1):
    return [[Move(0, 0, 1, 1)] for _ in range(n)]


def _fake_grid(atom_array, batches, save_path=None, title_suffix="", max_cols=3):
    Path(save_path).write_text("x")


class TestMoveVisualization:
    def test_disabled_by_default_is_noop(self, tmp_path):
        rec = SessionRecorder(tmp_path, run_dir=tmp_path / "viz_off")
        rec.begin_round(0)
        assert (
            rec.save_move_visualization(_tiny_atom_array(), _one_move_batches()) is None
        )
        assert not (rec.run_dir / "round_00_visualization").exists()

    def test_empty_move_batches_is_noop(self, tmp_path):
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_empty",
            visualization=VisualizationOptions(enabled=True),
        )
        rec.begin_round(0)
        assert rec.save_move_visualization(_tiny_atom_array(), []) is None

    def test_writes_grid_and_gif_by_default(self, tmp_path, monkeypatch):
        """gif defaults to True -- both the static grid.svg and the
        animated grid.gif must be written."""
        import atommovr.utils.imaging.visualization as viz_mod

        calls = []

        def _fake_grid_tracking(
            atom_array, batches, save_path=None, title_suffix="", max_cols=3
        ):
            calls.append(len(batches))
            Path(save_path).write_text("x")

        def _fake_frames(atom_array, batches, **kwargs):
            calls.append(len(batches))
            # 3x3x3 uint8 "frames" -- enough for _write_gif to accept.
            return [
                np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(len(batches) + 1)
            ]

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid_tracking)
        monkeypatch.setattr(viz_mod, "render_move_batch_frames", _fake_frames)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_both",
            visualization=VisualizationOptions(enabled=True),
        )
        rec.begin_round(3)
        stage = rec.save_move_visualization(_tiny_atom_array(), _one_move_batches(2))
        assert stage is not None
        assert stage.name == "round_03_visualization"
        assert (stage / "grid.svg").is_file()
        assert (stage / "grid.gif").is_file()
        assert calls == [2, 2]

    def test_gif_false_skips_gif_and_frame_rendering(self, tmp_path, monkeypatch):
        import atommovr.utils.imaging.visualization as viz_mod

        def _boom(*args, **kwargs):
            raise AssertionError("render_move_batch_frames must not be called")

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid)
        monkeypatch.setattr(viz_mod, "render_move_batch_frames", _boom)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_no_gif",
            visualization=VisualizationOptions(enabled=True, gif=False),
        )
        rec.begin_round(0)
        stage = rec.save_move_visualization(_tiny_atom_array(), _one_move_batches(1))
        assert (stage / "grid.svg").is_file()
        assert not (stage / "grid.gif").exists()

    def test_regression_max_batches_truncates(self, tmp_path, monkeypatch):
        """A round with many small parallel-move batches would otherwise
        render one enormous, unreadable figure -- max_batches must cap how
        many batches actually get visualized (both grid and gif)."""
        import atommovr.utils.imaging.visualization as viz_mod

        captured = {}

        def _fake_grid_tracking(
            atom_array, batches, save_path=None, title_suffix="", max_cols=3
        ):
            captured["grid_n_batches"] = len(batches)
            Path(save_path).write_text("x")

        def _fake_frames(atom_array, batches, **kwargs):
            captured["gif_n_batches"] = len(batches)
            return [
                np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(len(batches) + 1)
            ]

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid_tracking)
        monkeypatch.setattr(viz_mod, "render_move_batch_frames", _fake_frames)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_trunc",
            visualization=VisualizationOptions(enabled=True, max_batches=3),
        )
        rec.begin_round(0)
        rec.save_move_visualization(_tiny_atom_array(), _one_move_batches(10))
        assert captured["grid_n_batches"] == 3
        assert captured["gif_n_batches"] == 3

    def test_max_batches_none_disables_cap(self, tmp_path, monkeypatch):
        import atommovr.utils.imaging.visualization as viz_mod

        captured = {}

        def _fake_grid_tracking(
            atom_array, batches, save_path=None, title_suffix="", max_cols=3
        ):
            captured["n_batches"] = len(batches)
            Path(save_path).write_text("x")

        monkeypatch.setattr(viz_mod, "visualize_move_batches", _fake_grid_tracking)

        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_no_cap",
            visualization=VisualizationOptions(
                enabled=True, gif=False, max_batches=None
            ),
        )
        rec.begin_round(0)
        rec.save_move_visualization(_tiny_atom_array(), _one_move_batches(10))
        assert captured["n_batches"] == 10

    def test_regression_real_rendering_small_array(self, tmp_path):
        """End-to-end with the real atommovr.utils.imaging.visualization
        functions (no mocking) on a tiny array/round, to catch integration
        issues (wrong AtomArray attributes, signature mismatches, etc.) a
        mocked test can't -- checks both the static figure and the gif.
        """
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "viz_real",
            visualization=VisualizationOptions(enabled=True),
        )
        rec.begin_round(0)
        stage = rec.save_move_visualization(
            _tiny_atom_array(rows=3, cols=3), _one_move_batches(2)
        )
        assert stage is not None
        grid_path = stage / "grid.svg"
        gif_path = stage / "grid.gif"
        assert grid_path.is_file()
        assert grid_path.stat().st_size > 0
        assert gif_path.is_file()
        assert gif_path.stat().st_size > 0


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
        with atommovrController(sw, HardwareConfig(), camera=cam, recorder=rec) as ctrl:
            assert ctrl.recorder is rec
            assert (
                not hasattr(cam, "recorder") or getattr(cam, "recorder", None) is None
            )
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

    def test_regression_offline_spectrogram_uses_hardware_max_rate(
        self, tmp_path, monkeypatch
    ):
        """In sim/offline mode (no live ScappFeeder), the controller must
        pass the real M4i.6631-x8 max sample rate to save_spectrogram — not
        an arbitrary multiplier of the AOD tone frequency — so the offline
        synthesis matches what real hardware would actually stream.
        """
        from atommovr.utils.core import PhysicalParams
        from awg_controller.scripts.atommover_controller import (
            HardwareConfig,
            SoftwareConfig,
            atommovrController,
        )
        from awg_controller.src.awg_control import (
            AODSettings,
            M4I_6631_X8_MAX_SAMPLE_RATE_HZ,
        )

        captured_rates = []
        orig = SessionRecorder.save_spectrogram

        def _capture(self, rf_batches, **kwargs):
            captured_rates.append(kwargs.get("sample_rate_hz"))
            return orig(self, rf_batches, **kwargs)

        monkeypatch.setattr(SessionRecorder, "save_spectrogram", _capture)

        rows, cols = 6, 5
        rec = SessionRecorder(
            tmp_path,
            run_dir=tmp_path / "ctrl_spec_rate",
            spectrogram=SpectrogramOptions(enabled=True),
            gif=GifOptions(enabled=False),
        )
        # Atoms confined to the outer rows only, so the (inner) middle-fill
        # target starts unmet — guarantees at least one round of real moves,
        # so save_spectrogram actually gets called (round 0 would otherwise
        # short-circuit as "already filled" before ever reaching it).
        initial_occ = np.zeros((rows, cols), dtype=int)
        initial_occ[0, :] = 1
        initial_occ[rows - 1, :] = 1
        cam = OfflineArrayCamera(
            (rows, cols),
            image_generator=GaussianCameraConfig(
                image_shape=(200, 200), min_spacing_px=20.0, noise_level=0.0
            ),
            initial_occupancy=initial_occ,
            physical_params=PhysicalParams(loading_prob=0.9, spacing=5e-6),
            seed=3,
        )
        sw = SoftwareConfig(
            max_rounds=3,
            algorithm_name="Hungarian",
            physical_params=PhysicalParams(
                loading_prob=0.9, spacing=5e-6, middle_size=[2, 2]
            ),
            error_model=ZeroNoise(seed=3),
            aod_settings=AODSettings(grid_rows=rows, grid_cols=cols),
        )
        with atommovrController(sw, HardwareConfig(), camera=cam, recorder=rec) as ctrl:
            assert ctrl._feeder is None  # sim mode: no live card
            ctrl.run()

        assert captured_rates, "save_spectrogram was never called"
        assert all(r == M4I_6631_X8_MAX_SAMPLE_RATE_HZ for r in captured_rates)

    def test_offline_camera_shim_reexports(self):
        from awg_controller.src import offline_camera as shim
        from awg_controller.src import camera as cam_mod

        assert shim.OfflineArrayCamera is cam_mod.OfflineArrayCamera
        assert shim.GaussianCameraConfig is cam_mod.GaussianCameraConfig
