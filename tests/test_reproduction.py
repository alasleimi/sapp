import numpy as np
from sapp.experiments.common import Context, read_json


def test_receiver_rebuilds_published_power_ordered_peaks(root, tmp_path):
    from sapp.receiver import PeakFrontend, extract_delay_peaks
    from sapp.experiments.prepare import channel_frames

    settings = read_json(root / "configs/receiver.json")
    config = PeakFrontend(**settings["frontend"])
    frames = channel_frames(root / "data/nist/channels/L/LA_empty_survey/qdOutput.json.gz")
    seeds = settings["receiver_seeds"]["L_LA_empty_survey_standard"]
    with np.load(root / "data/nist/receiver/L_LA_empty_survey_standard_peaks.npz") as z:
        expected = z["ranges"]
    for i, frame in enumerate(frames):
        if i == 8:
            break
        actual = extract_delay_peaks(
            frame, rng=np.random.default_rng(seeds[i]), config=config, power_order=True
        )
        np.testing.assert_array_equal(actual, expected[i, np.isfinite(expected[i])])


def test_cnn_training_encodings_rebuild_from_surveys(root, tmp_path):
    from sapp.experiments.cnn import dataset

    ctx = Context(root, tmp_path)
    for room in ctx.rooms:
        x, y, ids, _ = dataset(ctx, room)
        with np.load(
            root / "reference/cnn/encodings/common" / f"{room['room_id']}.npz"
        ) as expected:
            np.testing.assert_array_equal(x, expected["inputs"])
            np.testing.assert_array_equal(y, expected["targets"])
            np.testing.assert_array_equal(ids, expected["coordinate_indices"])


def test_all_cnn_budget_encodings_rebuild(root, tmp_path):
    from sapp.experiments.cnn_budget import prepare_dataset

    ctx = Context(root, tmp_path)
    for room in read_json(root / "data/cnn/design.json")["rooms"]:
        for condition in ("layout_rich", "paper_diversity"):
            actual = prepare_dataset(ctx, condition, room)
            with (
                np.load(actual) as a,
                np.load(root / "reference/cnn/encodings/budget" / condition / actual.name) as b,
            ):
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def test_cnn_resume_restores_optimizer_and_shuffle(tmp_path, monkeypatch):
    import pytest
    import torch
    from sapp.experiments import cnn

    rng = np.random.default_rng(12)
    x = rng.normal(size=(20, 1, 6, 12)).astype(np.float32)
    y = rng.normal(size=(20, 2)).astype(np.float32)
    expected, _ = cnn.train(x, y, 37, 30, tmp_path / "uninterrupted.pt")
    save = cnn.save_torch

    def interrupt_after_checkpoint(path, value):
        save(path, value)
        if path.name == "resumed_checkpoint.pt":
            raise InterruptedError("Simulated process interruption after an atomic checkpoint")

    monkeypatch.setattr(cnn, "save_torch", interrupt_after_checkpoint)
    with pytest.raises(InterruptedError):
        cnn.train(x, y, 37, 30, tmp_path / "resumed.pt")
    monkeypatch.setattr(cnn, "save_torch", save)
    actual, history = cnn.train(x, y, 37, 30, tmp_path / "resumed.pt")
    assert len(history) == 30
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_verification_rejects_nonfinite_predictions(root, tmp_path):
    import shutil
    import pytest
    from sapp.experiments.verify import verify

    folder = tmp_path / "nist/predictions"
    shutil.copytree(root / "reference/nist/predictions", folder)
    path = folder / "L_LA.npz"
    with np.load(path) as data:
        values = {key: data[key] for key in data.files}
    values["sapp"][0] = np.nan
    np.savez_compressed(path, **values)
    with pytest.raises(RuntimeError):
        verify(Context(root, tmp_path), profile="primary")
    report = read_json(tmp_path / "verification.json")
    assert report["status"] == "FAIL"
    assert report["comparisons"]["nist/sapp"]["status"] == "FAIL"
    assert report["comparisons"]["nist/mode"]["status"] == "PASS"
