import sys
from pathlib import Path
from unittest.mock import patch


def test_train_cli_passes_args_to_train(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--max-steps", "10",
            "--output-dir", str(tmp_path / "out"),
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        mock_train.assert_called_once_with(
            manifest_path=manifest,
            val_manifest_path=None,
            num_epochs=2,
            max_steps=10,
            output_dir=str(tmp_path / "out"),
            resume_from_checkpoint=None,
            n_frames=8,
        )


def test_train_cli_epochs_arg(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--epochs", "3",
            "--output-dir", str(tmp_path / "out"),
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["num_epochs"] == 3
        assert call_kwargs["max_steps"] is None


def test_train_cli_auto_generates_output_dir_with_epochs(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--epochs", "2",
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["num_epochs"] == 2
        assert call_kwargs["max_steps"] is None
        assert "/runs/" in call_kwargs["output_dir"]
        assert call_kwargs["output_dir"].endswith("_ep2")


def test_train_cli_auto_generates_output_dir_with_max_steps(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--max-steps", "42",
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["max_steps"] == 42
        assert "/runs/" in call_kwargs["output_dir"]
        assert call_kwargs["output_dir"].endswith("_steps42")


def test_train_cli_passes_val_manifest_to_train(tmp_path):
    train_manifest = tmp_path / "train.jsonl"
    val_manifest = tmp_path / "val.jsonl"
    train_manifest.write_text("")
    val_manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(train_manifest),
            "--val-manifest", str(val_manifest),
            "--max-steps", "10",
            "--output-dir", str(tmp_path / "out"),
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["val_manifest_path"] == val_manifest


def test_train_cli_resume_arg(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--max-steps", "10",
            "--output-dir", str(tmp_path / "out"),
            "--resume", "asl_lora_output/checkpoint-870",
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["resume_from_checkpoint"] == "asl_lora_output/checkpoint-870"
        assert call_kwargs["output_dir"] == str(tmp_path / "out")
        assert call_kwargs["max_steps"] == 10


def test_train_cli_resume_auto_generates_output_dir(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--epochs", "2",
            "--resume", "runs/run_20260426_ep2/checkpoint-500",
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["resume_from_checkpoint"] == "runs/run_20260426_ep2/checkpoint-500"
        assert "/runs/" in call_kwargs["output_dir"]
        assert call_kwargs["output_dir"].endswith("_ep2")


def test_train_cli_resume_with_custom_output_dir(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--max-steps", "20",
            "--output-dir", str(tmp_path / "custom_resumed"),
            "--resume", "some/other/checkpoint-123",
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["resume_from_checkpoint"] == "some/other/checkpoint-123"
        assert call_kwargs["output_dir"] == str(tmp_path / "custom_resumed")
        assert call_kwargs["max_steps"] == 20


def test_train_cli_frames_arg(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with patch("train.train") as mock_train:
        argv = [
            "train.py",
            "--manifest-path", str(manifest),
            "--max-steps", "10",
            "--output-dir", str(tmp_path / "out"),
            "--frames", "16",
        ]
        with patch.object(sys, "argv", argv):
            from train import cli
            cli()

        mock_train.assert_called_once_with(
            manifest_path=manifest,
            val_manifest_path=None,
            num_epochs=2,
            max_steps=10,
            output_dir=str(tmp_path / "out"),
            resume_from_checkpoint=None,
            n_frames=16,
        )
