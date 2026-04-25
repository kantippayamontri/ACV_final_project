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
            max_steps=10,
            output_dir=str(tmp_path / "out"),
        )
