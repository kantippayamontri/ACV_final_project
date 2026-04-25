import importlib
import sys
from pathlib import Path
from unittest.mock import patch


def test_preprocess_data_main_calls_build_manifest(tmp_path):
    tsv = tmp_path / "val.csv"
    videos = tmp_path / "videos"
    out = tmp_path / "out"
    tsv.write_text("")
    videos.mkdir()

    with patch("preprocess.extract.build_manifest") as mock_build:
        mock_build.return_value = out / "manifest.jsonl"

        argv = [
            "preprocess_data.py",
            "--tsv", str(tsv),
            "--videos", str(videos),
            "--out", str(out),
            "--n-frames", "4",
        ]
        with patch.object(sys, "argv", argv):
            import preprocess_data
            importlib.reload(preprocess_data)

            preprocess_data.main()

    mock_build.assert_called_once_with(tsv, videos, out, n_frames=4)
