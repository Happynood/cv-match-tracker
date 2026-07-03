"""CPU end-to-end smoke test: runs the full pipeline on a tiny fixture clip.

Uses the COCO-pretrained RF-DETR fallback (no football fine-tune is bundled
with this repo — see detect.py) and the manual-correspondences calibration
fixture. This only checks the pipeline runs end-to-end and produces
schema-valid outputs; it makes no accuracy claims.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CLIP = REPO_ROOT / "data" / "smoke" / "smoke_clip.mp4"
SMOKE_KEYPOINTS = REPO_ROOT / "data" / "smoke" / "manual_keypoints.json"
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.mark.smoke
def test_pipeline_runs_end_to_end_on_smoke_clip(tmp_path):
    from matchtracker.pipeline import run_pipeline

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")

    OmegaConf.set_struct(cfg, False)
    cfg.video.path = str(SMOKE_CLIP)
    cfg.video.target_fps = 5
    cfg.team.sample_fps = 1
    cfg.team.n_clusters = 2
    cfg.calibration.method = "manual"
    cfg.calibration.correspondences_path = str(SMOKE_KEYPOINTS)
    cfg.calibration.min_keypoints = 4
    cfg.tracker.min_track_len = 1
    cfg.ball.enabled = False
    cfg.output.dir = str(tmp_path)
    cfg.output.save_annotated_video = False

    result = run_pipeline(cfg, output_dir=tmp_path)

    assert result.players_parquet_path.exists()
    assert result.stats_json_path.exists()

    stats_json = result.stats_json_path.read_text()
    assert '"config_sha256"' in stats_json

    import pandas as pd

    df = pd.read_parquet(result.players_parquet_path)
    expected_columns = {
        "frame",
        "t_s",
        "track_id",
        "team",
        "cls",
        "u",
        "v",
        "x_m",
        "y_m",
        "speed_ms",
        "calib_valid",
    }
    assert expected_columns.issubset(set(df.columns))
