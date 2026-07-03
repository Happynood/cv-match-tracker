"""``matchtracker`` console entry point: ``run`` / ``eval`` / ``calibrate``.

Config loading uses Hydra's compose API directly (rather than the
``@hydra.main`` decorator) so the CLI can mirror a plain
``--config path --output dir`` invocation style.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    resolved = Path(config_path).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(resolved.parent)):
        return compose(config_name=resolved.stem, overrides=overrides or [])


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.override)
    if args.video:
        cfg.video.path = args.video
    if args.output:
        cfg.output.dir = args.output

    from matchtracker.pipeline import run_pipeline

    result = run_pipeline(cfg, output_dir=cfg.output.dir)
    print(f"players.parquet: {result.players_parquet_path}")
    print(f"stats.json:      {result.stats_json_path}")
    if result.report_html_path:
        print(f"report:          {result.report_html_path}")
    if result.annotated_video_path:
        print(f"annotated.mp4:   {result.annotated_video_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.override)

    from matchtracker.calibrate import CalibrationConfig, calibrate_static, pitch_config

    calib_cfg = CalibrationConfig.from_mapping(
        OmegaConf.to_container(cfg.calibration, resolve=True)
    )
    if args.correspondences:
        calib_cfg = CalibrationConfig(
            static=calib_cfg.static,
            method="manual",
            refit_on_bump=calib_cfg.refit_on_bump,
            min_keypoints=calib_cfg.min_keypoints,
            reproj_error_max_m=calib_cfg.reproj_error_max_m,
            correspondences_path=args.correspondences,
        )

    pitch = pitch_config(cfg.pitch.length_m, cfg.pitch.width_m)
    result = calibrate_static(calib_cfg, pitch)

    print(f"n_points:              {result.n_points}")
    print(f"reprojection_error_m:  {result.reprojection_error_m:.3f}")
    print(f"valid (<= {calib_cfg.reproj_error_max_m}m): {result.valid}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "homography": result.transformer.m.tolist(),
                    "reprojection_error_m": result.reprojection_error_m,
                    "n_points": result.n_points,
                    "valid": result.valid,
                },
                indent=2,
            )
        )
        print(f"Saved homography to {out_path}")

    return 0 if result.valid else 1


def cmd_eval(args: argparse.Namespace) -> int:
    import pandas as pd

    try:
        import motmetrics as mm
    except ImportError:
        print(
            "motmetrics is not installed. Install the 'eval' extra: uv sync --extra eval",
            file=sys.stderr,
        )
        return 1

    pred_df = pd.read_parquet(args.predictions)
    gt_df = pd.read_csv(
        args.ground_truth,
        header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis"],
    )

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_df["frame"].unique()):
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_ids = gt_frame["track_id"].tolist()
        pred_ids = pred_frame["track_id"].tolist()
        gt_boxes = gt_frame[["x", "y", "w", "h"]].to_numpy()  # type: ignore[attr-defined]
        pred_xy = pred_frame[["u", "v"]]
        pred_boxes = (
            pred_xy.assign(w=1.0, h=1.0)[["u", "v", "w", "h"]].to_numpy()  # type: ignore[attr-defined]
            if len(pred_frame)
            else pred_xy.to_numpy()  # type: ignore[attr-defined]
        )

        distances = (
            mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5) if len(pred_boxes) else []
        )
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name="matchtracker")
    print(summary)

    if args.output:
        Path(args.output).write_text(summary.to_json(indent=2))  # type: ignore[attr-defined]
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchtracker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full tracking + stats pipeline on a video.")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--video", default=None, help="Override video.path from the config.")
    p_run.add_argument("--override", nargs="*", default=[], help="Hydra-style key=value overrides.")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="Compute MOTA/IDF1 against a MOT-format ground truth.")
    p_eval.add_argument("--predictions", required=True, help="players.parquet path.")
    p_eval.add_argument(
        "--ground-truth", required=True, help="MOTChallenge-format ground truth CSV."
    )
    p_eval.add_argument("--output", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_calib = sub.add_parser("calibrate", help="Fit and validate the static homography.")
    p_calib.add_argument("--config", required=True)
    p_calib.add_argument("--correspondences", default=None, help="Manual correspondences JSON.")
    p_calib.add_argument("--output", default=None, help="Where to save the fitted homography JSON.")
    p_calib.add_argument("--override", nargs="*", default=[])
    p_calib.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
