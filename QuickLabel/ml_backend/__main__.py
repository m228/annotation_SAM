"""CLI entry point for the bundled ml_backend.

Run as ``python -m ml_backend <command>`` — QuickLabel's web server launches
these as subprocesses (see backend/sam_runtime.py and backend/train_runtime.py):

    sam         — SAM 2 / SAM 3 segmentation service (JSON-lines over stdin/stdout)
    train       — RF-DETR training job (config JSON; progress on stdout)
    train-yolo  — Ultralytics YOLO training job (config JSON; progress on stdout)
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml_backend", description="QuickLabel ML backend")
    sub = parser.add_subparsers(dest="command")

    sam = sub.add_parser("sam", help="Start the SAM segmentation service")
    sam.add_argument("--model", default="", help="Optional model to pre-load")

    train = sub.add_parser("train", help="Run an RF-DETR training job")
    train.add_argument("--config", required=True, help="Path to training config JSON")

    train_yolo = sub.add_parser("train-yolo", help="Run an Ultralytics YOLO training job")
    train_yolo.add_argument("--config", required=True, help="Path to training config JSON")

    predict = sub.add_parser("predict", help="Run inference (test) with a trained model")
    predict.add_argument("--config", required=True, help="Path to predict config JSON")

    args = parser.parse_args()

    if args.command == "sam":
        from .sam_service import run_sam_service
        run_sam_service(initial_model=args.model)
    elif args.command == "train":
        from .training_service import run_training
        run_training(config_path=args.config)
    elif args.command == "train-yolo":
        from .yolo_train_service import run_yolo_training
        run_yolo_training(config_path=args.config)
    elif args.command == "predict":
        from .predict_service import run_predict
        run_predict(config_path=args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
