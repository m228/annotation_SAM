"""CLI entry point for the bundled SAM service.

Run as ``python -m ml_backend sam`` — QuickLabel's web server launches this as a
subprocess and talks JSON-lines over stdin/stdout (see backend/sam_runtime.py).
Only the SAM service is bundled in QuickLabel; training/classification commands
from the original VisoLabel backend are intentionally omitted.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml_backend", description="QuickLabel SAM service")
    sub = parser.add_subparsers(dest="command")
    sam = sub.add_parser("sam", help="Start the SAM segmentation service")
    sam.add_argument("--model", default="", help="Optional model to pre-load")
    args = parser.parse_args()

    if args.command == "sam":
        from .sam_service import run_sam_service
        run_sam_service(initial_model=args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
