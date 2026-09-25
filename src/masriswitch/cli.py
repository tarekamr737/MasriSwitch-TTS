"""Stable project command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from masriswitch.config import Paths, validate_configs

LOG = logging.getLogger("masriswitch")
EXPENSIVE = {"audit-data", "prepare-data", "bootstrap", "train-pilot", "train", "eval"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masriswitch")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "validate-config",
        "lock-sources",
        "bootstrap",
        "audit-data",
        "prepare-data",
        "benchmark",
        "train-pilot",
        "train",
        "eval",
        "demo",
        "api",
        "release-check",
        "release",
        "normalize",
    ):
        command = sub.add_parser(name)
        if name in EXPENSIVE:
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--max-samples", type=int)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--seed", type=int, default=42)
        if name == "normalize":
            command.add_argument("text")
        if name == "prepare-data":
            command.add_argument("--finalize-only", action="store_true")
        if name == "benchmark":
            command.add_argument("--seed", type=int, default=42)
        if name == "eval":
            command.add_argument("--rows", type=Path)
            command.add_argument("--stage", choices=("E0", "E1"), default="E0")
            command.add_argument("--subset", choices=("all", "validation"), default="all")
            command.add_argument("--checkpoint-sha256")
        if name in {"train-pilot", "train"}:
            command.add_argument("--archive-url")
            command.add_argument(
                "--max-updates", type=int, default=500 if name == "train-pilot" else 1000
            )
            command.add_argument("--resume-checkpoint-url")
            command.add_argument("--resume-checkpoint-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    paths = Paths()
    try:
        if args.command == "validate-config":
            validate_configs(paths.root)
            LOG.info("Configs valid")
        elif args.command == "normalize":
            from masriswitch.text.normalize import normalize_text

            normalization = normalize_text(args.text)
            print(
                json.dumps(
                    {
                        "text": normalization.normalized_text,
                        "entities": [entity.__dict__ for entity in normalization.entities],
                    },
                    ensure_ascii=False,
                )
            )
        elif args.command == "benchmark":
            from masriswitch.eval.benchmark import write_benchmark

            print(json.dumps(write_benchmark(paths.artifacts, args.seed)))
        elif args.command == "release-check":
            from masriswitch.release.gate import check_release

            release_status = check_release(paths)
            print(json.dumps(release_status))
            return 0 if release_status["weight_publication_allowed"] else 2
        elif args.command == "release":
            from masriswitch.release.package import package_release

            print(json.dumps(package_release(paths)))
        elif args.command == "audit-data":
            from masriswitch.data.audit import audit_source
            from masriswitch.data.registry import load_sources

            source = load_sources()["d1"]
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "source": source.repo,
                            "revision": source.revision,
                            "max_samples": args.max_samples,
                            "would_download": True,
                        }
                    )
                )
            else:
                print(json.dumps(audit_source(source, paths, args.max_samples)))
        elif args.command == "lock-sources":
            from masriswitch.data.lock import lock_sources

            locked = lock_sources(paths)
            print(
                json.dumps(
                    {
                        "revisions": {
                            key: value["revision"] for key, value in locked["sources"].items()
                        },
                        "model_downloaded": locked["verified"],
                    }
                )
            )
        elif args.command == "bootstrap":
            from masriswitch.train.bootstrap_silma import bootstrap_model

            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "files": ["silma/model.pt", "vocos/pytorch_model.bin"],
                            "would_download": True,
                        }
                    )
                )
            else:
                print(json.dumps(bootstrap_model(paths)))
        elif args.command == "prepare-data":
            from masriswitch.data.prepare import finalize_existing, prepare_data
            from masriswitch.data.registry import load_sources

            source = load_sources()["d1"]
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "source": source.repo,
                            "revision": source.revision,
                            "max_samples": args.max_samples,
                            "resume": args.resume,
                        }
                    )
                )
            elif args.finalize_only:
                print(json.dumps(finalize_existing(source, paths)))
            else:
                print(
                    json.dumps(
                        prepare_data(
                            source,
                            paths,
                            max_samples=args.max_samples,
                            resume=args.resume,
                            seed=args.seed,
                        )
                    )
                )
        elif args.command in {"train-pilot", "train"}:
            pilot = args.command == "train-pilot"
            archive_manifest = paths.artifacts / (
                "kaggle_pilot_archive.json" if pilot else "kaggle_full_archive.json"
            )
            archive = json.loads(archive_manifest.read_text(encoding="utf-8"))
            archive_url = args.archive_url or os.environ.get("MASRISWITCH_ARCHIVE_URL")
            if not archive_url and not args.dry_run:
                raise ValueError("Set MASRISWITCH_ARCHIVE_URL to a private Kaggle archive URL")
            script = (
                paths.root
                / "scripts"
                / ("kaggle_train_pilot.py" if pilot else "kaggle_train_e1.py")
            )
            command = [
                sys.executable,
                str(script),
                "--root",
                str(paths.root),
                "--archive-url",
                archive_url or "https://www.kaggleusercontent.com/dry-run",
                "--archive-sha256",
                archive["archive_sha256"],
                "--arrow-sha256",
                archive["train_arrow_sha256"] if not pilot else archive["pilot_arrow_sha256"],
                "--archive-manifest",
                str(archive_manifest),
                "--patch-manifest",
                str(paths.artifacts / "train_patch.json"),
                "--probe-result",
                str(paths.artifacts / "train_probe.json"),
                "--max-updates",
                str(args.max_updates),
                "--seed",
                str(args.seed),
            ]
            if not pilot:
                command += [
                    "--pilot-result",
                    str(paths.artifacts / "pilot_500_result.json"),
                    "--full-probe-result",
                    str(paths.artifacts / "full_probe_result.json"),
                ]
            if args.max_samples is not None:
                command += ["--max-samples", str(args.max_samples)]
            if args.resume:
                command.append("--resume")
            resume_url = args.resume_checkpoint_url or os.environ.get(
                "MASRISWITCH_RESUME_CHECKPOINT_URL"
            )
            resume_sha = args.resume_checkpoint_sha256 or os.environ.get(
                "MASRISWITCH_RESUME_CHECKPOINT_SHA256"
            )
            if resume_url or resume_sha:
                if not resume_url or not resume_sha or not args.resume:
                    raise ValueError("Resume URL and SHA256 require --resume together")
                command += [
                    "--resume-checkpoint-url",
                    resume_url,
                    "--resume-checkpoint-sha256",
                    resume_sha,
                ]
            if args.dry_run:
                command.append("--dry-run")
            return subprocess.run(command, check=False).returncode
        elif args.command == "eval":
            from masriswitch.eval.aggregate import aggregate_e0, aggregate_evaluation

            default_rows = (
                "e0_eval_rows.jsonl" if args.stage == "E0" else f"e1_{args.subset}_eval_rows.jsonl"
            )
            rows = args.rows or paths.artifacts / default_rows
            count = args.max_samples or (189 if args.subset == "validation" else 489)
            if args.stage == "E0" and (args.subset != "all" or args.checkpoint_sha256):
                raise ValueError("E0 uses the full baseline plan and pinned SILMA model")
            if args.stage == "E1" and not args.checkpoint_sha256:
                raise ValueError("E1 requires --checkpoint-sha256")
            if args.dry_run:
                print(json.dumps({"stage": args.stage, "rows": str(rows), "planned": count}))
            else:
                if args.stage == "E0":
                    result = aggregate_e0(paths, rows, max_samples=count)
                else:
                    result = aggregate_evaluation(
                        paths,
                        rows,
                        stage="E1",
                        model_sha256=args.checkpoint_sha256,
                        subset=args.subset,
                        max_samples=count,
                    )
                if args.stage == "E0" and result["complete"]:
                    from masriswitch.eval.report import write_e0_report

                    write_e0_report(paths, result)
                print(json.dumps(result, ensure_ascii=False))
        elif args.command in {"api", "demo"}:
            from masriswitch.infer.engine import F5Engine, approved_engine_files

            try:
                engine = F5Engine(approved_engine_files(paths))
            except (FileNotFoundError, PermissionError, ValueError) as exc:
                LOG.warning("Synthesis unavailable: %s", exc)
                engine = None
            if args.command == "api":
                import uvicorn

                from masriswitch.api.app import create_app

                uvicorn.run(create_app(engine), host="127.0.0.1", port=8000)
            else:
                from masriswitch.infer.demo import create_demo

                create_demo(engine).launch(server_name="127.0.0.1", server_port=7860)
        else:
            LOG.error("%s is not implemented yet; no expensive work was started", args.command)
            return 2
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        LOG.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
