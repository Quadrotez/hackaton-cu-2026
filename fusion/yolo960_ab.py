"""Build separate YOLO960 A/B submissions on top of the team's CSV fusion."""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from attribute_postprocess import DEFAULT_MODEL, MODES, AttributePolicy
from submission_fusion import iou, load


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("yolo960_ab.json")
MAX_PER_IMAGE = 30
HEADER = ("image_id", "prediction")


@dataclass(frozen=True)
class Model:
    group: str
    fold: int
    checkpoint: Path
    imgsz: int
    weight: float

    @property
    def tag(self):
        return f"yolo11{self.group}_fold_{self.fold:02d}"


def read_config(path, include_nano):
    config = json.loads(path.read_text(encoding="utf-8"))
    models = []
    for group in ("m", "l", "n") if include_nano else ("m", "l"):
        spec = config["groups"][group]
        imgsz = spec["imgsz"]
        weight = spec["weight"]
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{group}: expected a finite positive weight")
        if imgsz != (640 if group == "n" else 960):
            raise ValueError(f"{group}: unexpected inference resolution {imgsz}")
        for fold in range(1, 6):
            checkpoint = (ROOT / spec["checkpoint"].format(fold=fold)).resolve()
            models.append(Model(group, fold, checkpoint, imgsz, float(weight)))
    if len({m.checkpoint for m in models}) != len(models):
        raise ValueError("Expected a different checkpoint for every fold")
    return (ROOT / config["primary_csv"]).resolve(), (ROOT / config["heads"]).resolve(), models


def fuse_frame(per_model, weights, support, primary_index=0):
    # Preserve the original: descending confidence, first same-form cluster
    # with IoU >= .5 to its initial (fixed) box, at most one vote per source.
    ordered = sorted(
        ((si, d) for si, detections in enumerate(per_model) for d in detections),
        key=lambda pair: -pair[1]["score"],
    )
    clusters = []
    for si, d in ordered:
        if si in support:
            continue
        c = next((c for c in clusters if c["form"] == d["form"] and iou(c["box"], d["box"]) >= .5), None)
        if c is None:
            clusters.append({"form": d["form"], "box": d["box"], "members": {si: d}})
        elif si not in c["members"]:
            c["members"][si] = d
    for si, d in ordered:
        if si not in support:
            continue
        c = next((c for c in clusters if c["form"] == d["form"] and iou(c["box"], d["box"]) >= .5), None)
        if c is not None and si not in c["members"]:
            c["members"][si] = d

    parts = []
    for c in clusters:
        members = c["members"]
        votes = [weights[si] * d["score"] for si, d in members.items()]
        total = sum(votes)
        if not total:
            continue
        box = [sum(d["box"][k] * w for d, w in zip(members.values(), votes)) / total for k in range(4)]
        coords = [f"{v:.1f}" for v in box]
        x1, y1, x2, y2 = map(float, coords)
        if not (x1 < x2 and y1 < y2):
            continue
        # Nano may move coordinates, but cannot set confidence or attributes.
        score = max(d["score"] for si, d in members.items() if si not in support)
        primary = members[primary_index] if primary_index in members else next(iter(members.values()))
        parts.append(f"{' '.join(coords)} {score:.6f} {primary['tail']}")
        if len(parts) == MAX_PER_IMAGE:
            break
    return ";".join(parts) or "none"


def fused_rows(sources, weights, support=()):
    support = frozenset(support)
    if (len(sources) < 2 or len(sources) != len(weights) or 0 in support
            or not support <= set(range(len(sources)))):
        raise ValueError("Invalid source weights or support indices")
    if any(src.keys() != sources[0].keys() for src in sources):
        raise ValueError("Input CSVs must have identical frame coverage")
    for fid in sorted(sources[0]):
        yield fid, fuse_frame([src[fid] for src in sources], weights, support)


def write_csv(out, rows, verify=None):
    with NamedTemporaryFile(dir=out.parent, mode="w", newline="", encoding="utf-8", delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(rows)
            f.close()
            if verify is not None:
                verify(temporary)
            out.hardlink_to(temporary)
        finally:
            temporary.unlink(missing_ok=True)


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checkpoint_hashes(models, primary_checkpoint, expected_primary_sha, *, check_archives=False):
    paths = {"yolo11s_primary": primary_checkpoint, **{m.tag: m.checkpoint for m in models}}
    result = {}
    for tag, path in paths.items():
        if check_archives:
            # Torch checkpoints in this config are ZIP archives; detect partial or corrupt downloads.
            with ZipFile(path) as archive:
                if not any(n.endswith("/data.pkl") for n in archive.namelist()) or archive.testzip() is not None:
                    raise ValueError(f"Invalid checkpoint archive: {path}")
        result[tag] = file_sha(path)
    if expected_primary_sha != result["yolo11s_primary"]:
        raise ValueError(f"Primary checkpoint has unexpected SHA-256: {primary_checkpoint}")
    if len(set(result.values())) != len(paths):
        raise ValueError("Two checkpoints have identical contents; refusing duplicate votes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--variant", choices=("A", "B", "both"), default="both")
    parser.add_argument("--check", action="store_true", help="Check files, checkpoint CRCs and hashes without inference")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "submission_yolo960_ab")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, mps or 0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--postprocess", choices=("best", *MODES), default="off",
                        help="Also write a *_post.csv using the fitted volume/take policy")
    parser.add_argument("--postprocess-model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")

    primary, heads, models = read_config(args.config, args.variant != "A")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    primary_checkpoint = (ROOT / config["primary_checkpoint"]).resolve()
    required = [primary, primary_checkpoint, heads, *(ROOT / "work" / file for file in
                ("astra_dino.safetensors", "astra_dino_head.npz", "astra_clip.bin")),
                *(m.checkpoint for m in models)]
    if args.postprocess != "off":
        required.append(args.postprocess_model)
    missing = [p for p in required if not p.is_file()]
    print(f"{len(required) - len(missing)}/{len(required)} required local files present.")
    for p in missing:
        print(f"MISSING {p}")
    if missing:
        return 1
    policy = AttributePolicy.load(args.postprocess_model) if args.postprocess != "off" else None
    sha = checkpoint_hashes(models, primary_checkpoint, config["primary_checkpoint_sha256"],
                            check_archives=args.check)
    if args.check:
        for name, digest in sha.items():
            print(f"{digest}  {name}")
        print(f"OK: {len(sha)} distinct checkpoints.")
        return 0

    from export_joint04 import Joint04Exporter

    exporter = Joint04Exporter(ROOT, heads, args.device, args.batch)
    exporter.validate(primary)
    rgb_sha = exporter.frame_key(exporter.frames)
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.out / "per_model"
    raw.mkdir()
    sources = [load(primary)]
    for m in models:
        path = raw / f"{m.tag}.csv"
        print(f"Exporting {m.tag} at {m.imgsz}: {m.checkpoint}", flush=True)
        write_csv(path, exporter.rows(m, sha[m.tag], rgb_sha), exporter.validate)
        sources.append(load(path))

    weights = [1., *(m.weight for m in models)]
    nano = {idx for idx, m in enumerate(models, 1) if m.group == "n"}
    for variant in ("A", "B") if args.variant == "both" else (args.variant,):
        count = 11 if variant == "A" else len(sources)
        frames = list(fused_rows(sources[:count], weights[:count], nano if variant == "B" else ()))
        out = args.out / f"submission_{variant}.csv"
        write_csv(out, frames, exporter.validate)
        print(f"Saved {out}")
        if policy is not None:
            adjusted = args.out / f"submission_{variant}_post.csv"
            write_csv(adjusted, policy.rows(args.postprocess, frames), exporter.validate)
            print(f"Saved {adjusted} ({policy.resolve(args.postprocess)})")
    report = {"primary_csv": str(primary), "primary_sha256": file_sha(primary), "heads": str(heads),
              "primary_checkpoint": str(primary_checkpoint), "primary_checkpoint_sha256": sha["yolo11s_primary"],
              "heads_sha256": file_sha(heads), "variant": args.variant, "device": args.device,
              "models": [{**vars(m), "checkpoint": str(m.checkpoint), "sha256": sha[m.tag]} for m in models]}
    if policy is not None:
        report["postprocess"] = {"requested": args.postprocess, "resolved": policy.resolve(args.postprocess),
                                 "model": str(args.postprocess_model),
                                 "sha256": file_sha(args.postprocess_model)}
    with (args.out / "inputs.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
