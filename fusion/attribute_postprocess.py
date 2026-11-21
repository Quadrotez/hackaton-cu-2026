"""Optional CSV-only volume/take postprocessor for the Astramis ensemble."""

import argparse
import bisect
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


SCHEMA_VERSION = 2
HEAVY_G = 500.0
MODES = ("off", "density", "density-small", "rescue-small", "bulk", "density+bulk", "density-small+bulk")
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "work" / "astramis_attr_v2" / "policy.json"


@dataclass(frozen=True)
class DensityPrior:
    # One entry per physical object, regardless of the number of its views.
    entries: tuple
    temperature: float = .35

    @classmethod
    def fit(cls, rows):
        groups = {}
        for object_id, form, material, volume, mass in rows:
            if not all(math.isfinite(x) and x > 0 for x in (volume, mass)):
                raise ValueError(f"{object_id}: density requires positive volume and mass")
            group = groups.setdefault(object_id, (form, material, []))
            if group[:2] != (form, material):
                raise ValueError(f"{object_id}: inconsistent form/material")
            group[2].append(math.log(mass) - math.log(volume))
        return cls(tuple((f, m, median(log_rho)) for _, (f, m, log_rho) in sorted(groups.items())))

    def __post_init__(self):
        if not self.entries or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid density prior")
        if any(not math.isfinite(rho) for _, _, rho in self.entries):
            raise ValueError("Invalid log density")

    def light_probability(self, form, material, volume):
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Positive finite volume required")
        log_limit = math.log(HEAVY_G) - math.log(volume)
        total = weighted_sum = 0.
        for f, m, rho in self.entries:
            weight = 1. + 2. * (f == form) + 4. * (m == material)
            # Smooth the empirical CDF in log density to avoid steps from a few objects.
            p = .5 + .5 * math.tanh((log_limit - rho) / (2. * self.temperature))
            total += weight
            weighted_sum += weight * p
        return weighted_sum / total


@dataclass(frozen=True)
class TakeRule:
    cut: float
    small_ml: float = 0.

    def __post_init__(self):
        if not 0 <= self.cut <= 1 or not math.isfinite(self.small_ml) or self.small_ml < 0:
            raise ValueError("Invalid take threshold")

    def predict(self, probability, volume):
        return int(probability >= self.cut or 0 < volume <= self.small_ml)


@dataclass(frozen=True)
class BulkCalibration:
    log_x: tuple
    log_y: tuple
    strength: float = .5

    def __post_init__(self):
        if (not 0 <= self.strength <= 1 or len(self.log_x) != len(self.log_y)
                or any(not math.isfinite(x) for x in (*self.log_x, *self.log_y))
                or any(a >= b for a, b in zip(self.log_x, self.log_x[1:]))
                or any(a > b for a, b in zip(self.log_y, self.log_y[1:]))):
            raise ValueError("Invalid monotone bulk calibration")

    def apply(self, volume):
        if not self.log_x or volume <= 0:
            return volume
        x = math.log(volume)
        j = bisect.bisect_right(self.log_x, x)
        if j == 0 or j == len(self.log_x):
            # Keep a constant multiplier past the training range, not a flat volume.
            k = 0 if j == 0 else -1
            offset = self.log_y[k] - self.log_x[k]
        else:
            x0, x1 = self.log_x[j - 1:j + 1]
            y0, y1 = self.log_y[j - 1:j + 1]
            offset = y0 + (y1 - y0) * (x - x0) / (x1 - x0) - x
        limit = math.log(2.)
        return volume * math.exp(max(-limit, min(limit, self.strength * offset)))


@dataclass(frozen=True)
class AttributePolicy:
    prior: DensityPrior
    density: TakeRule
    density_small: TakeRule
    rescue_small: TakeRule
    bulk: BulkCalibration
    recommended: str = "off"

    def __post_init__(self):
        if self.recommended not in MODES or self.density.small_ml != 0 or self.density_small.small_ml <= 0:
            raise ValueError("Invalid attribute policy")

    def resolve(self, mode):
        if mode == "best":
            return self.recommended
        if mode not in MODES:
            raise ValueError(f"Unknown attribute mode: {mode}")
        return mode

    def apply(self, mode, form, material, volume, mass, take):
        mode = self.resolve(mode)
        if mode == "off" or volume <= 0:
            return volume, mass, take
        if mode.startswith("density"):
            rule = self.density_small if mode.startswith("density-small") else self.density
            take = rule.predict(self.prior.light_probability(form, material, volume), volume)
        if mode == "rescue-small" and take == 0 and volume <= self.rescue_small.small_ml:
            take = int(self.prior.light_probability(form, material, volume) >= self.rescue_small.cut)
        if mode.endswith("bulk") and form == "bulk":
            # Keep take's input at its calibrated scale; mass follows the exported volume.
            corrected = self.bulk.apply(volume)
            mass *= corrected / volume
            volume = corrected
        return volume, mass, take

    def to_dict(self):
        return {"version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, spec):
        if spec["version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported policy version: {spec['version']}")
        return cls(DensityPrior(**spec["prior"]), TakeRule(**spec["density"]),
                   TakeRule(**spec["density_small"]), TakeRule(**spec["rescue_small"]),
                   BulkCalibration(**spec["bulk"]),
                   spec["recommended"])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def prediction(self, mode, prediction):
        if prediction == "none" or self.resolve(mode) == "off":
            return prediction
        parts = []
        for text in prediction.split(";"):
            fields = text.split()
            if len(fields) != 10:
                raise ValueError("Expected a 10-field prediction")
            volume, mass = map(float, fields[7:9])
            if not all(math.isfinite(x) and x >= 0 for x in (volume, mass)):
                raise ValueError("Invalid volume or mass")
            v, m, t = self.apply(mode, fields[5], fields[6], volume, mass, int(fields[9]))
            if v != volume:
                fields[7] = f"{v:.12g}"
            if m != mass:
                fields[8] = f"{m:.12g}"
            fields[9] = str(t)
            parts.append(" ".join(fields))
        return ";".join(parts)

    def rows(self, mode, rows):
        for image_id, prediction in rows:
            yield image_id, self.prediction(mode, prediction)


def main():
    from yolo960_ab import HEADER, write_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("best", *MODES), default="best")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = AttributePolicy.load(args.model)
    with args.input.open(newline="", encoding="utf-8") as source:
        data = csv.DictReader(source)
        if tuple(data.fieldnames or ()) != HEADER:
            raise ValueError(f"{args.input}: expected {HEADER}")
        write_csv(args.out, policy.rows(args.mode, ((r["image_id"], r["prediction"]) for r in data)))
    print(f"Saved {args.out} ({policy.resolve(args.mode)})")


if __name__ == "__main__":
    main()
