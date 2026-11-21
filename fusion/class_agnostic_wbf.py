"""Class-agnostic weighted box fusion and final class-agnostic NMS."""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from submission_fusion import iou
from yolo960_ab import MAX_PER_IMAGE


IOU = .55


@dataclass
class Cluster:
    box: tuple
    members: dict

    def add(self, source, obj, weights):
        self.members[source] = obj
        votes = {s: weights[s] * d["score"] for s, d in self.members.items()}
        self.box = tuple(sum(d["box"][k] * votes[s] for s, d in self.members.items()) /
                         sum(votes.values()) for k in range(4))


def fuse_frame(per_model, weights, *, suppress=True, primary_index=0):
    if len(per_model) != len(weights) or any(not isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Expected one finite positive weight per model")
    ordered = sorted(((s, d) for s, ds in enumerate(per_model) for d in ds if d["score"] > 0),
                     key=lambda item: -weights[item[0]] * item[1]["score"])
    clusters = []
    for source, obj in ordered:
        best_iou, index = max(((iou(c.box, obj["box"]), i) for i, c in enumerate(clusters)),
                              key=lambda pair: pair[0], default=(0., -1))
        if best_iou < IOU:
            clusters.append(Cluster(tuple(obj["box"]), {source: obj}))
        elif source not in clusters[index].members:
            clusters[index].add(source, obj, weights)
        # A model's lower-confidence alternatives cannot vote for the same cluster.

    ranked = []
    for c in clusters:
        by_form = defaultdict(float)
        for s, d in c.members.items():
            by_form[d["form"]] += weights[s] * d["score"]
        form = max(by_form, key=by_form.get)
        supporters = {s: d for s, d in c.members.items() if d["form"] == form}
        chosen = supporters[primary_index] if primary_index in supporters else next(iter(supporters.values()))
        box = tuple(float(f"{v:.1f}") for v in c.box)
        if box[0] < box[2] and box[1] < box[3]:
            # Keep A's max-confidence convention for object ranking. The class
            # comes from the weighted vote, and attributes from a winning member.
            score = float(f"{max(d['score'] for d in c.members.values()):.6f}")
            ranked.append((box, score, chosen["tail"]))

    kept = []
    for row in sorted(ranked, key=lambda r: -r[1]):
        if suppress and any(iou(row[0], k[0]) >= IOU for k in kept):
            continue
        kept.append(row)
        if len(kept) == MAX_PER_IMAGE:
            break
    return ";".join(f"{' '.join(f'{v:.1f}' for v in box)} {score:.6f} {tail}"
                    for box, score, tail in kept) or "none"
