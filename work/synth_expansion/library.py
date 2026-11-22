"""
Library module: extracts and organizes source objects and real empty backgrounds.
"""
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np

from config import (
    DATA_DIR,
    SYNTH_JSON_PATH,
    REAL_JSON_PATH,
    DENSITIES,
)


class ObjectInstance:
    def __init__(self, obj_id, form, material, volume_ml, mass_g, take, crop_rgb, crop_mask, source="synth", donor_frame=""):
        self.obj_id = obj_id
        self.form = form
        self.material = material
        self.volume_ml = float(volume_ml)
        self.mass_g = float(mass_g)
        self.take = int(take)
        self.crop_rgb = crop_rgb          # (H, W, 3) uint8 BGR
        self.crop_mask = crop_mask        # (H, W) uint8 {0, 255}
        self.source = source
        self.donor_frame = donor_frame
        # Actual density of this physical object
        self.object_density = self.mass_g / self.volume_ml if self.volume_ml > 0 else DENSITIES[(form, material)]["density_g_per_ml"]


class DatasetLibrary:
    def __init__(self):
        self.objects_by_class = defaultdict(list)  # (form, material) -> list of ObjectInstance
        self.objects_by_id = defaultdict(list)     # object_id -> list of ObjectInstance
        self.empty_backgrounds = []                # list of Path to background images
        self.background_metadata = {}              # Path -> metadata dict
        self._load_empty_backgrounds()
        self._load_synthetic_objects()

    def _load_empty_backgrounds(self):
        """Loads all 49 empty conveyor backgrounds from real_train."""
        with open(REAL_JSON_PATH, "r", encoding="utf-8") as f:
            real_data = json.load(f)

        pos_ids = {a["image_id"] for a in real_data["annotations"]}
        empty_images = [img for img in real_data["images"] if img["id"] not in pos_ids]

        for img in empty_images:
            img_path = DATA_DIR / img["file_name"]
            if img_path.exists():
                self.empty_backgrounds.append(img_path)
                meta_path = img_path.parent / "meta.json"
                if meta_path.exists():
                    try:
                        self.background_metadata[str(img_path)] = json.loads(meta_path.read_text())
                    except Exception:
                        self.background_metadata[str(img_path)] = {}

        print(f"[DatasetLibrary] Loaded {len(self.empty_backgrounds)} real empty background frames.")

    def _load_synthetic_objects(self):
        """Extracts clean object cutouts and masks from dataset/synth/train/."""
        with open(SYNTH_JSON_PATH, "r", encoding="utf-8") as f:
            synth_data = json.load(f)

        img_map = {im["id"]: im for im in synth_data["images"]}

        loaded_count = 0
        for ann in synth_data["annotations"]:
            im_info = img_map[ann["image_id"]]
            img_dir = DATA_DIR / Path(im_info["file_name"]).parent
            color_path = img_dir / "color.png"
            mask_path = img_dir / "mask.png"

            if not (color_path.exists() and mask_path.exists()):
                continue

            inst = ann.get("instance", 1)
            attr = ann.get("attributes", {})
            form = attr.get("form")
            mat = attr.get("material")
            obj_id = attr.get("object_id")
            vol = attr.get("volume_ml")
            mass = attr.get("mass_g")
            take = attr.get("take")

            if not (form and mat and obj_id and vol is not None and mass is not None):
                continue

            # Read color and mask
            color = cv2.imread(str(color_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

            if color is None or mask is None:
                continue

            bin_mask = (mask == inst).astype(np.uint8) * 255
            nonzero_y, nonzero_x = np.where(bin_mask > 0)
            if len(nonzero_y) < 50:  # skip degenerate masks
                continue

            ymin, ymax = int(nonzero_y.min()), int(nonzero_y.max())
            xmin, xmax = int(nonzero_x.min()), int(nonzero_x.max())

            crop_rgb = color[ymin : ymax + 1, xmin : xmax + 1].copy()
            crop_mask = bin_mask[ymin : ymax + 1, xmin : xmax + 1].copy()

            obj_inst = ObjectInstance(
                obj_id=obj_id,
                form=form,
                material=mat,
                volume_ml=vol,
                mass_g=mass,
                take=take,
                crop_rgb=crop_rgb,
                crop_mask=crop_mask,
                source="synth",
                donor_frame=img_dir.name,
            )

            self.objects_by_class[(form, mat)].append(obj_inst)
            self.objects_by_id[obj_id].append(obj_inst)
            loaded_count += 1

        print(f"[DatasetLibrary] Loaded {loaded_count} source synthetic object instances across {len(self.objects_by_class)} classes.")
        for (f, m), objs in sorted(self.objects_by_class.items()):
            print(f"  - {f:<5} {m:<8}: {len(objs):3d} source instances (unique obj_ids: {len(set(o.obj_id for o in objs))})")

    def get_background(self, rng=None):
        """Returns a loaded empty background image (BGR) and its path."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(len(self.empty_backgrounds))
        bg_path = self.empty_backgrounds[idx]
        bg_img = cv2.imread(str(bg_path))
        return bg_img, bg_path

    def sample_object(self, form, material, rng=None):
        """Randomly samples an object instance of the specified form and material."""
        if rng is None:
            rng = np.random.default_rng()
        candidates = self.objects_by_class.get((form, material), [])
        if not candidates:
            raise ValueError(f"No source objects for class: ({form}, {material})")
        idx = rng.integers(len(candidates))
        return candidates[idx]
