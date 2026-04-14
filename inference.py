import os
import glob
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model.network import NeuralNetwork
from configs.default import CFG_INFERENCE

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_DIR      = "data/assets/One-Punchman_pages"
ANNOTATION_DIR = "data/assets/Annotations"
CHECKPOINT_DIR = "checkpoints"


class Inference:
    def __init__(self, cfg=CFG_INFERENCE):
        self.cfg   = cfg
        self.H     = cfg.g.H
        self.W     = cfg.g.W
        self.C     = cfg.g.C
        self.model = None

    def build_model(self):
        self.model = NeuralNetwork(self.cfg)

    def load_image(self, path):
        img = Image.open(path).convert("L" if self.C == 1 else "RGB")
        img = img.resize((self.W, self.H))
        img = np.array(img).astype(np.float32) / 255.0
        if self.C == 1:
            img = img[..., None]
        return img

    def load_annotations(self, path):
        data        = np.load(path)
        img_arr     = data["image"]
        targets_arr = data["targets"]

        img_pil = Image.fromarray((img_arr[:, :, 0] * 255.0).astype(np.uint8), mode="L")
        img_pil = img_pil.resize((self.W, self.H), Image.LANCZOS)
        image   = np.array(img_pil, dtype=np.float32)[..., None] / 255.0

        resized_channels = []
        for ch in range(targets_arr.shape[-1]):
            ch_arr    = targets_arr[:, :, ch]
            ch_pil    = Image.fromarray(ch_arr, mode="F")
            ch_pil    = ch_pil.resize((self.W, self.H), Image.BICUBIC)
            ch_resized = np.clip(np.array(ch_pil, dtype=np.float32), 0.0, 1.0)
            resized_channels.append(ch_resized)

        targets = np.stack(resized_channels, axis=-1)
        return image, targets

    @staticmethod
    def soft_dice_score(y_pred, y_true, eps=1.0):
        intersection = np.sum(y_pred * y_true)
        union        = np.sum(y_pred) + np.sum(y_true)
        return float((2.0 * intersection + eps) / (union + eps))

    def score_annotation(self, image, targets):
        x_batch    = image[None, ...]
        x_batch    = self.model.NHWC_check(x_batch)
        y_pred     = np.array(self.model.predict(x_batch))[0]
        char_dice  = self.soft_dice_score(y_pred[:, :, 0], targets[:, :, 0])
        affin_dice = self.soft_dice_score(y_pred[:, :, 1], targets[:, :, 1])
        return {
            "char_dice":  char_dice,
            "affin_dice": affin_dice,
            "mean_dice":  (char_dice + affin_dice) / 2.0,
        }

    def compare_weights(self, checkpoint_dir=CHECKPOINT_DIR, annotation_dir=ANNOTATION_DIR):
        checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pkl")))
        if not checkpoint_paths:
            print("No checkpoints found.")
            return

        annotation_paths = sorted(glob.glob(os.path.join(annotation_dir, "*.npz")))
        if not annotation_paths:
            print("No annotations found.")
            return

        self.model = NeuralNetwork(self.cfg)

        annotations = []
        for npz_path in annotation_paths:
            image, targets = self.load_annotations(npz_path)
            annotations.append({
                "name":    os.path.basename(npz_path),
                "image":   image,
                "targets": targets,
            })

        results = []
        for ckpt_path in checkpoint_paths:
            ckpt_name = os.path.basename(ckpt_path)
            self.model.load_filepath = ckpt_path
            self.model.load_weights(inference=True)
            per_image_scores = [
                self.score_annotation(ann["image"], ann["targets"])
                for ann in annotations
            ]

            mean_char  = float(np.mean([s["char_dice"]  for s in per_image_scores]))
            mean_affin = float(np.mean([s["affin_dice"] for s in per_image_scores]))
            mean_total = float(np.mean([s["mean_dice"]  for s in per_image_scores]))

            results.append({
                "path":       ckpt_path,
                "name":       ckpt_name,
                "char_dice":  mean_char,
                "affin_dice": mean_affin,
                "mean_dice":  mean_total,
            })
            print(f"  [{ckpt_name:<20}]  char={mean_char:.4f}  affin={mean_affin:.4f}  mean={mean_total:.4f}")

        results.sort(key=lambda r: r["mean_dice"], reverse=True)
        print(f"\n{'─'*60}")
        print(f"{'Top 3 checkpoints':^60}")
        print(f"{'─'*60}")
        for rank, r in enumerate(results[:3], 1):
            print(f"  #{rank}  {r['name']}  mean={r['mean_dice']:.4f}  (char={r['char_dice']:.4f}, affin={r['affin_dice']:.4f})")
        print(f"{'─'*60}")
        return results

    def run(self):
        self.build_model()
        image_files = sorted(
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        for fname in image_files:
            img = self.load_image(os.path.join(IMAGE_DIR, fname))
            self.model.plot_predictions(img)
            self.model.heatmap_decoder(img)


if __name__ == "__main__":
    try:
        inf = Inference(CFG_INFERENCE)
        inf.run()
    except KeyboardInterrupt:
        print("Interrupted.")