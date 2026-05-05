import os
import sys
import gc
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from model.network import NeuralNetwork
from configs.default import CFG_INFERENCE

# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_DIR = "data/assets/One-Punchman_pages"

# ── Layer types ────────────────────────────────────────────
LAYER_TYPE_NAMES = {
    0:  "Flatten",
    1:  "FC",
    2:  "Conv",
    3:  "MaxPool",
    4:  "SumPool",
    5:  "LayerNorm",
    6:  "Transformer",
    7:  "Dropout",
    8:  "Add",
    9:  "GPoolMax",
    10: "GPoolSum",
    11: "NNUpsampling",
    12: "Concatenation",
    13: "BilinearUpsampling",
    14: "Bias",
    15: "FlattenSpatial",
    16: "UnflattenSpatial",
    17: "PositionalEmbedding2D",
    18: "TransformerEncoder",
    19: "Branch",
    20: "CrossAttention"
}

class VisualizeActivations:
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

    def show_layer(self, feat, layer_idx, layer_type_id, n_channels=16, cmap="viridis"):
        feat = np.array(feat[0])  # first batch item

        if feat.ndim != 3:
            return
        H, W, C = feat.shape
        if W == 1:
            return

        n_show = min(n_channels, C)
        cols   = min(8, n_show)
        rows   = (n_show + cols - 1) // cols

        layer_name = LAYER_TYPE_NAMES.get(layer_type_id, f"type_{layer_type_id}")
        fig, axes  = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes       = np.array(axes).reshape(-1)

        fig.suptitle(
            f"Layer {layer_idx:03d}  —  {layer_name}  —  shape {feat.shape}",
            fontsize=12, y=1.01,
        )

        for c in range(n_show):
            channel  = feat[:, :, c]
            vmin, vmax = channel.min(), channel.max()
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
            axes[c].imshow(channel, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[c].set_title(f"ch {c}", fontsize=7)
            axes[c].axis("off")

        for c in range(n_show, len(axes)):
            axes[c].axis("off")

        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    def run(self, image_path, n_channels=16, cmap="viridis"):
        img     = self.load_image(image_path)
        x_batch = self.model.NHWC_check(img[None, ...])

        saves = self.model.predict_saves(x_batch)

        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        ax.imshow(np.array(x_batch[0, :, :, 0]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Input — {Path(image_path).name}", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

        configs = self.model.layer_configs_static
        for idx in range(self.model.num_layers):
            feat          = saves[idx + 1]
            layer_type_id = configs[idx].type
            self.show_layer(feat, idx + 1, layer_type_id, n_channels, cmap)

    def cleanup(self):
        plt.close("all")
        if self.model is not None:
            self.model.release_gpu()
            del self.model
            self.model = None
        gc.collect()


if __name__ == "__main__":
    try:
        viz = VisualizeActivations(CFG_INFERENCE)
        viz.build_model()
        viz.run("data/assets/One-Punchman_pages/0025.png", n_channels=16, cmap="viridis")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        viz.cleanup()
        sys.exit(0)