import os
if os.environ.get("JAX_PLATFORMS") is None:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from pathlib import Path
from PIL import Image as PILImage


class BackgroundGenerator:
    def __init__(self, config, **kwargs):
        from configs.configurations import TrainConfig, BenchmarkConfig
        if isinstance(config, (TrainConfig, BenchmarkConfig)):
            global_config     = config.g
            background_config = config.background

            self.H        = global_config.H
            self.W        = global_config.W
            self.C        = global_config.C
            self.seed     = global_config.seed
            self.N_images = global_config.batch_size
            self.N_images = kwargs.get("N_images", self.N_images)

            self.prob_real  = background_config.prob_real
            self.brightness = background_config.brightness

            self.prob_solid      = background_config.prob_solid
            self.prob_screentone = background_config.prob_screentone
            self.prob_hatching   = background_config.prob_hatching
            self.weights         = np.array([self.prob_solid, self.prob_screentone, self.prob_hatching])

            self.prob_panel_border = background_config.prob_panel_border

            self.solid_value = background_config.solid_value

            self.screentone_dot_radius     = background_config.screentone_dot_radius
            self.screentone_spacing_factor = background_config.screentone_spacing_factor
            self.screentone_angle          = background_config.screentone_angle
            self.screentone_dot_value      = background_config.screentone_dot_value

            self.hatching_angle      = background_config.hatching_angle
            self.hatching_spacing    = background_config.hatching_spacing
            self.hatching_thickness  = background_config.hatching_thickness
            self.hatching_line_value = background_config.hatching_line_value

            self.panel_n_h_cuts     = background_config.panel_n_h_cuts
            self.panel_n_v_cuts     = background_config.panel_n_v_cuts
            self.panel_border_width = background_config.panel_border_width

        self.image_folder = Path(__file__).parent / "assets" / "Backgrounds"
        self.real_images  = None

        if self.prob_real > 0.0:
            self.real_images = self.load_images()
            if self.real_images is None:
                raise FileNotFoundError(
                    f"No background images were found in: {self.image_folder}"
                )
    def load_images(self):
        root = Path(self.image_folder)
        paths = [p for p in root.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

        images = []
        for p in paths:
            img = PILImage.open(p).convert("L")
            img = img.resize((self.W, self.H))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr[..., None]
            images.append(arr)
        return np.stack(images, axis=0)
    
    def brightness_augmentation(self, bg, np_rng):
        low, high  = self.brightness
        brightness = np_rng.uniform(low, high)
        bg         = np.clip(bg * brightness, 0.0, 1.0).astype(np.float32)
        return bg

    def solid_augmentation(self, bg, np_rng):
        lo    = self.solid_value[0]
        hi    = self.solid_value[1]
        value = np_rng.uniform(lo, hi)
        return np.full((self.H, self.W, self.C), value, dtype=np.float32)
    
    def screentone_augmentation(self, bg, np_rng):
        dot_radius = np_rng.uniform(self.screentone_dot_radius[0], self.screentone_dot_radius[1])
        spacing    = np_rng.uniform(dot_radius * self.screentone_spacing_factor[0], dot_radius * self.screentone_spacing_factor[1])
        angle_deg  = np_rng.uniform(self.screentone_angle[0], self.screentone_angle[1])
        dot_value  = np_rng.uniform(self.screentone_dot_value[0], self.screentone_dot_value[1])
        angle      = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        u      = cos_a * xs + sin_a * ys
        v      = -sin_a * xs + cos_a * ys
        u_mod  = (u % spacing) - spacing / 2
        v_mod  = (v % spacing) - spacing / 2
        dist   = np.sqrt(u_mod**2 + v_mod**2)
        dots   = dist < dot_radius
        bg[dots] = dot_value
        return bg

    def hatching_augmentation(self, bg, np_rng):
        angle_deg = np_rng.uniform(self.hatching_angle[0],      self.hatching_angle[1])
        spacing   = np_rng.uniform(self.hatching_spacing[0],    self.hatching_spacing[1])
        thickness = np_rng.uniform(self.hatching_thickness[0],  self.hatching_thickness[1])
        line_val  = np_rng.uniform(self.hatching_line_value[0], self.hatching_line_value[1])
        angle     = np.deg2rad(angle_deg)
        ys, xs    = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        proj      = xs * np.cos(angle) + ys * np.sin(angle)
        lines     = (proj % spacing) < thickness
        bg[lines] = line_val
        return bg

    def panel_borders_augmentation(self, bg, np_rng):
        n_h_cuts     = np_rng.integers(self.panel_n_h_cuts[0],     self.panel_n_h_cuts[1])
        n_v_cuts     = np_rng.integers(self.panel_n_v_cuts[0],     self.panel_n_v_cuts[1])
        border_width = np_rng.integers(self.panel_border_width[0], self.panel_border_width[1])
        border_val   = 0.0

        h_positions = sorted(np_rng.uniform(0.15, 0.85, size=n_h_cuts))
        v_positions = sorted(np_rng.uniform(0.15, 0.85, size=n_v_cuts))

        for frac in h_positions:
            y  = int(frac * self.H)
            y0 = max(0, y - border_width // 2)
            y1 = min(self.H, y + border_width // 2)
            bg[y0:y1, :, :] = border_val

        for frac in v_positions:
            x  = int(frac * self.W)
            x0 = max(0, x - border_width // 2)
            x1 = min(self.W, x + border_width // 2)
            bg[:, x0:x1, :] = border_val

        return bg

    def make_real_images(self, np_rng):
        idx = np_rng.integers(0, len(self.real_images))
        bg  = self.real_images[idx].copy()
        bg  = self.brightness_augmentation(bg, np_rng)
        return bg.astype(np.float32)

    def make_synthetic_image(self, np_rng):
        bg     = np.ones((self.H, self.W, self.C), dtype=np.float32)
        choice = np_rng.choice(3, p=self.weights)
        if choice == 0:
            bg = self.solid_augmentation(bg, np_rng)
        elif choice == 1:
            bg = self.screentone_augmentation(bg, np_rng)
        else:
            bg = self.hatching_augmentation(bg, np_rng)
        if np_rng.random() < self.prob_panel_border:
            bg = self.panel_borders_augmentation(bg, np_rng)
        bg = self.brightness_augmentation(bg, np_rng)
        return bg.astype(np.float32)

    def make_background(self, np_rng):
        backgrounds = []
        for _ in range(self.N_images):
            if np_rng.random() < self.prob_real:
                bg = self.make_real_images(np_rng)
            else:
                bg = self.make_synthetic_image(np_rng)
            backgrounds.append(bg)
        return np.stack(backgrounds, axis=0)