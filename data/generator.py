import os
if os.environ.get("JAX_PLATFORMS") is None:
    os.environ["JAX_PLATFORMS"] = "cpu"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "generator")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR

import io
import time
from functools import partial
from dataclasses import replace
from typing import NamedTuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageDraw
import cv2
import jax
import jax.numpy as jnp
from jax import jit, random, lax

from data.atlas import Atlas
from data.background import BackgroundGenerator

jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

ELLIPSE   = 0
JAGGED    = 1
RECTANGLE = 2
WAVY      = 3
NONE      = 4

class SentenceBatch(NamedTuple):
    sentences:       jnp.ndarray
    font_ids:        jnp.ndarray
    lengths:         jnp.ndarray
    directions:      jnp.ndarray
    scalings:        jnp.ndarray
    max_line_widths:  jnp.ndarray
    max_line_heights: jnp.ndarray
    image_ids:       jnp.ndarray

class LayoutAug(NamedTuple):
    char_spacing_vertical:   jnp.ndarray
    char_spacing_horizontal: jnp.ndarray
    jitter:                  jnp.ndarray
    rotation:                jnp.ndarray

class RenderAug(NamedTuple):
    intensity: jnp.ndarray
    stroke:    jnp.ndarray
    blur:      jnp.ndarray
    dropout:   jnp.ndarray
    scaling:   jnp.ndarray

class ImageAug(NamedTuple):
    noise_sigma:      jnp.ndarray
    brightness:       jnp.ndarray
    colour_inversion: jnp.ndarray

class CharHeatmapAug(NamedTuple):
    sigma:     jnp.ndarray
    radius:    jnp.ndarray
    intensity: jnp.ndarray
    kernel:    jnp.ndarray

class AffinHeatmapAug(NamedTuple):
    sigma:     jnp.ndarray
    radius:    jnp.ndarray
    intensity: jnp.ndarray
    kernel:    jnp.ndarray

class RelativeGeo(NamedTuple):
    x_pos:     jnp.ndarray
    y_pos:     jnp.ndarray
    centers:   jnp.ndarray
    pairs:     jnp.ndarray
    pair_mask: jnp.ndarray

class Geo(NamedTuple):
    x_pos:     jnp.ndarray
    y_pos:     jnp.ndarray
    centers:   jnp.ndarray
    pairs:     jnp.ndarray
    pair_mask: jnp.ndarray

class SyntheticDataGenerator:
    def __init__(self, config, workers_init=False, i=None, **kwargs):
        from configs.configurations import TrainConfig, BenchmarkConfig
        if isinstance(config, (TrainConfig, BenchmarkConfig)):
            global_config        = config.g
            atlas_config         = config.atlas
            text_config          = config.text
            background_config    = config.background
            sentence_config      = config.sentence
            char_heatmap_config  = config.char
            affin_heatmap_config = config.affin
            speech_bubble_config = config.speechbubble
            image_aug_config     = config.image
            real_data_config     = config.real_data

            # Global params
            self.H        = global_config.H
            self.W        = global_config.W
            self.C        = global_config.C
            self.seed     = global_config.seed
            self.N_images = global_config.batch_size
            self.N_images = kwargs.get("N_images", self.N_images)

            self.text_config          = text_config
            self.background_config    = background_config
            self.sentence_config      = sentence_config
            self.char_heatmap_config  = char_heatmap_config
            self.affin_heatmap_config = affin_heatmap_config
            self.image_aug_config     = image_aug_config
            self.speech_bubble_config = speech_bubble_config
            self.real_data_config     = real_data_config 

            self.probs     = np.array([tc.prob for tc in text_config], dtype=np.float32)
            self.probs     = self.probs / self.probs.sum()
            self.max_char  = max(tc.layoutconfig.length[1] for tc in text_config)
            self.bubble_margin = sentence_config.bubble_margin
            self.image_aug     = self.image_aug_config.image_aug

            # Image ids params
            self.min_repeat          = sentence_config.max_sentences_per_image[0]
            self.max_repeat          = sentence_config.max_sentences_per_image[1]
            self.sentence_count_mean = sentence_config.sentence_count_mean
            self.sentence_count_std  = sentence_config.sentence_count_std
            self.prob_next_sentence  = sentence_config.prob_next_sentence

            # Affinity heatmap
            self.n_steps = int(self.affin_heatmap_config.n_steps)

            # Sentence params
            self.max_sentence_per_image = self.max_repeat
            self.N_unique_sentences     = self.N_images * self.max_sentence_per_image
            self.max_line_height_bounds = np.array(sentence_config.max_line_height_bounds, dtype=np.int32)  # vertical direction
            self.max_line_width_bounds  = np.array(sentence_config.max_line_width_bounds,  dtype=np.int32)  # horizontal direction
            self.prob_real_sentence     = sentence_config.prob_real_sentence

            # Bounds for layout, render and kernels
            self.initialize_bounds()

            # Atlas
            atlas_config    = replace(atlas_config, max_scaling_sentence=self.max_scaling)
            self.atlas      = Atlas(atlas_config)
            self.glyph_size = self.atlas.glyph_size
            self.max_glyph_width  = float(self.atlas.max_glyph_width)
            self.max_glyph_height = float(self.atlas.max_glyph_height)
            self.ink_offset_arr   = jnp.array(self.atlas.ink_offsets, dtype=jnp.float32)
            self.glyphs           = jnp.array(self.atlas.glyphs)
            self.atlas.glyphs     = None
            self.char_choices     = np.array(self.atlas.char_list)
            self.char_to_idx      = self.atlas.char_to_idx

            # RNG
            self.np_rng  = np.random.default_rng(self.seed)
            self.jax_rng = jax.random.PRNGKey(self.seed)

            # Real sentences
            if self.prob_real_sentence != 0.0:
                self.sentences_idxs = self.load_sentences()

            # Background
            self.speech_bubble        = speech_bubble_config.speech_bubble
            self.background_generator = None
            if self.background_config.background:
                self.background_generator = BackgroundGenerator(config=config, N_images=self.N_images)

            # Real data
            self.real_data = []
            st_dir = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "Self_training")
            if os.path.isdir(st_dir) and self.real_data_config.prob_real_data > 0.0:
                for p in sorted(Path(st_dir).glob("*.npz")):
                    data = np.load(str(p))
                    self.real_data.append((
                        data["image"].astype(np.float32),
                        data["targets"].astype(np.float32),
                    ))
                print(f"Self-training pool: {len(self.real_data)} samples")


            # Pre-jit
            if workers_init:
                self.warmup(i)

    def warmup(self, i=None):
        if i is not None:
            print(f"Pre-jitting the {i}th worker")
        t0 = time.time()

        N  = self.N_unique_sentences
        MC = self.max_char
        effective_max_sc = max(1, float(self.scaling_bounds[:, 1].max()))
        ks_char  = 2 * int(np.ceil(self.char_radius_bounds[1]  * effective_max_sc)) + 1
        ks_affin = 2 * int(np.ceil(self.affin_radius_bounds[1] * effective_max_sc)) + 1
        border_r = int(self.stroke_bounds_b[:, 1].max())

        sentence_batch = SentenceBatch(
            sentences        = jnp.zeros((N, MC), dtype=jnp.int32),
            font_ids         = jnp.zeros(N,       dtype=jnp.int32),
            lengths          = jnp.ones (N,       dtype=jnp.int32),
            directions       = jnp.zeros(N,       dtype=jnp.int32),
            scalings         = jnp.ones (N,       dtype=jnp.float32),
            max_line_widths  = jnp.ones (N,       dtype=jnp.int32),
            max_line_heights = jnp.ones (N,       dtype=jnp.int32),
            image_ids        = jnp.zeros(N,       dtype=jnp.int32),
        )
        layout_aug = LayoutAug(
            char_spacing_vertical   = jnp.ones (N, dtype=jnp.float32),
            char_spacing_horizontal = jnp.ones (N, dtype=jnp.float32),
            jitter                  = jnp.zeros((N, MC, 2), dtype=jnp.float32),
            rotation                = jnp.zeros(N, dtype=jnp.float32),
        )
        render_aug = RenderAug(
            intensity = jnp.ones (N, dtype=jnp.float32),
            stroke    = jnp.ones (N, dtype=jnp.float32),
            blur      = jnp.zeros(N, dtype=jnp.float32),
            dropout   = jnp.zeros(N, dtype=jnp.float32),
            scaling   = jnp.ones (N, dtype=jnp.float32),
        )
        char_aug = CharHeatmapAug(
            sigma     = jnp.ones (N, dtype=jnp.float32),
            radius    = jnp.ones (N, dtype=jnp.float32),
            intensity = jnp.ones (N, dtype=jnp.float32),
            kernel    = jnp.zeros((N, ks_char,  ks_char),  dtype=jnp.float32),
        )
        affin_aug = AffinHeatmapAug(
            sigma     = jnp.ones (N, dtype=jnp.float32),
            radius    = jnp.ones (N, dtype=jnp.float32),
            intensity = jnp.ones (N, dtype=jnp.float32),
            kernel    = jnp.zeros((N, ks_affin, ks_affin), dtype=jnp.float32),
        )
        backgrounds = jnp.ones((self.N_images, self.H, self.W, self.C), dtype=jnp.float32)

        rel_geo, boxes = SyntheticDataGenerator.compute_relative_geometries(
            sentence_batch, layout_aug,
            self.ink_offset_arr,
            self.max_char, self.N_unique_sentences,
        )
        start_coords = jnp.zeros((N, 2), dtype=jnp.float32)
        geos = SyntheticDataGenerator.apply_coords(rel_geo, start_coords)

        SyntheticDataGenerator.make_images_jitted(
            geos, sentence_batch, render_aug, backgrounds,
            self.N_images, self.H, self.W,
            self.glyphs,
            self.atlas.glyph_height, self.atlas.glyph_width, border_r=border_r,
        )
        SyntheticDataGenerator.make_char_heatmaps_jitted(
            geos, sentence_batch, char_aug,
            self.N_images, self.H, self.W,
        )
        SyntheticDataGenerator.make_affinity_heatmaps_jitted(
            geos, sentence_batch, affin_aug,
            self.N_images, self.H, self.W,
            self.n_steps,
        )

        jax.effects_barrier()
        if i is not None:
            print(f"Finished Pre-jit in {time.time() - t0:.1f}s")

    def initialize_bounds(self):
        tc = self.text_config
        self.N_configs = len(tc)

        self.length_bounds     = np.array([t.layoutconfig.length    for t in tc], dtype=np.int32)
        self.direction_probs   = np.array([t.layoutconfig.direction  for t in tc], dtype=np.float32)
        self.scaling_bounds    = np.array([t.sentence_scaling        for t in tc], dtype=np.float32)

        self.char_spacing_v_b  = np.array([t.layoutconfig.char_spacing_vertical   for t in tc], dtype=np.float32)
        self.char_spacing_h_b  = np.array([t.layoutconfig.char_spacing_horizontal for t in tc], dtype=np.float32)
        self.jitter_bounds_b   = np.array([t.layoutconfig.jitters                 for t in tc], dtype=np.float32)
        self.rotation_bounds_b = np.array([t.layoutconfig.rotation                for t in tc], dtype=np.float32)

        self.stroke_bounds_b    = np.array([t.renderconfig.stroke    for t in tc], dtype=np.int32)
        self.blur_bounds_b      = np.array([t.renderconfig.blur      for t in tc], dtype=np.float32)
        self.intensity_bounds_b = np.array([t.renderconfig.intensity for t in tc], dtype=np.float32)
        self.dropout_bounds_b   = np.array([t.renderconfig.dropout   for t in tc], dtype=np.float32)

        self.char_sigma_bounds     = np.array(self.char_heatmap_config.sigma,     dtype=np.float32)
        self.char_radius_bounds    = np.array(self.char_heatmap_config.radius,    dtype=np.float32)
        self.char_intensity_bounds = np.array(self.char_heatmap_config.intensity, dtype=np.float32)

        self.affin_sigma_bounds     = np.array(self.affin_heatmap_config.sigma,     dtype=np.float32)
        self.affin_radius_bounds    = np.array(self.affin_heatmap_config.radius,    dtype=np.float32)
        self.affin_intensity_bounds = np.array(self.affin_heatmap_config.intensity, dtype=np.float32)

        self.prob_noise            = self.image_aug_config.prob_noise
        self.prob_brightness       = self.image_aug_config.prob_brightness
        self.prob_jpeg             = self.image_aug_config.prob_jpeg
        self.prob_colour_inversion = self.image_aug_config.prob_colour_inversion
        self.prob_x_flip           = self.image_aug_config.prob_x_flip
        self.prob_y_flip           = self.image_aug_config.prob_y_flip

        self.noise_sigma_bounds  = np.array(self.image_aug_config.noise_sigma,   dtype=np.float32)
        self.brightness_bounds   = np.array(self.image_aug_config.brightness,    dtype=np.float32)
        self.jpeg_quality_bounds = np.array(self.image_aug_config.jpeg_quality,  dtype=np.float32)
        self.crop_scale_bounds   = np.array(self.image_aug_config.crop_size,     dtype=np.float32)

        self.max_scaling = int(np.max(self.scaling_bounds))

    def load_sentences(self):
        sentences_path = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "Sentences", "sentences.txt")
        if not os.path.exists(sentences_path):
            raise FileNotFoundError(
                f"Sentences file not found: {sentences_path}"
            )
        
        sentences = []
        with open(sentences_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                sentences.append(line)
        sentences_idxs = []

        for i in sentences:
            idxs = []
            for ch in i:
                if ch in self.char_to_idx:
                    idxs.append(self.char_to_idx[ch])
                else:
                    print(ch)
            if idxs:
                sentences_idxs.append(idxs)

        if len(sentences_idxs) == 0:
            raise ValueError(
                f"The characters in the sentences file may not exist in the atlas character list."
            )
        
        self.N_load_sentences       = len(sentences_idxs)
        self.lengths_load_sentences = list(map(len, sentences_idxs))
        self.max_char               = max(self.max_char, max(self.lengths_load_sentences))
        return sentences_idxs

    def make_image_ids(self):
        image_ids = np.full(self.N_unique_sentences, -1, dtype=np.int32)

        if self.N_unique_sentences == 0:
            return image_ids

        mu    = self.sentence_count_mean
        sigma = self.sentence_count_std
        lo    = self.min_repeat
        hi    = self.max_repeat

        counts = np.round(
            self.np_rng.normal(mu, sigma, size=self.N_images)
        ).astype(np.int32)
        counts = np.clip(counts, lo, hi)

        sentence_idx = 0
        for img_idx in range(self.N_images):
            n = counts[img_idx]
            for _ in range(n):
                if sentence_idx >= self.N_unique_sentences:
                    break
                image_ids[sentence_idx] = img_idx
                sentence_idx += 1
            if sentence_idx >= self.N_unique_sentences:
                break

        return image_ids

    def sample_batch(self):
        rng = self.np_rng
        N   = self.N_unique_sentences
        tc  = self.text_config

        config_idx      = rng.choice(self.N_configs, size=N, p=self.probs)
        self.config_idx = config_idx

        lengths    = rng.integers(self.length_bounds[config_idx, 0], self.length_bounds[config_idx, 1]).astype(np.int32)
        directions = (rng.random(N) >= self.direction_probs[config_idx]).astype(np.int32)
        scalings   = rng.uniform(self.scaling_bounds[config_idx, 0], self.scaling_bounds[config_idx, 1]).astype(np.float32)

        # Layout
        char_spacing_v = rng.uniform(self.char_spacing_v_b[config_idx, 0], self.char_spacing_v_b[config_idx, 1]).astype(np.float32)
        char_spacing_h = rng.uniform(self.char_spacing_h_b[config_idx, 0], self.char_spacing_h_b[config_idx, 1]).astype(np.float32)
        jitter_x       = rng.uniform(self.jitter_bounds_b[config_idx, 0][:, None], self.jitter_bounds_b[config_idx, 1][:, None], size=(N, self.max_char)).astype(np.float32)
        jitter_y       = rng.uniform(self.jitter_bounds_b[config_idx, 0][:, None], self.jitter_bounds_b[config_idx, 1][:, None], size=(N, self.max_char)).astype(np.float32)
        jitter         = np.stack([jitter_x, jitter_y], axis=-1)
        rotation       = rng.uniform(self.rotation_bounds_b[config_idx, 0], self.rotation_bounds_b[config_idx, 1]).astype(np.float32)
        max_line_widths  = rng.integers(self.max_line_width_bounds[0],  self.max_line_width_bounds[1],  size=N).astype(np.int32)
        max_line_heights = rng.integers(self.max_line_height_bounds[0], self.max_line_height_bounds[1], size=N).astype(np.int32)

        # Render
        stroke    = rng.integers(self.stroke_bounds_b[config_idx, 0],    self.stroke_bounds_b[config_idx, 1]).astype(np.float32)
        blur      = rng.uniform(self.blur_bounds_b[config_idx, 0],        self.blur_bounds_b[config_idx, 1]).astype(np.float32)
        intensity = rng.uniform(self.intensity_bounds_b[config_idx, 0],   self.intensity_bounds_b[config_idx, 1]).astype(np.float32)
        dropout   = rng.uniform(self.dropout_bounds_b[config_idx, 0],     self.dropout_bounds_b[config_idx, 1]).astype(np.float32)

        # Char heatmap
        char_sigma_base        = rng.uniform(self.char_sigma_bounds[0],     self.char_sigma_bounds[1],     size=N).astype(np.float32)
        char_radius_base       = rng.uniform(self.char_radius_bounds[0],    self.char_radius_bounds[1],    size=N).astype(np.float32)
        char_intensity         = rng.uniform(self.char_intensity_bounds[0], self.char_intensity_bounds[1], size=N).astype(np.float32)
        char_sigma             = char_sigma_base  * scalings
        char_radius            = char_radius_base * scalings
        effective_max_sc       = max(1, float(self.scaling_bounds[:, 1].max()))
        max_kernel_radius_char = int(np.ceil(self.char_radius_bounds[1] * effective_max_sc))
        char_kernels           = SyntheticDataGenerator.make_kernels(char_sigma, char_radius, char_intensity, max_kernel_radius_char)

        # Affinity heatmap
        affin_sigma_base        = rng.uniform(self.affin_sigma_bounds[0],     self.affin_sigma_bounds[1],     size=N).astype(np.float32)
        affin_radius_base       = rng.uniform(self.affin_radius_bounds[0],    self.affin_radius_bounds[1],    size=N).astype(np.float32)
        affin_intensity         = rng.uniform(self.affin_intensity_bounds[0], self.affin_intensity_bounds[1], size=N).astype(np.float32)
        affin_sigma             = affin_sigma_base  * scalings
        affin_radius            = affin_radius_base * scalings
        max_kernel_radius_affin = int(np.ceil(self.affin_radius_bounds[1] * effective_max_sc))
        affin_kernels           = SyntheticDataGenerator.make_kernels(affin_sigma, affin_radius, affin_intensity, max_kernel_radius_affin)

        # Image augmentation
        # Jax part
        noise_sigma = np.where(rng.random(self.N_images) < self.prob_noise,      rng.uniform(self.noise_sigma_bounds[0], self.noise_sigma_bounds[1], size=self.N_images), -1.0).astype(np.float32)
        brightness  = np.where(rng.random(self.N_images) < self.prob_brightness, rng.uniform(self.brightness_bounds[0], self.brightness_bounds[1], size=self.N_images), -1.0).astype(np.float32)
        colour_inversion = np.where(rng.random(self.N_images) < self.prob_colour_inversion, 1.0, -1.0).astype(np.float32)
        
        # Numpy part
        self.jpeg_quality = rng.uniform(self.jpeg_quality_bounds[0], self.jpeg_quality_bounds[1], size=self.N_images).astype(np.float32)
        self.crop_scale   = rng.uniform(self.crop_scale_bounds[0], self.crop_scale_bounds[1])

        # Sentences
        sentences = np.full((N, self.max_char), -1, dtype=np.int32)
        font_ids  = np.full((N), -1, dtype=np.int32)

        for i in range(N):
            cfg    = tc[config_idx[i]]
            length = lengths[i]

            font_probs  = np.array([fc.prob for fc in cfg.fonts], dtype=np.float32)
            font_ids[i] = self.np_rng.choice(len(cfg.fonts), p=font_probs)
            chars       = self.np_rng.choice(self.char_choices, size=length)
            if self.np_rng.random() < 1 - self.prob_real_sentence:
                sentences[i, :length] = [self.char_to_idx[ch] for ch in chars]
            else:
                idx = self.np_rng.integers(0, self.N_load_sentences)
                sentences[i, :self.lengths_load_sentences[idx]] = self.sentences_idxs[idx]
                lengths[i] = self.lengths_load_sentences[idx]

        image_ids = self.make_image_ids()

        sentence_batch = SentenceBatch(
            sentences        = jnp.asarray(sentences),
            font_ids         = jnp.asarray(font_ids),
            lengths          = jnp.asarray(lengths),
            directions       = jnp.asarray(directions),
            scalings         = jnp.asarray(scalings),
            max_line_widths  = jnp.asarray(max_line_widths),
            max_line_heights = jnp.asarray(max_line_heights),
            image_ids        = jnp.asarray(image_ids),
        )

        layout_aug = LayoutAug(
            char_spacing_vertical   = jnp.asarray(char_spacing_v),
            char_spacing_horizontal = jnp.asarray(char_spacing_h),
            jitter                  = jnp.asarray(jitter),
            rotation                = jnp.asarray(rotation),
        )

        render_aug = RenderAug(
            intensity = jnp.asarray(intensity),
            stroke    = jnp.asarray(stroke),
            blur      = jnp.asarray(blur),
            dropout   = jnp.asarray(dropout),
            scaling   = jnp.asarray(scalings),
        )

        charheatmap_aug = CharHeatmapAug(
            sigma     = jnp.asarray(char_sigma),
            radius    = jnp.asarray(char_radius),
            intensity = jnp.asarray(char_intensity),
            kernel    = char_kernels,
        )

        affinheatmap_aug = AffinHeatmapAug(
            sigma     = jnp.asarray(affin_sigma),
            radius    = jnp.asarray(affin_radius),
            intensity = jnp.asarray(affin_intensity),
            kernel    = affin_kernels,
        )

        image_aug = ImageAug(
            noise_sigma = jnp.asarray(noise_sigma),
            brightness  = jnp.asarray(brightness),
            colour_inversion = jnp.asarray(colour_inversion), 
        )

        return sentence_batch, layout_aug, render_aug, charheatmap_aug, affinheatmap_aug, image_aug

    @staticmethod
    @partial(jit, static_argnames=("max_char", "N_unique_sentences"))
    def compute_relative_geometries(sentence_batch, layout_aug, ink_offset_arr, max_char, N_unique_sentences):

        sentences  = sentence_batch.sentences
        font_ids   = sentence_batch.font_ids
        lengths    = sentence_batch.lengths
        scalings   = sentence_batch.scalings
        max_lws    = sentence_batch.max_line_widths
        max_lhs    = sentence_batch.max_line_heights
        directions = sentence_batch.directions
        sp_h       = layout_aug.char_spacing_horizontal
        sp_v       = layout_aug.char_spacing_vertical
        jitter     = layout_aug.jitter

        N  = N_unique_sentences
        MC = max_char

        safe_chars = jnp.clip(sentences, 0, ink_offset_arr.shape[1] - 1)
        char_ink   = ink_offset_arr[font_ids[:, None], safe_chars]
        pxs = char_ink[:, :, 0]
        pys = char_ink[:, :, 1]
        iws = char_ink[:, :, 2]
        ihs = char_ink[:, :, 3]

        valid  = (jnp.arange(MC)[None, :] < lengths[:, None]) & (sentences >= 0)
        max_iw = jnp.max(jnp.where(valid, iws, 0.0), axis=1)
        max_ih = jnp.max(jnp.where(valid, ihs, 0.0), axis=1)
        max_px = jnp.max(jnp.where(valid, pxs, 0.0), axis=1)
        max_py = jnp.max(jnp.where(valid, pys, 0.0), axis=1)

        def compute_one(args):
            px, py, iw, ih, sc, mlw, mlh, s_h, s_v, mih, miw, is_v, jit_x, jit_y = args

            # Horizontal scan: x advances right, y is closed-form
            def x_step(x_prev, j):
                advance = (px[j-1] + iw[j-1] - px[j] + s_h) * sc
                x_new   = x_prev + advance
                x_new   = jnp.where(j % mlw == 0, 0.0, x_new)
                return x_new, x_new + jit_x[j]

            _, x_rest_h = lax.scan(x_step, 0.0, jnp.arange(1, MC))
            x_raw_h     = jnp.concatenate([jnp.array([jit_x[0]]), x_rest_h])
            row         = jnp.arange(MC) // mlw
            y_raw_h     = (mih + s_v) * sc * row.astype(jnp.float32)
            group_h     = row

            # Vertical scan: y advances down, x is closed-form
            def y_step(y_prev, j):
                advance = (py[j-1] + ih[j-1] - py[j] + s_v) * sc
                y_new   = y_prev + advance
                y_new   = jnp.where(j % mlh == 0, 0.0, y_new)
                return y_new, y_new + jit_y[j]

            _, y_rest_v = lax.scan(y_step, 0.0, jnp.arange(1, MC))
            y_raw_v     = jnp.concatenate([jnp.array([jit_y[0]]), y_rest_v])
            col         = jnp.arange(MC) // mlh
            x_raw_v     = (miw + s_h) * sc * col.astype(jnp.float32)
            group_v     = col

            # Select — both scans done, select is near-free
            x_raw = jnp.where(is_v, x_raw_v, x_raw_h)
            y_raw = jnp.where(is_v, y_raw_v, y_raw_h)
            group = jnp.where(is_v, group_v, group_h)

            return x_raw, y_raw, group

        x_raw_all, y_raw_all, group_all = jax.vmap(compute_one)(
            (pxs, pys, iws, ihs, scalings, max_lws, max_lhs,
            sp_h, sp_v, max_ih, max_iw, directions == 1,
            layout_aug.jitter[:, :, 0],
            layout_aug.jitter[:, :, 1])
        )

        x_pos = jnp.where(valid, x_raw_all, -1.0)
        y_pos = jnp.where(valid, y_raw_all, -1.0)

        x_br  = (jnp.max(jnp.where(valid, x_raw_all, 0.0), axis=1)
                 + (max_iw + max_px + sp_h) * scalings)
        y_br  = (jnp.max(jnp.where(valid, y_raw_all, 0.0), axis=1)
                 + (max_ih + max_py + sp_v) * scalings)
        boxes = jnp.stack(
            [jnp.zeros(N), jnp.zeros(N), x_br, y_br], axis=-1
        )

        cx_raw = x_raw_all + (pxs + iws * 0.5) * scalings[:, None]
        cy_raw = y_raw_all + (pys + ihs * 0.5) * scalings[:, None]
        centers = jnp.where(
            valid[:, :, None],
            jnp.stack([cx_raw, cy_raw], axis=-1),
            -1.0,
        )                                                                   # (N, MC, 2)

        same_group = group_all[:, :-1] == group_all[:, 1:]
        both_valid = valid[:, :-1] & valid[:, 1:]
        pair_mask  = same_group & both_valid                               # (N, MC-1)

        pairs_raw = jnp.stack(
            [cx_raw[:, :-1], cy_raw[:, :-1],
             cx_raw[:, 1:],  cy_raw[:, 1:]], axis=-1
        )                                                                   # (N, MC-1, 4)
        pairs = jnp.where(pair_mask[:, :, None], pairs_raw, -1.0)

        rel_geo = RelativeGeo(
            x_pos     = x_pos,
            y_pos     = y_pos,
            centers   = centers,
            pairs     = pairs,
            pair_mask = pair_mask,
        )

        return rel_geo, boxes

    def compute_start_coords(self, boxes, sentence_batch):
        image_ids = np.array(sentence_batch.image_ids)
        boxes_np  = np.array(boxes)
        N         = len(image_ids)

        def intersects(boxA, boxB):
            ax0, ay0, ax1, ay1 = boxA
            bx0, by0, bx1, by1 = boxB
            return not (
                ax1 <= bx0 or
                ax0 >= bx1 or
                ay1 <= by0 or
                ay0 >= by1
            )

        def sample_position(x_rb, y_rb, existing, max_tries=50):
            safe_W = self.W - self.atlas.glyph_width
            safe_H = self.H - self.atlas.glyph_height

            if x_rb >= safe_W or y_rb >= safe_H:
                return 0.0, 0.0, None

            for _ in range(max_tries):
                sx = self.np_rng.uniform(0, safe_W - x_rb)
                sy = self.np_rng.uniform(0, safe_H - y_rb)
                new_box = (sx, sy, sx + x_rb, sy + y_rb)
                if all(not intersects(new_box, b) for b in existing):
                    return sx, sy, new_box

            return 0.0, 0.0, None

        placed_boxes  = {}
        start_coords  = np.zeros((N, 2), dtype=np.float32)
        bubble_params = np.zeros((N, 4), dtype=np.float32)

        for i in range(N):
            image_id = int(image_ids[i])
            if image_id < 0:
                continue

            if image_id not in placed_boxes:
                placed_boxes[image_id] = []

            x_rb = float(boxes_np[i, 2])
            y_rb = float(boxes_np[i, 3])

            sx, sy, new_box = sample_position(
                x_rb, y_rb, placed_boxes[image_id]
            )

            if new_box is None:
                image_ids[i]     = -1
                bubble_params[i] = 0.0
                start_coords[i]  = 0.0
                continue

            bubble_params[i]   = [sx, sy, x_rb, y_rb]
            start_coords[i, 0] = sx
            start_coords[i, 1] = sy
            placed_boxes[image_id].append(new_box)

        sentence_batch = sentence_batch._replace(
            image_ids=jnp.asarray(image_ids)
        )

        return start_coords, bubble_params, sentence_batch

    @staticmethod
    @partial(jit, static_argnames=())
    def apply_coords(rel_geo, start_coords):
        sx = start_coords[:, 0]
        sy = start_coords[:, 1]

        valid_x = rel_geo.x_pos >= 0.0
        valid_y = rel_geo.y_pos >= 0.0

        x_pos = jnp.where(valid_x, rel_geo.x_pos + sx[:, None], -1.0)
        y_pos = jnp.where(valid_y, rel_geo.y_pos + sy[:, None], -1.0)

        valid_c  = rel_geo.centers[:, :, 0] >= 0.0
        offset_c = jnp.stack([sx, sy], axis=-1)[:, None, :]
        centers  = jnp.where(valid_c[:, :, None], rel_geo.centers + offset_c, -1.0)

        valid_p  = rel_geo.pair_mask
        offset_p = jnp.stack([sx, sy, sx, sy], axis=-1)[:, None, :]
        pairs    = jnp.where(valid_p[:, :, None], rel_geo.pairs + offset_p, -1.0)

        return Geo(
            x_pos     = x_pos,
            y_pos     = y_pos,
            centers   = centers,
            pairs     = pairs,
            pair_mask = rel_geo.pair_mask,
        )

    def make_backgrounds(self):
        if self.background_generator is not None:
            backgrounds = self.background_generator.make_background(np_rng=self.np_rng)
        else:
            backgrounds = np.ones((self.N_images, self.H, self.W, self.C), dtype=np.float32)
        return backgrounds

    def draw_ellipse_bubble(self, draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin):
        cx = sx + x_rb / 2
        cy = sy + y_rb / 2

        m = margin
        a = np.sqrt(2) * (x_rb / 2 + m)
        b = np.sqrt(2) * (y_rb / 2 + m)

        bbox = [cx - a, cy - b, cx + a, cy + b]
        draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=outline_w)

    def draw_jagged_bubble(self, draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin, cfg, np_rng):
        cx = sx + x_rb / 2
        cy = sy + y_rb / 2

        n_spikes    = int(np_rng.integers(cfg.jagged_n_spikes[0], cfg.jagged_n_spikes[1] + 1))
        spike_ratio = np_rng.uniform(cfg.jagged_spike_ratio[0], cfg.jagged_spike_ratio[1])

        # Outer ellipse (spike tips), inner ellipse (spike bases)
        a_outer = np.sqrt(2) * (x_rb / 2 + margin)
        b_outer = np.sqrt(2) * (y_rb / 2 + margin)
        a_inner = a_outer * spike_ratio
        b_inner = b_outer * spike_ratio

        # 2*n_spikes points alternating between outer (tips) and inner (base)
        n_points = 2 * n_spikes
        angles   = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        points = []
        for i, angle in enumerate(angles):
            if i % 2 == 0:
                # Spike tip on outer ellipse
                r_x = a_outer * np.cos(angle)
                r_y = b_outer * np.sin(angle)
            else:
                # Spike base on inner ellipse
                r_x = a_inner * np.cos(angle)
                r_y = b_inner * np.sin(angle)
            points.append((cx + r_x, cy + r_y))

        draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_w)

    def draw_rectangle_bubble(self, draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin):
        cx = sx + x_rb / 2
        cy = sy + y_rb / 2

        a = x_rb / 2 + margin
        b = y_rb / 2 + margin

        bbox = [cx - a, cy - b, cx + a, cy + b]
        draw.rectangle(bbox, fill=fill_color, outline=outline_color, width=outline_w)

    def draw_wavy_bubble(self, draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin, cfg, np_rng):
        cx = sx + x_rb / 2
        cy = sy + y_rb / 2

        amplitude = np_rng.uniform(cfg.wavy_amplitude[0], cfg.wavy_amplitude[1])
        frequency = int(np_rng.integers(cfg.wavy_frequency[0], cfg.wavy_frequency[1] + 1))

        a_base = np.sqrt(2) * (x_rb / 2 + margin)
        b_base = np.sqrt(2) * (y_rb / 2 + margin)

        # Sample enough points to make the wave smooth
        n_points = frequency * 20
        angles   = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Wavy radius: base ellipse radius modulated by a sine wave
        # Ellipse radius at angle theta: r(theta) = 1 / sqrt((cos/a)^2 + (sin/b)^2)
        # We modulate outward from this by amplitude * sin(frequency * theta)
        r_ellipse = 1.0 / np.sqrt((np.cos(angles) / a_base)**2 + (np.sin(angles) / b_base)**2)
        r_wavy    = r_ellipse + amplitude * np.sin(frequency * angles)

        points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(r_wavy, angles)]
        draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_w)

    def make_speech_bubbles(self, bubble_params, sentence_batch, np_rng, backgrounds):
        cfg        = self.speech_bubble_config
        tc         = self.text_config
        config_idx = self.config_idx

        bg_pil = []
        for img_idx in range(self.N_images):
            arr = (np.clip(backgrounds[img_idx], 0, 1) * 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                pil_img = PILImage.fromarray(arr[..., 0], mode="L").convert("RGB")
            else:
                pil_img = PILImage.fromarray(arr, mode="RGB")
            bg_pil.append(pil_img)

        for s_idx in range(self.N_unique_sentences):
            img_idx = int(sentence_batch.image_ids[s_idx])
            if img_idx < 0 or img_idx >= self.N_images:
                continue
            t_idx = int(config_idx[s_idx])
            probs = np.array(tc[t_idx].speech_bubble_probs, dtype=np.float32)
            shape = int(self.np_rng.choice(len(probs), p=probs))

            if shape == NONE:
                continue

            outline_w     = int(self.np_rng.integers(cfg.outline_width[0], cfg.outline_width[1] + 1))
            fill_v        = self.np_rng.uniform(cfg.fill_value[0], cfg.fill_value[1])
            fill_color    = tuple([int(fill_v * 255)] * 3)
            outline_color = (0, 0, 0)
            sx, sy, x_rb, y_rb = bubble_params[s_idx]
            margin = self.np_rng.uniform(cfg.ellipse_margin[0], cfg.ellipse_margin[1])
            draw   = ImageDraw.Draw(bg_pil[img_idx])

            if shape == ELLIPSE:
                self.draw_ellipse_bubble(draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin)

            elif shape == JAGGED:
                self.draw_jagged_bubble(draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin, cfg, np_rng)

            elif shape == RECTANGLE:
                self.draw_rectangle_bubble(draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin)

            elif shape == WAVY:
                self.draw_wavy_bubble(draw, sx, sy, x_rb, y_rb, fill_color, outline_color, outline_w, margin, cfg, np_rng)

        # Convert back
        for img_idx in range(self.N_images):
            arr = np.array(bg_pil[img_idx])
            if self.C == 1:
                arr = arr[..., :1]
            backgrounds[img_idx] = arr.astype(np.float32) / 255.0

        return backgrounds

    @staticmethod
    @partial(jit, static_argnames=("N_images", "H", "W", "glyph_h", "glyph_w", "border_r"))
    def make_images_jitted(
        geos, sentence_batch, render_aug, backgrounds,
        N_images, H, W,
        glyphs,
        glyph_h, glyph_w,
        border_r,
    ):
        safe_fidx = jnp.clip(sentence_batch.font_ids,  0, glyphs.shape[0] - 1)
        safe_cidx = jnp.clip(sentence_batch.sentences, 0, glyphs.shape[1] - 1)
        row_idx   = jnp.arange(glyph_h)
        col_idx   = jnp.arange(glyph_w)

        def make_single_image(image_idx):
            canvas = backgrounds[image_idx]

            def render_sentence(canvas, s_idx):
                belongs = (sentence_batch.image_ids[s_idx] == image_idx)
                scaling = render_aug.scaling[s_idx]
                fid     = safe_fidx[s_idx]

                rh = jnp.maximum(jnp.int32(jnp.round(scaling * glyph_h)), 1)
                rw = jnp.maximum(jnp.int32(jnp.round(scaling * glyph_w)), 1)
                src_rows = jnp.clip(jnp.int32(row_idx / scaling), 0, glyph_h - 1)
                src_cols = jnp.clip(jnp.int32(col_idx / scaling), 0, glyph_w - 1)
                active   = (row_idx[:, None] < rh) & (col_idx[None, :] < rw)

                def render_char(canvas, i):
                    char_valid   = (geos.x_pos[s_idx, i] >= 0.0) & belongs
                    cidx         = safe_cidx[s_idx, i]
                    glyph_full   = glyphs[fid, cidx].astype(jnp.float32)
                    glyph_remap  = glyph_full[src_rows[:, None], src_cols[None, :]]
                    glyph_masked = jnp.where(active, glyph_remap, 1.0)[..., None]

                    x = jnp.clip(jnp.int32(geos.x_pos[s_idx, i]), 0, W - glyph_w)
                    y = jnp.clip(jnp.int32(geos.y_pos[s_idx, i]), 0, H - glyph_h)

                    existing = lax.dynamic_slice(canvas, (y, x, 0), (glyph_h, glyph_w, 1))

                    # ── white border ──
                    ink_mask = (glyph_masked[..., 0] < 1.0).astype(jnp.float32)
                    h_dil = lax.reduce_window(
                        ink_mask[None, None],
                        init_value=0.0, computation=lax.max,
                        window_dimensions=(1, 1, 1, 2 * border_r + 1),
                        window_strides=(1, 1, 1, 1),
                        padding=((0,0),(0,0),(0,0),(border_r, border_r)),
                    )[0, 0]
                    dilated = lax.reduce_window(
                        h_dil[None, None],
                        init_value=0.0, computation=lax.max,
                        window_dimensions=(1, 1, 2 * border_r + 1, 1),
                        window_strides=(1, 1, 1, 1),
                        padding=((0,0),(0,0),(border_r, border_r),(0,0)),
                    )[0, 0, ..., None]
                    white_board = jnp.where(dilated > 0.0, 1.0, existing)
                    canvas = lax.dynamic_update_slice(
                        canvas,
                        jnp.where(char_valid, white_board, existing),
                        (y, x, 0),
                    )

                    # ── ink ──
                    patch     = lax.dynamic_slice(canvas, (y, x, 0), (glyph_h, glyph_w, 1))
                    new_patch = jnp.minimum(patch, glyph_masked)
                    canvas    = lax.dynamic_update_slice(
                        canvas,
                        jnp.where(char_valid, new_patch, patch),
                        (y, x, 0),
                    )
                    return canvas, None

                canvas, _ = lax.scan(render_char, canvas, jnp.arange(geos.x_pos.shape[1]))
                return canvas, None

            canvas, _ = lax.scan(render_sentence, canvas, jnp.arange(sentence_batch.image_ids.shape[0]))
            return canvas

        return jax.vmap(make_single_image)(jnp.arange(N_images))

    # ── Kernels and Heatmaps ──────────────────────────────────────────────────
    @staticmethod
    def make_kernels(sigma, radius, intensity, max_kernel_radius):
        ks = 2 * max_kernel_radius + 1
        grid = np.arange(ks) - max_kernel_radius
        GX, GY = np.meshgrid(grid, grid)
        dist_sq = GX**2 + GY**2

        dist_sq   = dist_sq[None, :, :]    # (1, ks, ks)
        sigma     = sigma[:, None, None]
        radius    = radius[:, None, None]
        intensity = intensity[:, None, None]

        g       = intensity * np.exp(-dist_sq / (2 * sigma**2))
        kernels = np.where(dist_sq <= radius**2, g, 0.0)
        return jnp.array(kernels)

    @staticmethod
    @partial(jit, static_argnames=("N_images", "H", "W"))
    def make_char_heatmaps_jitted(geos, sentence_batch, char_aug, N_images, H, W):
        ks            = char_aug.kernel.shape[1]
        kernel_radius = (ks - 1) // 2
        pad           = kernel_radius

        def make_single_heatmap(image_idx):
            heatmap = jnp.zeros((H + 2 * pad, W + 2 * pad), dtype=jnp.float32)

            def render_sentence(heatmap, s_idx):
                belongs = (sentence_batch.image_ids[s_idx] == image_idx)
                kernel  = char_aug.kernel[s_idx]

                def render_char(heatmap, i):
                    char_valid = (geos.centers[s_idx, i, 0] >= 0.0) & belongs

                    cx = jnp.int32(jnp.round(geos.centers[s_idx, i, 0])) + pad
                    cy = jnp.int32(jnp.round(geos.centers[s_idx, i, 1])) + pad

                    ox = cx - kernel_radius
                    oy = cy - kernel_radius

                    patch     = lax.dynamic_slice(heatmap, (oy, ox), (ks, ks))
                    new_patch = jnp.maximum(patch, kernel)

                    heatmap = lax.dynamic_update_slice(
                        heatmap,
                        jnp.where(char_valid, new_patch, patch),
                        (oy, ox),
                    )
                    return heatmap, None

                heatmap, _ = lax.scan(render_char, heatmap, jnp.arange(geos.centers.shape[1]))
                return heatmap, None

            heatmap, _ = lax.scan(render_sentence, heatmap, jnp.arange(sentence_batch.image_ids.shape[0]))
            return heatmap[pad:pad + H, pad:pad + W]

        return jax.vmap(make_single_heatmap)(jnp.arange(N_images))

    @staticmethod
    @partial(jit, static_argnames=("N_images", "H", "W", "n_steps"))
    def make_affinity_heatmaps_jitted(
        geos,
        sentence_batch,
        affin_aug,
        N_images, H, W,
        n_steps,
    ):
        ks            = affin_aug.kernel.shape[1]
        kernel_radius = (ks - 1) // 2
        pad           = kernel_radius
        ts            = jnp.linspace(0.0, 1.0, n_steps + 2)

        def make_single_heatmap(image_idx):
            heatmap = jnp.zeros((H + 2 * pad, W + 2 * pad), dtype=jnp.float32)

            def render_sentence(heatmap, s_idx):
                belongs = (sentence_batch.image_ids[s_idx] == image_idx)
                kernel  = affin_aug.kernel[s_idx]

                def render_pair(heatmap, i):
                    valid = geos.pair_mask[s_idx, i] & belongs
                    x1 = geos.pairs[s_idx, i, 0]
                    y1 = geos.pairs[s_idx, i, 1]
                    x2 = geos.pairs[s_idx, i, 2]
                    y2 = geos.pairs[s_idx, i, 3]

                    dx   = x2 - x1
                    dy   = y2 - y1
                    dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)
                    margin = jnp.clip(jnp.float32(kernel_radius) / dist, 0.0, 0.49)

                    ax1 = x1 * (1.0 - margin) + x2 * margin
                    ay1 = y1 * (1.0 - margin) + y2 * margin
                    ax2 = x1 * margin + x2 * (1.0 - margin)
                    ay2 = y1 * margin + y2 * (1.0 - margin)

                    def render_t(heatmap, t):
                        mx = jnp.int32(jnp.round(ax1 * (1.0 - t) + ax2 * t)) + pad
                        my = jnp.int32(jnp.round(ay1 * (1.0 - t) + ay2 * t)) + pad

                        ox = mx - kernel_radius
                        oy = my - kernel_radius

                        patch     = lax.dynamic_slice(heatmap, (oy, ox), (ks, ks))
                        new_patch = jnp.maximum(patch, kernel)

                        heatmap = lax.dynamic_update_slice(
                            heatmap,
                            jnp.where(valid, new_patch, patch),
                            (oy, ox),
                        )
                        return heatmap, None

                    heatmap, _ = lax.scan(render_t, heatmap, ts)
                    return heatmap, None

                heatmap, _ = lax.scan(render_pair, heatmap, jnp.arange(geos.pair_mask.shape[1]))
                return heatmap, None

            heatmap, _ = lax.scan(render_sentence, heatmap, jnp.arange(sentence_batch.image_ids.shape[0]))
            return heatmap[pad:pad + H, pad:pad + W]

        return jax.vmap(make_single_heatmap)(jnp.arange(N_images))

    @staticmethod
    @jit
    def image_augmentations_jax(images, image_aug, jax_rng):
        jax_rng, subkey = random.split(jax_rng)
        keys        = random.split(subkey, images.shape[0])
        noise_sigmas = image_aug.noise_sigma
        brightnesses = image_aug.brightness
        colour_inversions = image_aug.colour_inversion

        def augment_image(image, noise_sigma, brightness, colour_inversion, key):
            noise = jax.random.normal(key, image.shape)
            image = jnp.where(noise_sigma > 0, image + noise_sigma * noise, image)
            image = jnp.where(brightness > 0, image * brightness, image)
            image = jnp.where(colour_inversion > 0, 1.0 - image, image)
            return jnp.clip(image, 0.0, 1.0)

        images = jax.vmap(augment_image)(images, noise_sigmas, brightnesses, colour_inversions, keys)
        return images, jax_rng

    def image_augmentations_numpy(self, images, targets):
        jpeg_qualities = self.jpeg_quality

        for i in range(self.N_images):
            if self.np_rng.random() < self.prob_jpeg:
                q       = float(jpeg_qualities[i])
                quality = int(np.clip(q, 1, 95))
                arr     = images[i]
                arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

                if self.C == 1:
                    pil_img = PILImage.fromarray(arr_uint8[..., 0], mode="L")
                else:
                    pil_img = PILImage.fromarray(arr_uint8, mode="RGB")

                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                pil_img_decoded = PILImage.open(buf).convert("L" if self.C == 1 else "RGB")
                decoded = np.array(pil_img_decoded, dtype=np.float32) / 255.0

                if self.C == 1:
                    decoded = decoded[..., None]

                images[i] = decoded

            if self.np_rng.random() < self.prob_x_flip:
                images[i]  = images[i,  :, ::-1, :]
                targets[i] = targets[i, :, ::-1, :]

            if self.np_rng.random() < self.prob_y_flip:
                images[i]  = images[i,  ::-1, :, :]
                targets[i] = targets[i,  ::-1, :, :]
        return images, targets

    def crop_augmentation(self, image, targets):
        crop_H = int(self.crop_scale * self.H)
        crop_W = int(self.crop_scale * self.W)

        y0 = self.np_rng.integers(0, self.H - crop_H + 1)
        x0 = self.np_rng.integers(0, self.W - crop_W + 1)    
        img_crop    = image[y0:y0+crop_H, x0:x0+crop_W, 0]
        pil_img     = PILImage.fromarray((img_crop * 255).astype(np.uint8), mode="L")
        pil_img     = pil_img.resize((self.W, self.H), PILImage.LANCZOS)
        real_image  = np.array(pil_img, dtype=np.float32)[..., None] / 255.0

        channels = []
        for ch in range(targets.shape[-1]):
            pil_ch   = PILImage.fromarray(targets[y0:y0+crop_H, x0:x0+crop_W, ch], mode="F")
            pil_ch   = pil_ch.resize((self.W, self.H), PILImage.BILINEAR)
            channels.append(np.array(pil_ch, dtype=np.float32))
        real_targets = np.stack(channels, axis=-1)

        return real_image, real_targets
    
    def generate_batch(self, rng):
        self.seed   = rng
        self.np_rng = np.random.default_rng(self.seed)
        jax_rng     = self.jax_rng

        sentence_batch, layout_aug, render_aug, char_aug, affin_aug, image_aug = self.sample_batch()

        rel_geo, boxes = SyntheticDataGenerator.compute_relative_geometries(
            sentence_batch, layout_aug,
            self.ink_offset_arr,
            self.max_char, self.N_unique_sentences,
        )

        start_coords, bubble_params, sentence_batch = self.compute_start_coords(
            boxes, sentence_batch
        )

        geos = SyntheticDataGenerator.apply_coords(
            rel_geo, jnp.array(start_coords)
        )

        backgrounds = self.make_backgrounds()

        if self.speech_bubble:
            backgrounds = self.make_speech_bubbles(
                bubble_params, sentence_batch, self.np_rng, backgrounds
            )

        backgrounds = jnp.array(backgrounds)

        border_r = int(render_aug.stroke.max())

        images = SyntheticDataGenerator.make_images_jitted(
            geos, sentence_batch, render_aug, backgrounds,
            self.N_images, self.H, self.W,
            self.glyphs,
            self.atlas.glyph_height, self.atlas.glyph_width,
            border_r=border_r,
        )

        char_heat = SyntheticDataGenerator.make_char_heatmaps_jitted(
            geos, sentence_batch, char_aug,
            self.N_images, self.H, self.W,
        )

        aff_heat = SyntheticDataGenerator.make_affinity_heatmaps_jitted(
            geos, sentence_batch, affin_aug,
            self.N_images, self.H, self.W,
            self.n_steps,
        )

        targets = np.stack(
            [np.array(char_heat), np.array(aff_heat)],
            axis=-1
        ).astype(np.float32)

        images = np.array(images, dtype=np.float32)

        if self.real_data_config.prob_real_data != 0.0:
            for i in range(self.N_images):
                if self.np_rng.random() >= self.real_data_config.prob_real_data:
                    continue
                idx = self.np_rng.integers(0, len(self.real_data))
                real_image, real_targets = self.real_data[idx]
                real_image, real_targets = self.crop_augmentation(real_image, real_targets)
                images[i]   = real_image
                targets[i]  = real_targets

        if self.image_aug:
            images, jax_rng = SyntheticDataGenerator.image_augmentations_jax(
                images, image_aug, jax_rng
            )

        images = np.array(images, dtype=np.float32)

        if self.image_aug:
            images, targets = self.image_augmentations_numpy(images, targets)

        self.jax_rng = jax_rng

        return images, targets

    @staticmethod
    def heatmap_decoder(x_sample, y_heatmap, ax):
        char_heatmap     = np.array(y_heatmap[:, :, 0])
        affinity_heatmap = np.array(y_heatmap[:, :, 1])

        text_mask = np.logical_or(char_heatmap , affinity_heatmap).astype(np.uint8)
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons    = [cv2.approxPolyDP(c, epsilon=2.0, closed=True) for c in contours]
 
        ax[0].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
        for poly in polygons:
            pts = poly.reshape(-1, 2)
            ax[0].plot(
                np.append(pts[:, 0], pts[0, 0]),
                np.append(pts[:, 1], pts[0, 1]),
                'r-', linewidth=2,
            )
        ax[0].set_title("Image with bounding boxes")
        ax[0].axis("off")

    def plot_heatmaps(self, images, targets, idx=0):
        img       = np.array(images[idx])
        char_heat = np.array(targets[idx, :, :, 0])
        aff_heat  = np.array(targets[idx, :, :, 1])
        char_heat /= (char_heat.max() + 1e-8)
        aff_heat  /= (aff_heat.max()  + 1e-8)

        fig, ax = plt.subplots(1, 3, figsize=(14, 9))

        self.heatmap_decoder(img, targets[idx, :, :, :], ax)

        ax[1].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[1].imshow(char_heat, cmap="jet", alpha=0.6)
        ax[1].set_title("Character Heatmap")
        ax[1].axis("off")

        ax[2].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[2].imshow(aff_heat, cmap="hot", alpha=0.6)
        ax[2].set_title("Affinity Heatmap")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

    def benchmark(self, n_rounds=10, n_warmups=5, rng=0, device="cpu"):
        self.seed   = rng
        self.np_rng = np.random.default_rng(self.seed)
        jax_rng     = self.jax_rng

        print(f"\n{'='*60}")
        print(f"{'Benchmark: Generator':^60}")
        print(f"{'='*60}")
        print(f"  rounds        : {n_rounds}")
        print(f"  batch size    : {self.N_images}")
        print(f"  n warmups     : {n_warmups}")
        print(f"  device        : {device}")
        print(f"{'─'*60}")

        print(f"{f'Pre jitting {n_warmups} times':^60}")
        t_warmup_start = time.perf_counter()
        for _ in range(n_warmups):
            self.warmup(i=None)
        t_warmup_end = time.perf_counter()
        print(f"Finished pre jit in {t_warmup_end - t_warmup_start:.2f}s")
        print(f"{'─'*60}")
        print(f"  {'round':>5}  {'total(s)':>9}  {'slowest':>22}  {'time(s)':>9}")
        print(f"{'─'*60}")

        benchmark_times = {
            "sample_batch":        [],
            "rel_geo":             [],
            "start_coords":        [],
            "apply_coords":        [],
            "backgrounds":         [],
            "speech_bubbles":      [],
            "to_jax_backgrounds":  [],
            "render_images":       [],
            "image_aug_jax":       [],
            "to_numpy":            [],
            "real_data_mix":       [],
            "image_aug_numpy":     [],
            "char_heat":           [],
            "aff_heat":            [],
            "stack_targets":       [],
            "total":               [],
        }
        start_time = time.perf_counter()
        for i in range(n_rounds):

            t0 = time.perf_counter()
            sentence_batch, layout_aug, render_aug, char_aug, affin_aug, image_aug = self.sample_batch()
            t1 = time.perf_counter()
            benchmark_times["sample_batch"].append(t1 - t0)

            rel_geo, boxes = SyntheticDataGenerator.compute_relative_geometries(
                sentence_batch, layout_aug,
                self.ink_offset_arr,
                self.max_char,
                self.N_unique_sentences,
            )
            jax.block_until_ready(rel_geo)
            jax.block_until_ready(boxes)
            t2 = time.perf_counter()
            benchmark_times["rel_geo"].append(t2 - t1)

            start_coords, bubble_params, sentence_batch = self.compute_start_coords(
                boxes, sentence_batch
            )
            t3 = time.perf_counter()
            benchmark_times["start_coords"].append(t3 - t2)

            geos = SyntheticDataGenerator.apply_coords(
                rel_geo, jnp.array(start_coords)
            )
            jax.block_until_ready(geos)
            t4 = time.perf_counter()
            benchmark_times["apply_coords"].append(t4 - t3)

            backgrounds = self.make_backgrounds()
            t5 = time.perf_counter()
            benchmark_times["backgrounds"].append(t5 - t4)

            if self.speech_bubble:
                backgrounds = self.make_speech_bubbles(
                    bubble_params, sentence_batch, self.np_rng, backgrounds
                )
            t6 = time.perf_counter()
            benchmark_times["speech_bubbles"].append(t6 - t5)

            backgrounds = jnp.array(backgrounds)
            t7 = time.perf_counter()
            benchmark_times["to_jax_backgrounds"].append(t7 - t6)

            border_r = int(render_aug.stroke.max())

            images = SyntheticDataGenerator.make_images_jitted(
                geos, sentence_batch, render_aug, backgrounds,
                self.N_images, self.H, self.W,
                self.glyphs,
                self.atlas.glyph_height,
                self.atlas.glyph_width,
                border_r=border_r,
            )
            jax.block_until_ready(images)
            t8 = time.perf_counter()
            benchmark_times["render_images"].append(t8 - t7)

            if self.image_aug:
                images, jax_rng = SyntheticDataGenerator.image_augmentations_jax(
                    images, image_aug, jax_rng
                )
                jax.block_until_ready(images)
                jax.block_until_ready(jax_rng)
            t9 = time.perf_counter()
            benchmark_times["image_aug_jax"].append(t9 - t8)

            images = np.array(images, dtype=np.float32)
            t10 = time.perf_counter()
            benchmark_times["to_numpy"].append(t10 - t9)

            char_heat = SyntheticDataGenerator.make_char_heatmaps_jitted(
                geos, sentence_batch, char_aug,
                self.N_images, self.H, self.W,
            )
            jax.block_until_ready(char_heat)
            t11 = time.perf_counter()
            benchmark_times["char_heat"].append(t11 - t10)

            aff_heat = SyntheticDataGenerator.make_affinity_heatmaps_jitted(
                geos, sentence_batch, affin_aug,
                self.N_images, self.H, self.W,
                self.n_steps,
            )
            jax.block_until_ready(aff_heat)
            t12 = time.perf_counter()
            benchmark_times["aff_heat"].append(t12 - t11)

            targets = np.stack(
                [np.array(char_heat), np.array(aff_heat)],
                axis=-1
            ).astype(np.float32)
            t13 = time.perf_counter()
            benchmark_times["stack_targets"].append(t13 - t12)

            if self.real_data_config.prob_real_data != 0.0:
                for j in range(self.N_images):
                    if self.np_rng.random() >= self.real_data_config.prob_real_data:
                        continue
                    idx = self.np_rng.integers(0, len(self.real_data))
                    real_image, real_targets = self.real_data[idx]
                    real_image, real_targets = self.crop_augmentation(real_image, real_targets)
                    images[j]  = real_image
                    targets[j] = real_targets
            t14 = time.perf_counter()
            benchmark_times["real_data_mix"].append(t14 - t13)

            if self.image_aug:
                images, targets = self.image_augmentations_numpy(images, targets)
            t15 = time.perf_counter()
            benchmark_times["image_aug_numpy"].append(t15 - t14)

            benchmark_times["total"].append(t15 - t0)
            self.jax_rng = jax_rng
            round_times  = {k: benchmark_times[k][i] for k in benchmark_times if k != "total"}
            slowest_key  = max(round_times, key=round_times.get)
            slowest_val  = round_times[slowest_key]
            print(f"  {i+1:>5}  {(t15-t0):>9.4f}  {slowest_key:>22}  {slowest_val:>9.4f}")

        end_time = time.perf_counter()
        print(f"{'─'*60}")
        print(f"Finished {n_rounds} rounds of generating data in {(end_time - start_time):.2f}s")
        print(f"{'─'*60}")
        print(f"{'Averages and stds':^60}")
        print(f"{'─'*60}")
        for k, v in benchmark_times.items():
            arr = np.array(v)
            print(f"{k:>20}: {arr.mean():.4f}s ± {arr.std():.4f}s")
        print(f"{'─'*60}")

if __name__ == "__main__":
    from configs.default import CFG_TRAIN
    config = CFG_TRAIN
    print("Start initialization")
    gen = SyntheticDataGenerator(config, workers_init=False, N_images = 4)
    print("Finished initialization")
    images, targets = gen.generate_batch(rng=0)
    for i in range(gen.N_images):
        gen.plot_heatmaps(images, targets, idx=i)
    
    #gen.benchmark()