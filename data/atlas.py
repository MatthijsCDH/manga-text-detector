import os
import random
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

# ── Paths ─────────────────────────────────────────────────────────────────────
font_dir   = os.path.join(os.path.dirname(__file__), "assets", "Fonts")
kanji_path = os.path.join(os.path.dirname(__file__), "assets", "Kanji", "jouyou_kanji.txt")


class Atlas:
    def __init__(self, atlas_config, benchmark=False):
        self.glyph_size           = atlas_config.glyph_size
        self.max_scaling_sentence = atlas_config.max_scaling_sentence
        self.glyph_width          = self.glyph_size * (self.max_scaling_sentence + 1)
        self.glyph_height         = self.glyph_size * (self.max_scaling_sentence + 1)

        self.cache_dir  = os.path.join(os.path.dirname(__file__), "..", ".cache", "atlas")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_key  = f"gs{self.glyph_size}_ms{self.max_scaling_sentence}"
        self.cache_path = os.path.join(self.cache_dir, f"{self.cache_key}.npz")

        if not benchmark:
            self.load_fonts()
            self.load_characters()
            if not self.load_cache():
                self.build_glyphs()
                self.save_cache()

    def load_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        data = np.load(self.cache_path)

        self.glyphs           = data["glyphs"]
        self.ink_offsets      = data["ink_offsets"]
        self.max_glyph_width  = data["max_glyph_width"].item()
        self.max_glyph_height = data["max_glyph_height"].item()

        return True

    def save_cache(self):
        print(f"Saving atlas cache: {self.cache_path}")
        np.savez_compressed(
            self.cache_path,
            glyphs=self.glyphs,
            ink_offsets=self.ink_offsets,
            max_glyph_width=self.max_glyph_width,
            max_glyph_height=self.max_glyph_height,
        )

    def load_fonts(self):
        font_paths = [
            os.path.join(font_dir, "Mplus1-Regular.ttf"),          # Font ID: 0
            os.path.join(font_dir, "Mplus1-Bold.ttf"),             # Font ID: 1
            os.path.join(font_dir, "ipagp.ttf"),                   # Font ID: 2
            os.path.join(font_dir, "NotoSansCJK-Regular.ttc"),     # Font ID: 3
            os.path.join(font_dir, "mplus-1p-thin.ttf"),           # Font ID: 4
            os.path.join(font_dir, "mplus-1p-heavy.ttf"),          # Font ID: 5
            os.path.join(font_dir, "mplus-1p-black.ttf"),          # Font ID: 6
            os.path.join(font_dir, "mplus-2p-bold.ttf"),           # Font ID: 7
            os.path.join(font_dir, "mplus-2p-heavy.ttf"),          # Font ID: 8
            os.path.join(font_dir, "ZenAntique-Regular.ttf"),      # Font ID: 9
            os.path.join(font_dir, "ZenAntiqueSoft-Regular.ttf"),  # Font ID: 10
        ]
        self.fonts     = []
        self.font_data = []
        for font_path in font_paths:
            pil_font = ImageFont.truetype(font_path, self.glyph_size)
            tt       = TTFont(font_path, fontNumber=0)
            cmap     = tt.getBestCmap()
            hmtx     = tt["hmtx"].metrics
            units    = tt["head"].unitsPerEm
            scale    = self.glyph_size / units

            self.fonts.append(pil_font)
            self.font_data.append((pil_font, cmap, hmtx, scale))

        self.num_fonts = len(self.fonts)

    def load_characters(self):
        hiragana     = [chr(i) for i in range(0x3041, 0x3097)]
        katakana     = [chr(i) for i in range(0x30A1, 0x30FF)]
        with open(kanji_path, encoding="utf-8") as f:
            kanji    = [line.strip() for line in f if line.strip()]

        latin        = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        numbers      = list("0123456789")
        jp_symbols   = list("。、！‼ ？ー「」『』（）［］…・々〃喰 噂 ← 殲 掻 塵 狼 仇 掴 掴 躊 躇 絆 絆 怯 渾 眩 怯 舐")
        ascii_symbols = list(".,!?-:;()[]{}\"'")

        all_chars = hiragana + katakana + kanji + latin + numbers + jp_symbols + ascii_symbols
        all_chars = sorted(set(all_chars))

        self.char_list   = all_chars
        self.char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
        self.num_chars   = len(all_chars)

    def build_glyphs(self):
        self.glyphs = np.zeros(
            (self.num_fonts, self.num_chars, self.glyph_height, self.glyph_width),
            dtype=np.float16,
        )
        self.ink_offsets = np.zeros((self.num_fonts, self.num_chars, 4), dtype=np.float32)

        for font_id, pil_font in enumerate(self.fonts):
            for ch_id, ch in enumerate(self.char_list):
                glyph, offset = self.build_glyph_for_font(ch, pil_font)
                self.glyphs[font_id, ch_id]      = glyph
                self.ink_offsets[font_id, ch_id] = offset

        self.max_glyph_width  = np.max(self.ink_offsets[:, :, 2])
        self.max_glyph_height = np.max(self.ink_offsets[:, :, 3])

    def build_glyph_for_font(self, ch, pil_font):
        G = self.glyph_width
        H = self.glyph_height

        canvas_size = G * 6
        canvas = Image.fromarray(np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255)
        draw   = ImageDraw.Draw(canvas)

        ox, oy = G * 4, H * 4
        draw.text((ox, oy), ch, font=pil_font, fill=0)

        left, top, right, bottom = draw.textbbox((ox, oy), ch, font=pil_font)

        w, h = right - left, bottom - top

        cell = np.ones((H, G), dtype=np.float32)

        if w <= 0 or h <= 0:
            return cell, np.array([0, 0, 0, 0], dtype=np.float32)

        glyph = np.array(canvas, dtype=np.float32)[top:bottom, left:right] / 255.0

        true_origin_x = left - ox
        true_origin_y = top  - oy

        sx = max(0, -true_origin_x)
        sy = max(0, -true_origin_y)
        dx = max(0,  true_origin_x)
        dy = max(0,  true_origin_y)

        cw  = min(w - sx, G - dx)
        ch_ = min(h - sy, H - dy)

        if cw > 0 and ch_ > 0:
            cell[dy:dy + ch_, dx:dx + cw] = glyph[sy:sy + ch_, sx:sx + cw]

        offset = np.array([true_origin_x, true_origin_y, w, h], dtype=np.float32)

        return cell, offset

    def glyph_properties(self, ch_id=None, font_id=None):
        if font_id is None:
            font_id = random.randint(0, self.num_fonts - 1)
        if ch_id is None:
            ch_id = random.randint(0, self.num_chars - 1)

        ch    = self.char_list[ch_id]
        glyph = self.glyphs[font_id, ch_id]
        px, py, iw, ih = self.ink_offsets[font_id, ch_id]

        print(f"\n{'='*60}")
        print(f"{'Glyph properties':^60}")
        print(f"{'='*60}")
        print(f"Font ID                     : {font_id}")
        print(f"Character ID                : {ch_id}")
        print(f"Character                   : '{ch}' (U+{ord(ch):04X})")
        print(f"Ink offset                  : {px:.2f}, {py:.2f}")
        print(f"Ink size                    : {iw:.2f} x {ih:.2f}")
        print(f"Glyph min/max               : {glyph.min():.3f} / {glyph.max():.3f}")
        print(f"Glyphs shape                : {self.glyphs.shape}")
        print(f"Total chars                 : {len(self.char_list)}")
        print(f"max_width                   : {self.max_glyph_width}")
        print(f"max_height                  : {self.max_glyph_height}")
        print(f"{'─'*60}")

        plt.figure(figsize=(4, 4))
        plt.imshow(glyph, cmap="gray", vmin=0, vmax=1)
        plt.title(f"font={font_id} idx={ch_id}")

        if iw > 0 and ih > 0:
            rect = plt.Rectangle(
                (max(px - 0.5, 0), max(py - 0.5, 0)), iw, ih,
                edgecolor="red", facecolor="none", linewidth=1,
            )
            plt.gca().add_patch(rect)

        plt.axis("off")
        plt.show()

    def benchmark(self, atlas_config, n_rounds=10):
        self.fonts     = []
        self.font_data = []

        print(f"\n{'='*60}")
        print(f"{'Benchmark: Atlas':^60}")
        print(f"{'='*60}")
        print(f"  rounds                      : {n_rounds}")
        print(f"  glyph size                  : {atlas_config.glyph_size}")
        print(f"  max scaling per sentence    : {atlas_config.max_scaling_sentence}")
        print(f"{'─'*60}")
        print(f"  {'round':>2} {'cache(s)':>10}  {'load_fonts(s)':>13}  {'load_char(s)':>12}  {'build_glyphs(s)':>15}")
        print(f"{'─'*60}")

        benchmark_times = {
            "cache":        [],
            "load_fonts":   [],
            "load_char":    [],
            "build_glyphs": [],
        }

        def stats(arr):
            arr = np.array(arr)
            return arr.mean(), arr.std()

        start_time = time.perf_counter()
        for i in range(n_rounds):
            atlas = Atlas.__new__(Atlas)

            t0 = time.perf_counter()
            atlas.__init__(atlas_config, benchmark=True)
            atlas.load_cache()
            t1 = time.perf_counter()
            benchmark_times["cache"].append(t1 - t0)

            atlas.load_fonts()
            t2 = time.perf_counter()
            benchmark_times["load_fonts"].append(t2 - t1)

            atlas.load_characters()
            t3 = time.perf_counter()
            benchmark_times["load_char"].append(t3 - t2)

            atlas.build_glyphs()
            t4 = time.perf_counter()
            benchmark_times["build_glyphs"].append(t4 - t3)

            print(f"  {i+1:>4}  {(t1-t0):>10.4f}  {(t2-t1):>13.4f}  {(t3-t2):>12.4f}  {(t4-t3):>15.4f}")

        end_time = time.perf_counter()
        print(f"{'─'*60}")
        print(f"Finished {n_rounds} rounds of benchmarking in {(end_time - start_time):.2f}s")
        print(f"{'─'*60}")
        print(f"{'Averages and stds':^60}")
        print(f"{'─'*60}")

        mean_c,  std_c  = stats(benchmark_times["cache"])
        mean_lf, std_lf = stats(benchmark_times["load_fonts"])
        mean_lc, std_lc = stats(benchmark_times["load_char"])
        mean_bg, std_bg = stats(benchmark_times["build_glyphs"])

        print(f"cache        : {mean_c:>6.3f}s ± {std_c:.3f}s")
        print(f"load fonts   : {mean_lf:>6.3f}s ± {std_lf:.3f}s")
        print(f"load chars   : {mean_lc:>6.3f}s ± {std_lc:.3f}s")
        print(f"build glyphs : {mean_bg:>6.3f}s ± {std_bg:.3f}s")
        print(f"{'─'*60}")


if __name__ == "__main__":
    from configs.default import ATLAS
    atlas_config = ATLAS
    start_time   = time.time()
    atlas        = Atlas(atlas_config)
    atlas.glyph_properties()
    atlas.benchmark(atlas_config)