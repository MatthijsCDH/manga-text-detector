import os
from configs.configurations import (
    GlobalConfig, ModelConfig, LoaderConfig, AtlasConfig,
    LayoutConfig, FontConfig, BackgroundConfig,
    RenderConfig, TextConfig, TrainConfig, RealDataConfig,
    AffinHeatmapConfig, CharHeatmapConfig, SentenceConfig, ImageAugConfig,
    SpeechBubbleConfig, LossConfig, InferenceConfig, BenchmarkParamsConfig, BenchmarkConfig
)
from model.loader import data_generator, data_generator_init

REGULAR = 0
BOLD    = 1
IPAG    = 2
NOTO    = 3
THIN    = 4
HEAVY   = 5
BLACK   = 6
BOLD2   = 7
HEAVY2  = 8
ZENHARD = 9
ZENSOFT = 10

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir   = os.path.join(os.path.dirname(__file__), "..", "data", "assets")
font_dir   = os.path.join(data_dir, "Fonts")
kanji_path = os.path.join(data_dir, "Kanji", "jouyou_kanji.txt")

# ── Global ────────────────────────────────────────────────────────────────────
GLOBAL = GlobalConfig(
    seed       = 1,
    batch_size = 1,
    H          = 1248,
    W          = 864,
    C          = 1,
)

# ── Atlas ─────────────────────────────────────────────────────────────────────
ATLAS = AtlasConfig(
    glyph_size           = 48,
    max_scaling_sentence = 2,
)

# ── 1. DIALOGUE ───────────────────────────────────────────────────────────────
fonts_dialogue = (
    FontConfig(font=REGULAR, prob=0.01),
    FontConfig(font=BOLD,    prob=0.099),
    FontConfig(font=IPAG,    prob=0.05),
    FontConfig(font=THIN,    prob=0.001),
    FontConfig(font=NOTO,    prob=0.25),
    FontConfig(font=HEAVY,   prob=0.10),
    FontConfig(font=BLACK,   prob=0.19),
    FontConfig(font=BOLD2,   prob=0.10),
    FontConfig(font=HEAVY2,  prob=0.10),
    FontConfig(font=ZENHARD, prob=0.05),
    FontConfig(font=ZENSOFT, prob=0.05),
)

layout_dialogue = LayoutConfig(
    length                  = (1, 45),
    direction               = 0.15,
    char_spacing_vertical   = (8, 12),
    char_spacing_horizontal = (8, 12),
    jitters                 = (0, 2),
    rotation                = (0.0, 0.0),
)

render_dialogue = RenderConfig(
    stroke    = (2, 6),
    blur      = (0.0, 0.4),
    intensity = (0.85, 1.0),
    dropout   = (0.0, 0.02),
)

DIALOGUE = TextConfig(
    name                = "dialogue",
    prob                = 0.8,
    # (ellipse, jagged, rectangle, wavy, none)
    speech_bubble_probs = (0.50, 0.10, 0.15, 0.15, 0.1),
    sentence_scaling    = (0.3, 1.5),
    char_scaling        = (0.8, 1.1),
    fonts               = fonts_dialogue,
    layoutconfig        = layout_dialogue,
    renderconfig        = render_dialogue,
)

# ── 2. SFX ────────────────────────────────────────────────────────────────────
fonts_sfx = (
    FontConfig(font=BOLD,    prob=0.10),
    FontConfig(font=NOTO,    prob=0.05),
    FontConfig(font=HEAVY,   prob=0.30),
    FontConfig(font=BLACK,   prob=0.30),
    FontConfig(font=BOLD2,   prob=0.05),
    FontConfig(font=HEAVY2,  prob=0.10),
    FontConfig(font=ZENHARD, prob=0.10),
)

layout_sfx = LayoutConfig(
    length                  = (1, 3),
    direction               = 0.35,
    char_spacing_vertical   = (2, 6),
    char_spacing_horizontal = (2, 6),
    jitters                 = (0, 12),
    rotation                = (0.0, 0.0),
)

render_sfx = RenderConfig(
    stroke    = (3, 8),
    blur      = (0.0, 0.8),
    intensity = (0.90, 1.0),
    dropout   = (0.0, 0.0),
)

SFX = TextConfig(
    name                = "sfx",
    prob                = 0.2,
    # (ellipse, jagged, rectangle, wavy, none)
    speech_bubble_probs = (0.05, 0.15, 0.00, 0.05, 0.75),
    sentence_scaling    = (1.0, 2.0),
    char_scaling        = (1.0, 1.1),
    fonts               = fonts_sfx,
    layoutconfig        = layout_sfx,
    renderconfig        = render_sfx,
)

# ── 3. SHOUT ──────────────────────────────────────────────────────────────────
fonts_shout = (
    FontConfig(font=IPAG,    prob=0.05),
    FontConfig(font=NOTO,    prob=0.05),
    FontConfig(font=HEAVY,   prob=0.35),
    FontConfig(font=BLACK,   prob=0.30),
    FontConfig(font=HEAVY2,  prob=0.10),
    FontConfig(font=ZENHARD, prob=0.15),
)

layout_shout = LayoutConfig(
    length                  = (2, 12),
    direction               = 0.05,
    char_spacing_vertical   = (3, 5),
    char_spacing_horizontal = (3, 5),
    jitters                 = (0, 4),
    rotation                = (0.0, 0.0),
)

render_shout = RenderConfig(
    stroke    = (2, 4),
    blur      = (0.0, 0.3),
    intensity = (0.90, 1.0),
    dropout   = (0.0, 0.0),
)

SHOUT = TextConfig(
    name                = "shout",
    prob                = 0.10,
    # (ellipse, jagged, rectangle, wavy, none)
    speech_bubble_probs = (0.10, 0.70, 0.00, 0.15, 0.05),
    sentence_scaling    = (0.8, 2.0),
    char_scaling        = (1.0, 1.1),
    fonts               = fonts_shout,
    layoutconfig        = layout_shout,
    renderconfig        = render_shout,
)

# ── 4. NARRATION ──────────────────────────────────────────────────────────────
fonts_narration = (
    FontConfig(font=REGULAR, prob=0.20),
    FontConfig(font=BOLD,    prob=0.05),
    FontConfig(font=IPAG,    prob=0.05),
    FontConfig(font=NOTO,    prob=0.05),
    FontConfig(font=THIN,    prob=0.10),
    FontConfig(font=BOLD2,   prob=0.05),
    FontConfig(font=HEAVY2,  prob=0.10),
    FontConfig(font=ZENHARD, prob=0.20),
    FontConfig(font=ZENSOFT, prob=0.20),
)

layout_narration = LayoutConfig(
    length                  = (15, 60),
    direction               = 0.25,
    char_spacing_vertical   = (4, 6),
    char_spacing_horizontal = (4, 6),
    jitters                 = (0, 1),
    rotation                = (0.0, 0.0),
)

render_narration = RenderConfig(
    stroke    = (1, 2),
    blur      = (0.0, 0.25),
    intensity = (0.80, 1.0),
    dropout   = (0.0, 0.01),
)

NARRATION = TextConfig(
    name                = "narration",
    prob                = 0.10,
    # (ellipse, jagged, rectangle, wavy, none)
    speech_bubble_probs = (0.00, 0.00, 0.88, 0.00, 0.12),
    sentence_scaling    = (0.55, 0.7),
    char_scaling        = (0.8, 1.0),
    fonts               = fonts_narration,
    layoutconfig        = layout_narration,
    renderconfig        = render_narration,
)

# ── 5. TITLE ──────────────────────────────────────────────────────────────────

fonts_title = (
    FontConfig(font=HEAVY,   prob=0.15),
    FontConfig(font=BLACK,   prob=0.25),
    FontConfig(font=BOLD2,   prob=0.15),
    FontConfig(font=HEAVY2,  prob=0.15),
    FontConfig(font=ZENHARD, prob=0.15),
    FontConfig(font=ZENSOFT, prob=0.15),
)

layout_title = LayoutConfig(
    length                  = (1, 6),
    direction               = 0.40,
    char_spacing_vertical   = (2, 4),
    char_spacing_horizontal = (2, 4),
    jitters                 = (0, 3),
    rotation                = (0.0, 0.0),
)

render_title = RenderConfig(
    stroke    = (2, 4),
    blur      = (0.0, 0.2),
    intensity = (0.95, 1.0),
    dropout   = (0.0, 0.0),
)

TITLE = TextConfig(
    name                = "title",
    prob                = 0.10,
    # (ellipse, jagged, rectangle, wavy, none)
    speech_bubble_probs = (0.00, 0.00, 0.00, 0.00, 1.00),
    sentence_scaling    = (1.0, 2.0),
    char_scaling        = (1.0, 1.1),
    fonts               = fonts_title,
    layoutconfig        = layout_title,
    renderconfig        = render_title,
)

# ── Background ────────────────────────────────────────────────────────────────
BACKGROUND = BackgroundConfig(
    background = True,
    prob_real  = 0.6,

    prob_solid        = 0.15,
    prob_screentone   = 0.40,
    prob_hatching     = 0.45,

    prob_panel_border = 0.60,

    solid_value       = (0.75, 1.0),

    screentone_dot_radius     = (1.0, 5.0),
    screentone_spacing_factor = (2.0, 4.5),
    screentone_angle          = (0.0, 90.0),
    screentone_dot_value      = (0.0, 0.5),

    hatching_angle       = (0.0, 180.0),
    hatching_spacing     = (2.0, 14.0),
    hatching_thickness   = (0.5, 3.5),
    hatching_line_value  = (0.0, 0.5),

    panel_n_h_cuts     = (0, 4),
    panel_n_v_cuts     = (0, 4),
    panel_border_width = (3, 15),

    brightness = (0.7, 1.1),
)

# ── Image augmentation ────────────────────────────────────────────────────────
IMAGE_AUG = ImageAugConfig(
    image_aug   = True,
    prob_noise  = 0.5,
    noise_sigma = (0.01, 0.08),

    prob_brightness = 0.5,
    brightness      = (0.70, 1.20),

    prob_jpeg    = 0.5,
    jpeg_quality = (30, 85),
    
    prob_x_flip  = 0.1,
    prob_y_flip  = 0.1,

    prob_colour_inversion = 0.05,

    crop_size = (0.75, 0.95)
)

# ── CharHeatmap ───────────────────────────────────────────────────────────────
CHARHEATMAP = CharHeatmapConfig(
    sigma     = (30, 40),
    radius    = (30, 40),
    intensity = (1.0, 1.0),
)

# ── AffinHeatmap ──────────────────────────────────────────────────────────────
AFFINHEATMAP = AffinHeatmapConfig(
    sigma     = (20, 40),
    radius    = (20, 40),
    intensity = (1.0, 1.0),
    n_steps   = 2,
)

# ── Mix ───────────────────────────────────────────────────────────────────────
REAL_DATA= RealDataConfig(
    prob_selftraining_data = 0.4,
    prob_annotator_data = 0.4,
)

# ── Sentence ──────────────────────────────────────────────────────────────────
SENTENCE = SentenceConfig(
    max_sentences_per_image = (0, 8),
    sentence_count_mean     = 5.0,
    sentence_count_std      = 3.0,
    prob_next_sentence      = 0.4,
    max_line_width_bounds   = (4, 10),
    max_line_height_bounds  = (4, 10),
    prob_real_sentence      = 0.5,
    bubble_margin           = 1,
)

# ── Speech bubble ─────────────────────────────────────────────────────────────
SPEECHBUBBLE = SpeechBubbleConfig(
    speech_bubble = True,

    prob_ellipse   = 0.45,
    prob_jagged    = 0.30,
    prob_rectangle = 0.05,
    prob_wavy      = 0.03,
    prob_none      = 0.17,

    ellipse_margin     = (8.0, 20.0),

    jagged_n_spikes    = (8, 24),
    jagged_spike_ratio = (0.6, 0.88),

    wavy_amplitude = (2.0, 8.0),
    wavy_frequency = (6, 20),

    outline_width = (2, 6),
    fill_value    = (0.95, 1.0),
    bubble_margin = 10.0,
)

# ── Loader ────────────────────────────────────────────────────────────────────
LOADER = LoaderConfig(
    N_workers           = 4,
    N_chunks            = 4,
    buffer_size         = 3,
    data_generator      = data_generator,
    data_generator_init = data_generator_init,
)

# ── Architecture ──────────────────────────────────────────────────────────────
architecture = [
    # --- Encoder block 1 ---
    {"type": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 1
    {"type": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 2
    {"type": "pool_max", "kernel_size": 2, "stride": 2},                                                   # 3

    # --- Encoder block 2 ---
    {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 4
    {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 5
    {"type": "pool_max", "kernel_size": 2, "stride": 2},                                                   # 6

    # --- Encoder block 3 ---
    {"type": "conv", "filters": 128, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},   # 7
    {"type": "conv", "filters": 128, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},   # 8
    {"type": "pool_max", "kernel_size": 2, "stride": 2},                                                   # 9

    # --- Bottleneck ---
    {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 1,
     "activation": "relu"},                                                                                # 10

    {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 2,
     "rhs_dilation": (2, 2), "activation": "relu"},                                                        # 11

    {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 4,
     "rhs_dilation": (4, 4), "activation": "relu"},                                                        # 12

    {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 8,
     "rhs_dilation": (8, 8), "activation": "relu"},                                                        # 13

    # --- Decoder block 3 ---
    {"type": "bilinear_upsampling", "scaling": 2},                                                         # 14
    {"type": "concatenation", "skip": 8},                                                                  # 15

    {"type": "conv", "filters": 128, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},   # 16
    {"type": "conv", "filters": 128, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},   # 17

    # --- Decoder block 2 ---
    {"type": "bilinear_upsampling", "scaling": 2},                                                         # 18
    {"type": "concatenation", "skip": 5},                                                                  # 19

    {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 20
    {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 21

    # --- Decoder block 1 ---
    {"type": "bilinear_upsampling", "scaling": 2},                                                         # 22
    {"type": "concatenation", "skip": 2},                                                                  # 23

    {"type": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 24
    {"type": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"},    # 25

    # --- Output ---
    {"type": "conv", "filters": 2, "kernel_size": 1, "stride": 1, "padding": 0, "activation": "sigmoid"},  # 26
]

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_TRAIN = ModelConfig(
    architecture  = architecture,
    epochs        = 25000,
    learning_rate = 1e-4,
    mode          = "train",
    do_validation = True,
    live_metrics  = True,
    n_warmups     = 2,
    save_filepath = "checkpoints/weights.pkl",
    load_filepath = "checkpoints/weights.pkl",
)

LOSSCONFIG = LossConfig(
    eps_BCE  = 1e-7,
    eps_dice = 1,

    lambda_reg  = 1e-5,
    lambda_BCE  = 0.5,
    lambda_dice = 0.5,

    weight = 40,
    alpha  = 0.9,
    gamma  = 2,

    use_auto_lambda = True,
)

# ── Train ─────────────────────────────────────────────────────────────────────
CFG_TRAIN = TrainConfig(
    g            = GLOBAL,
    model        = MODEL_TRAIN,
    loss         = LOSSCONFIG,
    loader       = LOADER,
    atlas        = ATLAS,
    text         = (DIALOGUE, SFX), #(DIALOGUE, SFX, NARRATION, SHOUT, TITLE),
    sentence     = SENTENCE,
    speechbubble = SPEECHBUBBLE,
    background   = BACKGROUND,
    image        = IMAGE_AUG,
    real_data    = REAL_DATA,
    char         = CHARHEATMAP,
    affin        = AFFINHEATMAP,
)

# ── Inference ─────────────────────────────────────────────────────────────────
MODEL_INFERENCE = ModelConfig(
    architecture  = architecture,
    epochs        = 100,
    learning_rate = 1e-3,
    mode          = "inference",
    load_filepath = "checkpoints/weights_epoch_25000.pkl",
)

CFG_INFERENCE = InferenceConfig(
    g               = GLOBAL,
    model           = MODEL_INFERENCE,
    sample_checking = True,
)

# ── Benchmark ─────────────────────────────────────────────────────────────────
MODEL_BENCHMARK = ModelConfig(
    architecture  = architecture,
    epochs        = 100,
    learning_rate = 1e-3,
    mode          = "train",
    do_validation = True,
    live_metrics  = True,
)

BENCHMARK = BenchmarkParamsConfig(
    benchmark_atlas      = False,
    n_rounds_atlas       = 20,

    benchmark_background = True,

    benchmark_generator  = True,
    n_rounds_generator   = 20,
    batch_size_generator = 1,
    n_warmups_generator  = 10,

    benchmark_loader = False,
    n_rounds_loader  = 20,

    benchmark_network = True,
    n_rounds_network  = 20,
    n_warmups_network = 2,
)

CFG_BENCHMARK = BenchmarkConfig(
    g            = GLOBAL,
    benchmark    = BENCHMARK,
    model        = MODEL_BENCHMARK,
    loss         = LOSSCONFIG,
    loader       = LOADER,
    atlas        = ATLAS,
    text         = (DIALOGUE, SFX), #(DIALOGUE, SFX, NARRATION, SHOUT, TITLE),
    sentence     = SENTENCE,
    speechbubble = SPEECHBUBBLE,
    background   = BACKGROUND,
    image        = IMAGE_AUG,
    real_data    = REAL_DATA,
    char         = CHARHEATMAP,
    affin        = AFFINHEATMAP,
)