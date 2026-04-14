from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class GlobalConfig:
    seed: int
    batch_size: int
    H: int
    W: int
    C: int


@dataclass(frozen=True)
class ModelConfig:
    architecture: tuple
    epochs: int
    learning_rate: float
    mode: str
    n_warmups:      Optional[int]  = 1
    do_validation:  Optional[bool] = True
    live_metrics:   Optional[bool] = True
    load_filepath:  Optional[str]  = None
    save_filepath:  Optional[str]  = None


@dataclass(frozen=True)
class FontConfig:
    font: int
    prob: float


@dataclass(frozen=True)
class BackgroundConfig:
    background: bool
    prob_real: float

    prob_solid: float
    prob_screentone: float
    prob_hatching: float

    prob_panel_border: float

    # Solid params
    solid_value: Tuple[float, float]

    # Screentone params
    screentone_dot_radius: Tuple[float, float]
    screentone_spacing_factor: Tuple[float, float]
    screentone_angle: Tuple[float, float]
    screentone_dot_value: Tuple[float, float]

    # Hatching params
    hatching_angle: Tuple[float, float]
    hatching_spacing: Tuple[float, float]
    hatching_thickness: Tuple[float, float]
    hatching_line_value: Tuple[float, float]

    # Panel borders
    panel_n_h_cuts: Tuple[int, int]
    panel_n_v_cuts: Tuple[int, int]
    panel_border_width: Tuple[int, int]

    # Brightness
    brightness: Tuple[float, float]


@dataclass(frozen=True)
class ImageAugConfig:
    image_aug: bool
    prob_noise: float
    noise_sigma: Tuple[float, float]

    prob_brightness: float
    brightness: Tuple[float, float]

    prob_jpeg: float
    jpeg_quality: Tuple[int, int]


@dataclass(frozen=True)
class SyntheticOrDataConfig:
    prob: float


@dataclass(frozen=True)
class DataConfig:
    X: Any = None
    y: Any = None
    data_filepath: Optional[str] = None


@dataclass(frozen=True)
class LoaderConfig:
    N_workers: int
    N_chunks: int
    buffer_size: int
    data_generator: Any
    data_generator_init: Any


@dataclass(frozen=True)
class AtlasConfig:
    glyph_size: int
    max_scaling_sentence: Optional[int] = 1
    fonts: Optional[Any] = None
    char_list: Optional[Any] = None


@dataclass(frozen=True)
class LayoutConfig:
    # Sentence
    length: Tuple[int, int]
    direction: float

    # Characters
    char_spacing_vertical: Tuple[float, float]
    char_spacing_horizontal: Tuple[float, float]
    jitters: Tuple[float, float]
    rotation: Tuple[float, float]


@dataclass(frozen=True)
class RenderConfig:
    stroke: Tuple[int, int]
    blur: Tuple[float, float]
    intensity: Tuple[float, float]
    dropout: Tuple[float, float]


@dataclass(frozen=True)
class TextConfig:
    name: str
    prob: int

    speech_bubble_probs: Tuple[float, ...]
    sentence_scaling: Tuple[float, float]
    char_scaling: Tuple[float, float]
    fonts: Tuple[FontConfig, ...]

    layoutconfig: LayoutConfig
    renderconfig: RenderConfig


@dataclass(frozen=True)
class SentenceConfig:
    max_sentences_per_image: Tuple[int, int]
    sentence_count_mean: float
    sentence_count_std: float
    prob_next_sentence: float
    max_line_width_bounds: Tuple[int, int]
    max_line_height_bounds: Tuple[int, int]
    prob_real_sentence: float
    bubble_margin: float


@dataclass(frozen=True)
class SpeechBubbleConfig:
    speech_bubble: bool

    prob_ellipse:   float   # standard dialogue
    prob_jagged:    float   # shouting/action
    prob_rectangle: float   # narration box
    prob_wavy:      float   # fear/hesitation
    prob_none:      float   # no bubble (SFX / floating text)

    # Ellipse params
    ellipse_margin:     Tuple[float, float]   # padding around text box

    # Jagged params
    jagged_n_spikes:    Tuple[int, int]       # number of spikes around perimeter
    jagged_spike_ratio: Tuple[float, float]   # inner/outer radius ratio, controls spike sharpness

    # Wavy params
    wavy_amplitude:     Tuple[float, float]   # how much the outline wobbles
    wavy_frequency:     Tuple[int, int]       # how many waves around the perimeter

    # Shared
    outline_width:  Tuple[int, int]           # border thickness in pixels
    fill_value:     Tuple[float, float]       # bubble interior brightness (usually near 1.0)
    bubble_margin:  float                     # space between text bounding box and bubble edge


@dataclass(frozen=True)
class CharHeatmapConfig:
    sigma: Tuple[float, float]
    radius: Tuple[float, float]
    intensity: Tuple[float, float]


@dataclass(frozen=True)
class AffinHeatmapConfig:
    sigma: Tuple[float, float]
    radius: Tuple[float, float]
    intensity: Tuple[float, float]
    n_steps: int


@dataclass(frozen=True)
class LossConfig:
    eps_BCE:  float
    eps_dice: float

    lambda_reg:  float
    lambda_BCE:  float
    lambda_dice: float

    weight:          Optional[float] = 40
    alpha:           Optional[float] = 0.9
    gamma:           Optional[int]   = 2
    use_auto_lambda: Optional[bool]  = False


@dataclass(frozen=True)
class TrainConfig:
    g:           GlobalConfig
    model:       ModelConfig
    atlas:       AtlasConfig
    text:        Tuple[TextConfig, ...]
    sentence:    SentenceConfig
    speechbubble: SpeechBubbleConfig
    background:  BackgroundConfig
    image:       ImageAugConfig
    mix:         SyntheticOrDataConfig
    affin:       AffinHeatmapConfig
    char:        CharHeatmapConfig
    loss:        LossConfig
    loader:      Optional[LoaderConfig] = None
    data:        Optional[DataConfig]   = None


@dataclass(frozen=True)
class InferenceConfig:
    g:     GlobalConfig
    model: ModelConfig


@dataclass(frozen=True)
class BenchmarkParamsConfig:
    benchmark_atlas:      Optional[bool] = False
    n_rounds_atlas:       Optional[int]  = 10
    device_atlas:         Optional[str]  = "cpu"

    benchmark_loader:     Optional[bool] = False
    n_rounds_loader:      Optional[int]  = 10
    device_loader:        Optional[str]  = "cpu"

    benchmark_generator:  Optional[bool] = False
    device_generator:     Optional[str]  = "cpu"
    n_rounds_generator:   Optional[int]  = 10
    n_warmups_generator:  Optional[int]  = 5
    batch_size_generator: Optional[int]  = 10

    benchmark_network:    Optional[bool] = False
    device_network:       Optional[str]  = "cuda"
    n_rounds_network:     Optional[int]  = 10
    n_warmups_network:    Optional[int]  = 5

    benchmark_background: Optional[bool] = False
    device_background:    Optional[str]  = "cpu"


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark:    BenchmarkParamsConfig
    g:            GlobalConfig
    model:        ModelConfig
    atlas:        AtlasConfig
    text:         Tuple[TextConfig, ...]
    sentence:     SentenceConfig
    speechbubble: SpeechBubbleConfig
    background:   BackgroundConfig
    image:        ImageAugConfig
    mix:          SyntheticOrDataConfig
    affin:        AffinHeatmapConfig
    char:         CharHeatmapConfig
    loss:         LossConfig
    loader:       Optional[LoaderConfig] = None
    data:         Optional[DataConfig]   = None