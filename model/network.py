import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "network")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, lax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)

import time
import sys
import gc
import threading
import multiprocessing as mp
from functools import partial
from typing import NamedTuple, Optional, Tuple

import numpy as np
import optax
import flax
import cv2
import matplotlib.pyplot as plt

from model.loader import MultiProcessLoader, data_generator, data_generator_init
from configs.configurations import TrainConfig, InferenceConfig, BenchmarkConfig

# ── Layer type IDs ────────────────────────────────────────────────────────────
FLATTEN       = 0
FC            = 1
CONV          = 2
POOL_MAX      = 3
POOL_SUM      = 4
LAYER_NORM    = 5
TRANSFORMER   = 6
DROPOUT       = 7
ADD           = 8
GPOOL_MAX     = 9
GPOOL_SUM     = 10
NNUPSAMPLING  = 11
CONCATENATION = 12
BUPSAMPLING   = 13

# ── Params and State ──────────────────────────────────────────────────────────
class FCParams(NamedTuple):
    W: jnp.ndarray
    b: jnp.ndarray

class ConvParams(NamedTuple):
    W: jnp.ndarray
    b: jnp.ndarray

class NormParams(NamedTuple):
    gamma: jnp.ndarray
    beta:  jnp.ndarray

class TrainState(NamedTuple):
    params:    any
    opt_state: any
    rng:       jax.random.PRNGKey

# ── Layer configs ─────────────────────────────────────────────────────────────
class FCLayerConfig(NamedTuple):
    activation: Optional[str]
    units:      Optional[int]
    type:       int = FC

class ConvLayerConfig(NamedTuple):
    strides:      Tuple[int, int]
    padding:      any
    activation:   Optional[str]
    rhs_dilation: Tuple[int, int]
    lhs_dilation: Tuple[int, int]
    type:         int = CONV

class PoolMaxLayerConfig(NamedTuple):
    window:  Tuple[int, int, int, int]
    strides: Tuple[int, int, int, int]
    type:    int = POOL_MAX

class PoolSumLayerConfig(NamedTuple):
    window:  Tuple[int, int, int, int]
    strides: Tuple[int, int, int, int]
    type:    int = POOL_SUM

class LayerNormConfig(NamedTuple):
    epsilon: float = 1e-5
    type:    int   = LAYER_NORM

class FlattenConfig(NamedTuple):
    type: int = FLATTEN

class DropoutConfig(NamedTuple):
    p:    float
    type: int = DROPOUT

class AddConfig(NamedTuple):
    skip: int
    type: int = ADD

class GPoolMaxConfig(NamedTuple):
    type: int = GPOOL_MAX

class GPoolSumConfig(NamedTuple):
    type: int = GPOOL_SUM

class NearestNeighbourUpsamplingConfig(NamedTuple):
    scaling: int
    type:    int = NNUPSAMPLING

class ConcatenatingConfig(NamedTuple):
    skip: int
    type: int = CONCATENATION

class BilinearUpsamplingConfig(NamedTuple):
    scaling: int
    type:    int = BUPSAMPLING

# ── Metrics ───────────────────────────────────────────────────────────────────
class Metrics(NamedTuple):
    total_loss_avg: float
    total_loss_std: float

    BCE_loss_avg: float
    BCE_loss_std: float

    dice_loss_avg: float
    dice_loss_std: float

    reg_loss_avg: float
    reg_loss_std: float

# ── Neural Network ────────────────────────────────────────────────────────────
class NeuralNetwork:
    """
    A fully custom convolutional neural network (CNN) implementation built using JAX.
    This class provides a framework for defining, training and evaluating CNN architectures
    using JAX operations. It supports multiple layer types including convolution, pooling,
    flatten, fully connected, and upsampling. Users may specify an arbitrary layer sequence
    through defining an architecture in the following manner:

    architecture = [
        {"type": "fc", "units": 128, "activation": "relu"},
        {"type": "flatten"},
        {"type": "layer_normalization"},
        {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1,
         "lhs_dilation": (1,1), "rhs_dilation": (1,1), "activation": "leaky_relu"},
        {"type": "pool_max", "kernel_size": 2, "stride": 2},
        {"type": "pool_sum", "kernel_size": 2, "stride": 2},
        {"type": "transformer", "heads": 8},
        {"type": "add", "skip": 0},           # sums output of nth layer with current
        {"type": "gpool_max"},
        {"type": "gpool_sum"},
        {"type": "nearest_neighbour_upsampling", "scaling": 2},
        {"type": "concatenation", "skip": 0},  # concatenates output of nth layer with current
        {"type": "bilinear_upsampling", "scaling": 2},
    ]
    """

    def __init__(self, config=None, *args, **kwargs):
        self.t0_init = time.time()
        self.loader  = kwargs.get("loader", None)

        if isinstance(config, InferenceConfig):
            model_config  = config.model
            global_config = config.g
            architecture  = list(model_config.architecture)
            rng_seed      = global_config.seed

            self.mode          = model_config.mode
            self.load_filepath = model_config.load_filepath
            self.save_filepath = model_config.save_filepath or "checkpoints/weights.pkl"

            H, W, C          = global_config.H, global_config.W, global_config.C
            N_targets        = 2
            self.input_size  = (H, W, C)
            self.output_size = (H, W, N_targets)

            init_log_lambdas = jnp.zeros(2, dtype=jnp.float32)

        elif isinstance(config, (TrainConfig, BenchmarkConfig)):
            model_config  = config.model
            global_config = config.g
            loader_config = config.loader
            data_config   = config.data
            self.loss_config = config.loss

            architecture          = list(model_config.architecture)
            rng_seed              = global_config.seed
            self.mode             = model_config.mode
            self.epochs           = model_config.epochs
            self.train_batch_size = global_config.batch_size
            self.do_validation    = model_config.do_validation
            self.live_metrics     = model_config.live_metrics

            self.lambda_BCE  = self.loss_config.lambda_BCE
            self.lambda_dice = self.loss_config.lambda_dice
            self.lambda_reg  = self.loss_config.lambda_reg
            init_log_lambdas = jnp.array(
                [-0.5 * float(jnp.log(2.0 * self.lambda_BCE)),
                 -0.5 * float(jnp.log(2.0 * self.lambda_dice))],
                dtype=jnp.float32,
            )

            self.learning_rate = model_config.learning_rate
            self.load_filepath = model_config.load_filepath
            self.save_filepath = model_config.save_filepath or "checkpoints/weights.pkl"

            H, W, C   = global_config.H, global_config.W, global_config.C
            N_targets = 2
            self.input_size    = (H, W, C)
            self.output_size   = (H, W, N_targets)
            self.X_dummy_batch = jnp.zeros((1, H, W, C),         dtype=jnp.float32)
            self.y_dummy_batch = jnp.zeros((1, H, W, N_targets),  dtype=jnp.float32)
            self.n_warmup      = model_config.n_warmups

            if self.mode == "train":
                if loader_config is not None:
                    self.N_input  = self.loader.n_chunks * self.loader.n_batches
                    self.N_output = 1
                elif data_config is not None:
                    pass  # TODO: load X, y from data_config
                else:
                    raise ValueError("Train mode requires either loader or data config")

        else:
            architecture = kwargs.get("architecture", None)
            if architecture is None:
                raise ValueError("Insert architecture")
            self.mode          = kwargs.get("mode", "inference")
            rng_seed           = kwargs.get("rng_seed", 0)
            self.load_filepath = kwargs.get("load_filepath", None)
            self.save_filepath = kwargs.get("save_filepath", "checkpoints/weights.pkl")

            if self.mode == "inference":
                self.init_inference(**kwargs)
            elif self.mode == "train":
                X = kwargs.get("X", None)
                y = kwargs.get("y", None)
                self.init_train(X, y, **kwargs)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        self.rng          = random.PRNGKey(rng_seed)
        self.rng_np       = rng_seed
        self.architecture = [dict(layer) for layer in architecture]
        self.num_layers   = len(self.architecture)

        self.initiate_metrics()
        self.params               = []
        self.layer_configs        = []
        self.layer_output_channels = [self.input_size[2]]
        self.initializers()
        self.initialize_params()

        all_params = {
            "net":         self.params,
            "log_lambdas": init_log_lambdas,
        }

        if self.mode == "train":
            decay_steps  = kwargs.get("decay_steps",  max(1, self.epochs * max(1, self.N_input // self.train_batch_size)))
            warmup_steps = kwargs.get("warmup_steps", max(200, decay_steps // 40))

            self.lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=1e-6,
            )
            lambda_lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=1e-6,
            )
            param_labels = {
                "net":         "net",
                "log_lambdas": "lambda",
            }
            lambda_opt     = optax.adam(lambda_lr_schedule) if self.loss_config.use_auto_lambda else optax.set_to_zero()
            self.optimizer = optax.multi_transform(
                {
                    "net": optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.adamw(self.lr_schedule, weight_decay=1e-5),
                    ),
                    "lambda": lambda_opt,
                },
                param_labels,
            )

            self.state = TrainState(
                params=all_params,
                opt_state=self.optimizer.init(all_params),
                rng=self.rng,
            )
        elif self.mode == "inference":
            self.state = TrainState(
                params=all_params,
                opt_state=None,
                rng=self.rng,
            )

        if self.load_filepath is not None:
            self.load_weights()
        elif self.mode == "inference":
            raise ValueError("Inference mode requires load_filepath")

        self.warmup = False

    def init_train(self, X, y, **kwargs):
        self.epochs           = kwargs.get("epochs", 1000)
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.lambda_reg       = kwargs.get("lambda_reg", 1e-6)
        self.lambda_BCE       = kwargs.get("lambda_BCE", 0.5)
        self.lambda_dice      = kwargs.get("lambda_dice", 0.5)
        self.lambdas          = jnp.array([self.lambda_BCE, self.lambda_dice, self.lambda_reg])
        self.learning_rate    = kwargs.get("learning_rate", 0.5e-1)
        if self.loader is None:
            self.N_input  = X.shape[0]
            self.N_output = y.shape[0]

            X = self.NHWC_check(X)
            y = y.astype(np.float32)
            self.X_train, self.X_val = self.make_train_and_val_points(X, self.N_input)
            self.y_train, self.y_val = self.make_train_and_val_points(y, self.N_output)
            self.X_train_batches, self.y_train_batches = NeuralNetwork.make_train_batch(self.train_batch_size, self.X_train, self.y_train)
            self.X_val_batches,   self.y_val_batches   = NeuralNetwork.make_train_batch(self.train_batch_size, self.X_val,   self.y_val)
            self.X_dummy_batch = self.X_train_batches[0]
            self.y_dummy_batch = self.y_train_batches[0]
            self.input_size    = self.X_train.shape[1:]
            self.output_size   = self.y_train.shape[1:]
        else:
            H         = self.loader.H
            W         = self.loader.W
            C         = self.loader.C
            N_targets = self.loader.N_targets

            self.input_size    = (H, W, C)
            self.output_size   = (H, W, N_targets)
            self.X_dummy_batch = jnp.zeros((1, H, W, C),         dtype=jnp.float32)
            self.y_dummy_batch = jnp.zeros((1, H, W, N_targets),  dtype=jnp.float32)
            self.N_input       = self.loader.n_chunks * self.loader.n_batches
            self.N_output      = 1

    def init_inference(self, **kwargs):
        self.input_size  = kwargs.get("input_size",  (512, 512, 1))
        self.output_size = kwargs.get("output_size", None)

    def NHWC_check(self, X):
        if X.ndim != 4:
            raise ValueError(f"Incorrect input size: {X.shape}")
        if X.shape[1] in [1, 3]:
            X = np.transpose(X, (0, 2, 3, 1))
        return X.astype(np.float32)

    def normalize(self, X, std=None, mean=None):
        if mean is None:
            mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
        if std is None:
            std = np.std(X, axis=(0, 1, 2), keepdims=True) + 1e-8
        X = (X - mean) / std
        return X.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    def make_train_and_val_points(self, X, N, split=0.8):
        rng  = np.random.default_rng(self.rng_np)
        perm = rng.permutation(N)
        X    = X[perm]
        X_train = X[:int(N * split)]
        X_val   = X[int(N * split):]
        return X_train, X_val

    @staticmethod
    def make_train_batch(train_batch_size, X, y):
        N           = X.shape[0]
        num_batches = N // train_batch_size
        if num_batches == 0:
            raise ValueError("Batch size is larger than the number of train data points")
        X_batches = X[:num_batches * train_batch_size].reshape((num_batches, train_batch_size, *X.shape[1:]))
        y_batches = y[:num_batches * train_batch_size].reshape((num_batches, train_batch_size, *y.shape[1:]))
        return X_batches, y_batches

    def fc_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        rng, k      = random.split(rng)
        unit_size   = layer['units']
        activation  = layer['activation']
        if activation == "relu":
            W = random.normal(k, (unit_size, previous_channel_size)) * jnp.sqrt(2.0 / previous_channel_size)
        elif activation == "leaky_relu":
            slope = layer.get("slope", 0.01)
            W = random.normal(k, (unit_size, previous_channel_size)) * jnp.sqrt(2.0 / ((1 + slope**2) * previous_channel_size))
        elif activation in ["tanh", "sigmoid", "linear", "softmax"]:
            W = random.normal(k, (unit_size, previous_channel_size)) * jnp.sqrt(2.0 / (previous_channel_size + unit_size))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        b = jnp.zeros((unit_size,))
        self.params.append(FCParams(W, b))
        self.layer_configs.append(FCLayerConfig(activation=activation, units=unit_size))
        previous_channel_size = unit_size
        return rng, previous_height, previous_width, previous_channel_size

    def flatten_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        self.params.append(None)
        self.layer_configs.append(FlattenConfig())
        previous_channel_size = previous_channel_size * previous_height * previous_width
        return rng, previous_height, previous_width, previous_channel_size

    def layer_normalization_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        gamma = jnp.ones((previous_channel_size,))
        beta  = jnp.zeros((previous_channel_size,))
        self.params.append(NormParams(gamma, beta))
        self.layer_configs.append(LayerNormConfig())
        return rng, previous_height, previous_width, previous_channel_size

    def convolution_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        rng, k1     = random.split(rng, 2)
        filters     = layer['filters']
        kernel_size = layer['kernel_size']
        stride      = layer.get('stride', 1)
        padding     = layer.get('padding', 0)
        activation  = layer['activation']
        strides     = (stride, stride)

        rhs_dilation = layer.get('rhs_dilation', (1, 1))
        lhs_dilation = layer.get('lhs_dilation', (1, 1))

        fan_in  = previous_channel_size * kernel_size * kernel_size
        fan_out = filters * kernel_size * kernel_size

        if activation == "relu":
            W = random.normal(k1, (kernel_size, kernel_size, previous_channel_size, filters)) * jnp.sqrt(2.0 / fan_in)
        elif activation == "leaky_relu":
            slope = layer.get("slope", 0.01)
            W = random.normal(k1, (kernel_size, kernel_size, previous_channel_size, filters)) * jnp.sqrt(2.0 / ((1 + slope**2) * fan_in))
        elif activation in ["tanh", "sigmoid", "linear", "softmax"]:
            W = random.normal(k1, (kernel_size, kernel_size, previous_channel_size, filters)) * jnp.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        b = jnp.zeros((filters,))
        self.params.append(ConvParams(W, b))

        padding_mode = "VALID" if padding == 0 else ((padding, padding), (padding, padding))
        self.layer_configs.append(ConvLayerConfig(
            strides=strides, padding=padding_mode, activation=activation,
            rhs_dilation=rhs_dilation, lhs_dilation=lhs_dilation,
        ))

        previous_height_eff = previous_height + (previous_height - 1) * (lhs_dilation[0] - 1)
        previous_width_eff  = previous_width  + (previous_width  - 1) * (rhs_dilation[1] - 1)
        kernel_eff_height   = kernel_size + (kernel_size - 1) * (rhs_dilation[0] - 1)
        kernel_eff_width    = kernel_size + (kernel_size - 1) * (rhs_dilation[1] - 1)

        previous_height       = (previous_height_eff - kernel_eff_height + 2 * padding) // stride + 1
        previous_width        = (previous_width_eff  - kernel_eff_width  + 2 * padding) // stride + 1
        previous_channel_size = filters
        return rng, previous_height, previous_width, previous_channel_size

    def pool_max_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        kernel_size = layer['kernel_size']
        stride      = layer.get('stride', kernel_size)
        window      = (1, kernel_size, kernel_size, 1)
        strides     = (1, stride, stride, 1)
        self.params.append(None)
        self.layer_configs.append(PoolMaxLayerConfig(window=window, strides=strides))
        previous_height = (previous_height - kernel_size) // stride + 1
        previous_width  = (previous_width  - kernel_size) // stride + 1
        return rng, previous_height, previous_width, previous_channel_size

    def pool_sum_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        kernel_size = layer['kernel_size']
        stride      = layer.get('stride', kernel_size)
        window      = (1, kernel_size, kernel_size, 1)
        strides     = (1, stride, stride, 1)
        self.params.append(None)
        self.layer_configs.append(PoolSumLayerConfig(window=window, strides=strides))
        previous_height = (previous_height - kernel_size) // stride + 1
        previous_width  = (previous_width  - kernel_size) // stride + 1
        return rng, previous_height, previous_width, previous_channel_size

    def transformer_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        heads   = layer['heads']
        d_model = previous_channel_size
        d_ff    = layer.get("d_ff", 4 * d_model)
        rng, k1, k2, k3, k4, k5, k6 = random.split(rng, 7)
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by number of heads ({heads})")
        Wq = random.normal(k1, (d_model, d_model)) * jnp.sqrt(1.0 / previous_channel_size)
        Wk = random.normal(k2, (d_model, d_model)) * jnp.sqrt(1.0 / previous_channel_size)
        Wv = random.normal(k3, (d_model, d_model)) * jnp.sqrt(1.0 / previous_channel_size)
        Wo = random.normal(k4, (d_model, d_model)) * jnp.sqrt(1.0 / previous_channel_size)
        W1 = random.normal(k5, (d_model, d_ff))    * jnp.sqrt(1.0 / previous_channel_size)
        b1 = jnp.zeros((d_ff,))
        W2 = random.normal(k6, (d_ff, d_model))    * jnp.sqrt(1.0 / previous_channel_size)
        b2 = jnp.zeros((d_model,))
        self.params.append({"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "W1": W1, "b1": b1, "W2": W2, "b2": b2})
        previous_channel_size = d_model
        return rng, previous_height, previous_width, previous_channel_size

    def dropout_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        p = layer.get("p", 0.5)
        self.params.append(None)
        self.layer_configs.append(DropoutConfig(p=p))
        return rng, previous_height, previous_width, previous_channel_size

    def add_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        skip = layer["skip"]
        self.layer_configs.append(AddConfig(skip=skip))
        self.params.append(None)
        return rng, previous_height, previous_width, previous_channel_size

    def gpool_max_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        self.params.append(None)
        self.layer_configs.append(GPoolMaxConfig())
        previous_height = 1
        previous_width  = 1
        return rng, previous_height, previous_width, previous_channel_size

    def gpool_sum_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        self.params.append(None)
        self.layer_configs.append(GPoolSumConfig())
        previous_height = 1
        previous_width  = 1
        return rng, previous_height, previous_width, previous_channel_size

    def nearest_neighbour_upsampling_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        scaling = layer.get('scaling', 1)
        self.params.append(None)
        self.layer_configs.append(NearestNeighbourUpsamplingConfig(scaling=scaling))
        previous_height = previous_height * scaling
        previous_width  = previous_width  * scaling
        return rng, previous_height, previous_width, previous_channel_size

    def concatenation_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        skip = layer["skip"]
        self.layer_configs.append(ConcatenatingConfig(skip=skip))
        self.params.append(None)
        previous_channel_size = previous_channel_size + self.layer_output_channels[skip]
        return rng, previous_height, previous_width, previous_channel_size

    def bilinear_upsampling_layer_initialization(self, rng, layer, previous_height, previous_width, previous_channel_size):
        scaling = layer.get('scaling', 1)
        self.params.append(None)
        self.layer_configs.append(BilinearUpsamplingConfig(scaling=scaling))
        previous_height = previous_height * scaling
        previous_width  = previous_width  * scaling
        return rng, previous_height, previous_width, previous_channel_size

    def initiate_metrics(self):
        self.history = {"train": [], "val": [], "lambdas": []}

    def initializers(self):
        self.initializers = {
            "fc":                             self.fc_layer_initialization,
            "flatten":                        self.flatten_layer_initialization,
            "conv":                           self.convolution_layer_initialization,
            "layer_normalization":            self.layer_normalization_layer_initialization,
            "pool_max":                       self.pool_max_layer_initialization,
            "pool_sum":                       self.pool_sum_layer_initialization,
            "transformer":                    self.transformer_layer_initialization,
            "dropout":                        self.dropout_layer_initialization,
            "add":                            self.add_layer_initialization,
            "gpool_max":                      self.gpool_max_layer_initialization,
            "gpool_sum":                      self.gpool_sum_layer_initialization,
            "nearest_neighbour_upsampling":   self.nearest_neighbour_upsampling_layer_initialization,
            "concatenation":                  self.concatenation_layer_initialization,
            "bilinear_upsampling":            self.bilinear_upsampling_layer_initialization,
        }

    def initialize_params(self):
        rng                   = self.rng
        previous_height       = self.input_size[0]
        previous_width        = self.input_size[1]
        previous_channel_size = self.input_size[2]
        for layer in self.architecture:
            layer_type  = layer['type']
            initializer = self.initializers.get(layer_type)
            if initializer is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            rng, previous_height, previous_width, previous_channel_size = initializer(
                rng, layer, previous_height, previous_width, previous_channel_size
            )
            self.layer_output_channels.append(previous_channel_size)
        self.layer_configs_static = tuple(self.layer_configs)
        self.rng = rng

    activation_lookup = {
        "relu":       lambda x: jnp.maximum(0, x),
        "leaky_relu": lambda x: jnp.where(x > 0, x, 0.01 * x),
        "tanh":       jnp.tanh,
        "sigmoid":    jax.nn.sigmoid,
        "linear":     lambda x: x,
        "softmax":    lambda x: jax.nn.softmax(x, axis=-1),
    }

    @staticmethod
    def apply_activation(Z, activation):
        try:
            act_fn = NeuralNetwork.activation_lookup[activation]
        except KeyError:
            raise ValueError(f"Unsupported activation: {activation}")
        return act_fn(Z)

    @staticmethod
    def fc_layer_forward(layer_params, X, config):
        W = layer_params.W
        b = layer_params.b
        Z = jnp.dot(X, W.T) + b
        return NeuralNetwork.apply_activation(Z, config.activation)

    @staticmethod
    def flatten_layer_forward(layer_params, X, config):
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def layer_normalization_layer_forward(layer_params, X, config):
        gamma, beta = layer_params
        mean   = jnp.mean(X, axis=-1, keepdims=True)
        var    = jnp.var(X,  axis=-1, keepdims=True)
        X_norm = (X - mean) / jnp.sqrt(var + 1e-5)
        return gamma * X_norm + beta

    @staticmethod
    def convolution_layer_forward(layer_params, X, config):
        W = layer_params.W
        b = layer_params.b
        Z = lax.conv_general_dilated(
            X, W,
            window_strides=config.strides,
            padding=config.padding,
            lhs_dilation=config.lhs_dilation,
            rhs_dilation=config.rhs_dilation,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        Z = Z + b.reshape((1, 1, 1, -1))
        return NeuralNetwork.apply_activation(Z, config.activation)

    @staticmethod
    def pool_max_layer_forward(layer_params, X, config):
        return lax.reduce_window(X, -jnp.inf, lax.max, config.window, config.strides, padding="VALID")

    @staticmethod
    def pool_sum_layer_forward(layer_params, X, config):
        return lax.reduce_window(X, 0.0, lax.add, config.window, config.strides, padding="VALID")

    @staticmethod
    def transformer_layer_forward(layer_params, X, config):
        pass

    @staticmethod
    def dropout_layer_forward(layer_params, X, config, rng, training=True):
        keep_prob = 1.0 - config.p
        def train_fn(X):
            mask = random.bernoulli(rng, keep_prob, X.shape)
            return (X * mask) / keep_prob
        def eval_fn(X):
            return X
        return lax.cond(training, train_fn, eval_fn, X)

    @staticmethod
    def add_layer_forward(layer_params, X, config, saves):
        return X + saves[config.skip]

    @staticmethod
    def gpool_max_layer_forward(layer_params, X, config):
        return jnp.max(X, axis=(1, 2))

    @staticmethod
    def gpool_sum_layer_forward(layer_params, X, config):
        return jnp.sum(X, axis=(1, 2))

    @staticmethod
    def nearest_neighbour_upsampling_layer_forward(layer_params, X, config):
        scaling = config.scaling
        X = jnp.repeat(X, scaling, axis=1)
        X = jnp.repeat(X, scaling, axis=2)
        return X

    @staticmethod
    def concatenation_layer_forward(layer_params, X, config, saves):
        return jnp.concatenate([X, saves[config.skip]], axis=-1)

    @staticmethod
    def bilinear_upsampling_layer_forward(layer_params, X, config):
        scaling  = config.scaling
        B, H, W, C = X.shape
        new_shape  = (B, H * scaling, W * scaling, C)
        return jax.image.resize(X, new_shape, method='linear')

    layer_forward_int = (
        lambda p, X, c, rng, training, saves: NeuralNetwork.flatten_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.fc_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.convolution_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.pool_max_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.pool_sum_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.layer_normalization_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.transformer_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.dropout_layer_forward(p, X, c, rng, training),
        lambda p, A, c, rng, training, saves: NeuralNetwork.add_layer_forward(p, A, c, saves),
        lambda p, A, c, rng, training, saves: NeuralNetwork.gpool_max_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.gpool_sum_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.nearest_neighbour_upsampling_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.concatenation_layer_forward(p, A, c, saves),
        lambda p, A, c, rng, training, saves: NeuralNetwork.bilinear_upsampling_layer_forward(p, A, c),
    )

    @partial(jax.jit, static_argnames=("layer_configs_static", "layer_forward_int", "num_layers"))
    def forward_propagation(params, X, num_layers, layer_configs_static, layer_forward_int, rng, training=True):
        A       = X
        saves   = [A]
        configs = layer_configs_static
        forward = layer_forward_int
        for idx in range(num_layers):
            config      = configs[idx]
            layer_type  = config.type
            layer_params = params[idx]
            rng, subkey = random.split(rng)
            A = forward[layer_type](layer_params, A, config, subkey, training, saves)
            saves.append(A)
        return A

    @staticmethod
    def MSE_loss_function(y_pred, y_true):
        return jnp.mean((y_pred - y_true) ** 2)

    @staticmethod
    def entropy_loss_function(y_pred, y_true):
        y_pred_shift = y_pred - jnp.max(y_pred, axis=1, keepdims=True)
        log_probs    = y_pred_shift - jnp.log(jnp.sum(jnp.exp(y_pred_shift), axis=1, keepdims=True))
        return -jnp.mean(jnp.sum(y_true * log_probs, axis=1))

    @staticmethod
    def BCE_loss_function(y_pred, y_true, loss_config):
        eps    = loss_config.eps_BCE
        gamma  = loss_config.gamma
        alpha  = loss_config.alpha
        y_pred = jnp.clip(y_pred, eps, 1.0 - eps)

        bce_pos = -jnp.log(y_pred)
        bce_neg = -jnp.log(1.0 - y_pred)

        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_w = (1.0 - p_t) ** gamma
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        loss = alpha_t * focal_w * (y_true * bce_pos + (1.0 - y_true) * bce_neg)
        return jnp.mean(loss)

    @staticmethod
    def dice_loss_function(y_pred, y_true, loss_config):
        eps          = loss_config.eps_dice
        axes         = (1, 2, 3)
        intersection = jnp.sum(y_pred * y_true, axis=axes)
        union        = jnp.sum(y_pred, axis=axes) + jnp.sum(y_true, axis=axes)
        dice         = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - jnp.mean(dice)

    @staticmethod
    def regularization_loss_function(params):
        leaves = jax.tree_util.tree_leaves(params)
        return sum(jnp.sum(leaf**2) for leaf in leaves if jnp.asarray(leaf).size > 0)

    @staticmethod
    def total_loss_function(params, X, y, num_layers, layer_configs_static, loss_config, layer_forward_int, rng, training=True):
        net_params  = params["net"]
        log_lambdas = params["log_lambdas"]
        log_lambdas = jnp.clip(log_lambdas, -5.0, 5.0)
        y_pred   = NeuralNetwork.forward_propagation(net_params, X, num_layers, layer_configs_static, layer_forward_int, rng, training)
        loss_BCE  = NeuralNetwork.BCE_loss_function(y_pred, y, loss_config)
        loss_dice = NeuralNetwork.dice_loss_function(y_pred, y, loss_config)
        loss_r    = NeuralNetwork.regularization_loss_function(net_params)
        return jnp.array([loss_BCE, loss_dice, loss_r]), log_lambdas

    @partial(jax.jit, static_argnames=("layer_configs_static", "loss_config", "layer_forward_int", "num_layers", "optimizer"))
    def train_step_jitted(state, X, y, num_layers, layer_configs_static, loss_config, layer_forward_int, optimizer):
        params, opt_state, rng = state
        rng, subkey = random.split(rng)

        def loss_fn(p):
            all_losses, log_lambdas = NeuralNetwork.total_loss_function(
                p, X, y, num_layers, layer_configs_static, loss_config, layer_forward_int, subkey
            )
            total_loss = (
                0.5 * jnp.exp(-2.0 * log_lambdas[0]) * all_losses[0] + log_lambdas[0] +
                0.5 * jnp.exp(-2.0 * log_lambdas[1]) * all_losses[1] + log_lambdas[1] +
                loss_config.lambda_reg * all_losses[2]
            )
            return total_loss, all_losses

        (_, all_losses), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_state  = TrainState(new_params, new_opt_state, rng)
        return new_state, all_losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "loss_config", "layer_forward_int", "num_layers", "optimizer"))
    def train_epoch_jitted(state, X_train, y_train, num_layers, layer_configs_static, loss_config, layer_forward_int, optimizer):
        def train_step(state, batch):
            Xb, yb = batch
            new_state, losses = NeuralNetwork.train_step_jitted(
                state, Xb, yb, num_layers, layer_configs_static, loss_config, layer_forward_int, optimizer
            )
            return new_state, losses

        batches = (X_train, y_train)
        state, all_losses = lax.scan(train_step, state, batches)
        return state, all_losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "loss_config", "layer_forward_int", "num_layers"))
    def val_step_jitted(state, X_val, y_val, num_layers, layer_configs_static, loss_config, layer_forward_int):
        params = state.params
        rng    = random.PRNGKey(0)
        losses, _ = NeuralNetwork.total_loss_function(
            params, X_val, y_val, num_layers, layer_configs_static, loss_config, layer_forward_int, rng, training=False
        )
        return losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "loss_config", "layer_forward_int", "num_layers"))
    def val_epoch_jitted(state, X_val, y_val, num_layers, layer_configs_static, loss_config, layer_forward_int):
        def val_step(state, batch):
            Xb, yb = batch
            losses = NeuralNetwork.val_step_jitted(
                state, Xb, yb, num_layers, layer_configs_static, loss_config, layer_forward_int
            )
            return state, losses

        batches = (X_val, y_val)
        _, all_losses = lax.scan(val_step, state, batches)
        return all_losses

    @staticmethod
    def compute_metrics(all_losses, log_lambdas, loss_config):
        lambda_BCE  = 0.5 * jnp.exp(-2.0 * log_lambdas[0])
        lambda_dice = 0.5 * jnp.exp(-2.0 * log_lambdas[1])
        lambda_reg  = loss_config.lambda_reg

        BCE_losses   = all_losses[:, 0] * lambda_BCE
        dice_losses  = all_losses[:, 1] * lambda_dice
        reg_losses   = all_losses[:, 2] * lambda_reg
        total_losses = BCE_losses + dice_losses + reg_losses

        def stats(x):
            return jnp.mean(x), jnp.std(x)

        total_loss_avg, total_loss_std = stats(total_losses)
        BCE_loss_avg,   BCE_loss_std   = stats(BCE_losses)
        dice_loss_avg,  dice_loss_std  = stats(dice_losses)
        reg_loss_avg,   reg_loss_std   = stats(reg_losses)

        return Metrics(
            total_loss_avg=total_loss_avg, total_loss_std=total_loss_std,
            BCE_loss_avg=BCE_loss_avg,     BCE_loss_std=BCE_loss_std,
            dice_loss_avg=dice_loss_avg,   dice_loss_std=dice_loss_std,
            reg_loss_avg=reg_loss_avg,     reg_loss_std=reg_loss_std,
        )

    def compute_history(self):
        train_stack  = jnp.stack(self.history["train"])
        lambda_stack = jnp.stack(self.history["lambdas"])
        train_batch  = jax.device_get(NeuralNetwork.compute_metrics_all(train_stack, lambda_stack, self.loss_config))

        has_val = self.history["val"][0] is not None
        if has_val:
            val_stack = jnp.stack(self.history["val"])
            val_batch = jax.device_get(NeuralNetwork.compute_metrics_all(val_stack, lambda_stack, self.loss_config))

        lambdas_np = jax.device_get(lambda_stack)
        n_epochs   = len(self.history["train"])
        self.history["train"]   = [Metrics(*[f[e] for f in train_batch]) for e in range(n_epochs)]
        self.history["val"]     = [Metrics(*[f[e] for f in val_batch]) for e in range(n_epochs)] if has_val else [None] * n_epochs
        self.history["lambdas"] = [lambdas_np[e] for e in range(n_epochs)]

    @staticmethod
    def compute_metrics_all(all_losses_epochs, log_lambdas_epochs, loss_config):
        def compute_one(losses, log_lambdas):
            return NeuralNetwork.compute_metrics(losses, log_lambdas, loss_config)
        return jax.vmap(compute_one)(all_losses_epochs, log_lambdas_epochs)

    def store_metrics(self, history, metrics):
        for k, v in metrics.items():
            history[k].append(float(v))

    @staticmethod
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"

    @staticmethod
    def build_bar(progress, width=40):
        filled = int(progress * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def print_training_ui(epoch, epochs, train_metrics, val_metrics, loss_config, log_lambdas, start_time, live_metrics):
        elapsed     = time.time() - start_time
        progress    = epoch / epochs
        it_per_sec  = epoch / elapsed if elapsed > 0 else 0
        eta         = (epochs - epoch) / it_per_sec if it_per_sec > 0 else 0
        bar         = NeuralNetwork.build_bar(progress)

        if live_metrics:
            lambdas = (
                f"λ BCE      : {0.5 * float(jnp.exp(-2.0 * log_lambdas[0])):.4f}\n"
                f"λ dice     : {0.5 * float(jnp.exp(-2.0 * log_lambdas[1])):.4f}\n"
            )
            text = (
                f"Epoch {epoch}/{epochs}\n"
                f"{bar}\n"
                f"train loss : {train_metrics.total_loss_avg:.3e} ± {train_metrics.total_loss_std:.1e}\n"
                f"val loss   : {val_metrics.total_loss_avg:.3e} ± {val_metrics.total_loss_std:.1e}\n"
                f"BCE loss   : {train_metrics.BCE_loss_avg:.3e} ± {train_metrics.BCE_loss_std:.1e}\n"
                f"dice loss  : {train_metrics.dice_loss_avg:.3e} ± {train_metrics.dice_loss_std:.1e}\n"
                f"reg loss   : {train_metrics.reg_loss_avg:.3e} ± {train_metrics.reg_loss_std:.1e}\n"
                f"{lambdas}"
                f"speed      : {it_per_sec:.2f} it/s\n"
                f"elapsed    : {NeuralNetwork.format_time(elapsed)}\n"
                f"ETA        : {NeuralNetwork.format_time(eta)}"
            )
        else:
            text = (
                f"Epoch {epoch}/{epochs}\n"
                f"{bar}\n"
                f"speed      : {it_per_sec:.2f} it/s\n"
                f"elapsed    : {NeuralNetwork.format_time(elapsed)}\n"
                f"ETA        : {NeuralNetwork.format_time(eta)}"
            )

        lines = text.count("\n") + 1
        sys.stdout.write(f"\033[{lines}F")
        sys.stdout.write("\033[J")
        print(text)

    def warmup_model(self):
        H, W, C   = self.input_size
        N_targets = self.output_size[2]
        n_batches = self.loader.n_batches if self.loader is not None else self.train_batch_size

        X_step  = jnp.zeros((n_batches, H, W, C),         dtype=jnp.float32)
        y_step  = jnp.zeros((n_batches, H, W, N_targets),  dtype=jnp.float32)
        X_epoch = X_step[None]
        y_epoch = y_step[None]

        _ = NeuralNetwork.forward_propagation(
            self.state.params["net"], X_step, self.num_layers,
            self.layer_configs_static, self.layer_forward_int, self.state.rng,
        )
        _ = NeuralNetwork.train_step_jitted(
            self.state, X_step, y_step,
            self.num_layers, self.layer_configs_static,
            self.loss_config, self.layer_forward_int, self.optimizer,
        )
        _ = NeuralNetwork.val_step_jitted(
            self.state, X_step, y_step,
            self.num_layers, self.layer_configs_static,
            self.loss_config, self.layer_forward_int,
        )
        _ = NeuralNetwork.train_epoch_jitted(
            self.state, X_epoch, y_epoch,
            self.num_layers, self.layer_configs_static,
            self.loss_config, self.layer_forward_int, self.optimizer,
        )
        _ = NeuralNetwork.val_epoch_jitted(
            self.state, X_epoch, y_epoch,
            self.num_layers, self.layer_configs_static,
            self.loss_config, self.layer_forward_int,
        )
        jax.effects_barrier()
        self.warmup = True

    def training_stats(self):
        print(f"\n{'='*60}")
        print(f"{'Training configurations':^60}")
        print(f"{'='*60}")

        # Devices
        print(f"\n{'Devices':^60}")
        print(f"{'-'*60}")
        devices = jax.devices()
        print(f"  Backend                   : {devices[0].platform}")
        print(f"  Number of devices         : {len(devices)}")
        print(f"{'-'*60}")

        # Data
        print(f"\n{'Data':^60}")
        print(f"{'-'*60}")
        if self.loader is not None:
            print(f"  Source                    : Synthetic")
            print(f"  Number of workers         : {self.loader.n_workers}")
            print(f"  Number of chunks          : {self.loader.n_chunks}")
            print(f"  Number of batches         : {self.loader.n_batches}")
            print(f"  Buffer size               : {self.loader.buffer_size}")
        else:
            print(f"  Source                    : Memory dataset")
        print(f"                               (H,  W,  C)")
        print(f"  Input shape               : {self.input_size}")
        print(f"  Output shape              : {self.output_size}")
        print(f"{'-'*60}")

        # Model
        print(f"\n{'Model':^60}")
        print(f"{'-'*60}")
        print(f"  Number of layers          : {self.num_layers}")
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params) if hasattr(x, "size"))
        print(f"  Number of parameters      : {n_params:,}")
        print(f"{'-'*60}")

        # Training
        print(f"\n{'Training parameters':^60}")
        print(f"{'-'*60}")
        print(f"  Epochs                    : {self.epochs}")
        print(f"  Learning rate             : {self.learning_rate:.2e}")
        print(f"  Auto lambda               : {self.loss_config.use_auto_lambda}")
        print(f"  lambda_reg                : {self.loss_config.lambda_reg:.2e}")
        print(f"  lambda BCE                : {self.loss_config.lambda_BCE}")
        print(f"  lambda dice               : {self.loss_config.lambda_dice}")
        print(f"  do validation             : {self.do_validation}")
        print(f"  live metrics              : {self.live_metrics}")
        print(f"{'-'*60}")

        print(f"  Initialization time       : {NeuralNetwork.format_time(time.time() - self.t0_init)}")
        print(f"{'-'*60}")
        print(f"\n{'Starting training':^60}")
        print("\n" * 10)

    def train(self):
        do_validation = self.do_validation
        live_metrics  = self.live_metrics

        if not self.warmup:
            print("Pre jitting architecture")
            for _ in range(self.n_warmup):
                self.warmup_model()
            print("Pre jitting architecture complete")

        gpu        = jax.devices('gpu')[0]
        start_time = time.time()

        print(f"Pre jitting {self.loader.n_workers} synthetic data generators")

        def prefetch(result_holder):
            buf_idx, features, targets = self.loader.acquire_buffer()
            result_holder['buf_idx'] = buf_idx
            result_holder['X']       = jax.device_put(features, device=gpu)
            result_holder['y']       = jax.device_put(targets,  device=gpu)

        holder = {}
        prefetch(holder)
        self.loader.release_buffer(holder['buf_idx'])
        X_dev, y_dev = holder['X'], holder['y']
        print(f"Pre jitting {self.loader.n_workers} synthetic data generators complete")
        self.training_stats()

        split = max(1, int(self.loader.n_chunks * 0.8)) if do_validation else self.loader.n_chunks

        for epoch in range(1, self.epochs + 1):
            next_holder = {}
            t = threading.Thread(target=prefetch, args=(next_holder,))
            t.start()

            self.state, all_losses_train = NeuralNetwork.train_epoch_jitted(
                self.state,
                X_dev[:split], y_dev[:split],
                self.num_layers,
                self.layer_configs_static, self.loss_config, self.layer_forward_int,
                self.optimizer,
            )
            if do_validation:
                all_losses_val = NeuralNetwork.val_epoch_jitted(
                    self.state,
                    X_dev[split:], y_dev[split:],
                    self.num_layers,
                    self.layer_configs_static, self.loss_config, self.layer_forward_int,
                )
            else:
                all_losses_val = None

            t.join()
            self.loader.release_buffer(next_holder['buf_idx'])
            X_dev, y_dev = next_holder['X'], next_holder['y']

            if live_metrics:
                log_lambdas   = jax.device_get(self.state.params["log_lambdas"])
                train_metrics = NeuralNetwork.compute_metrics(all_losses_train, log_lambdas, self.loss_config)
                val_metrics   = NeuralNetwork.compute_metrics(all_losses_val, log_lambdas, self.loss_config) if do_validation else None
                train_metrics = jax.device_get(train_metrics)
                val_metrics   = jax.device_get(val_metrics)

                self.history["train"].append(train_metrics)
                self.history["val"].append(val_metrics)
                self.history["lambdas"].append(log_lambdas)

                NeuralNetwork.print_training_ui(
                    epoch, self.epochs,
                    train_metrics, val_metrics,
                    self.loss_config, log_lambdas,
                    start_time, live_metrics,
                )
            else:
                self.history["train"].append(all_losses_train)
                self.history["val"].append(all_losses_val)
                self.history["lambdas"].append(self.state.params["log_lambdas"])
                NeuralNetwork.print_training_ui(
                    epoch, self.epochs, None, None,
                    self.loss_config, None, start_time, live_metrics,
                )

            if epoch == 250:
                self.save_weights(training=True, epoch=epoch)
            if epoch % 500 == 0:
                self.save_weights(training=True, epoch=epoch)

        self.save_weights()
        if not live_metrics:
            self.compute_history()

    def save_weights(self, training=False, epoch=None):
        base_path = self.save_filepath
        dir_name  = os.path.dirname(base_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if training and epoch is not None:
            root, ext = os.path.splitext(base_path)
            save_path = f"{root}_epoch_{epoch}{ext}"
        else:
            save_path = base_path

        save_params = self.state.params
        bytes_out   = flax.serialization.to_bytes(save_params)

        with open(save_path, "wb") as f:
            f.write(bytes_out)

        if not training:
            print(f"Saved weights to {save_path}")

    def load_weights(self, inference=False):
        if not os.path.exists(self.load_filepath):
            raise FileNotFoundError(f"Weights file not found: {self.load_filepath}")
        with open(self.load_filepath, "rb") as f:
            bytes_in = f.read()
        new_params = flax.serialization.from_bytes(self.state.params, bytes_in)
        if self.mode == "train":
            self.state = TrainState(
                params=new_params,
                opt_state=self.optimizer.init(new_params),
                rng=self.state.rng,
            )
        else:
            self.state = TrainState(
                params=new_params,
                opt_state=None,
                rng=self.rng,
            )
        if not inference:
            print(f"Loaded weights from {self.load_filepath}")

    def release_gpu(self):
        self.state  = None
        self.params = None
        gc.collect()
        jax.clear_caches()  

    def predict(self, X):
        return NeuralNetwork.forward_propagation(
            self.state.params["net"], X, self.num_layers,
            self.layer_configs_static, self.layer_forward_int, self.state.rng, training=False,
        )

    def plot_training_history(self, log_scale=False):
        train  = self.history["train"]
        val    = self.history["val"]
        epochs = np.arange(1, len(train) + 1)

        train_loss     = np.array([m.total_loss_avg for m in train])
        train_loss_std = np.array([m.total_loss_std for m in train])

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(epochs, train_loss, label="Train Loss")
        ax.fill_between(epochs, train_loss - train_loss_std, train_loss + train_loss_std, alpha=0.25)

        if self.do_validation:
            val_loss     = np.array([m.total_loss_avg for m in val])
            val_loss_std = np.array([m.total_loss_std for m in val])
            ax.plot(epochs, val_loss, label="Val Loss")
            ax.fill_between(epochs, val_loss - val_loss_std, val_loss + val_loss_std, alpha=0.25)

        if log_scale:
            ax.set_yscale("log")

        ax.set_title("Total Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("predictions/training_history.png", dpi=150, bbox_inches="tight")
        plt.show()

    def plot_lambdas_history(self):
        lambdas_history = self.history["lambdas"]
        if not lambdas_history:
            print("No lambda history recorded yet.")
            return

        epochs   = np.arange(1, len(lambdas_history) + 1)
        log_bce  = np.array([ll[0] for ll in lambdas_history])
        log_dice = np.array([ll[1] for ll in lambdas_history])
        w_bce    = 0.5 * np.exp(-2.0 * log_bce)
        w_dice   = 0.5 * np.exp(-2.0 * log_dice)

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(epochs, w_bce,  label="λ BCE")
        ax.plot(epochs, w_dice, label="λ dice")
        ax.set_title("Effective loss weights")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight  (0.5 · exp(−2 · log_λ))")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("predictions/lambda_history.png", dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def heatmap_decoder(x_sample, y_heatmap, ax, char_thresh, affinity_thresh, title):
        char_heatmap     = np.array(y_heatmap[:, :, 0])
        affinity_heatmap = np.array(y_heatmap[:, :, 1])
 
        char_mask = char_heatmap     > char_thresh
        aff_mask  = affinity_heatmap > affinity_thresh
        text_mask = np.logical_or(char_mask, aff_mask).astype(np.uint8)
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons    = [cv2.approxPolyDP(c, epsilon=2.0, closed=True) for c in contours]
 
        ax.imshow(x_sample, cmap="gray", vmin=0, vmax=1)
        for poly in polygons:
            pts = poly.reshape(-1, 2)
            ax.plot(
                np.append(pts[:, 0], pts[0, 0]),
                np.append(pts[:, 1], pts[0, 1]),
                'r-', linewidth=2,
            )
        ax.set_title(title)
        ax.axis("off")
 
    def plot_predictions(self, x_test, y_test=None, char_thresh=0.5, affinity_thresh=0.5, sample_check=False):
        x_batch       = x_test[None, ...]
        x_batch       = self.NHWC_check(x_batch)
        y_pred        = self.predict(x_batch)
        x_sample      = x_batch[0]
        y_pred_sample = y_pred[0]
 
        # print(f"Character heatmap     min={float(y_pred_sample[:,:,0].min()):.3f}  max={float(y_pred_sample[:,:,0].max()):.3f}")
        # print(f"Affinity  heatmap     min={float(y_pred_sample[:,:,1].min()):.3f}  max={float(y_pred_sample[:,:,1].max()):.3f}")
 
        if y_test is not None:
            y_true_sample = y_test
            fig, ax = plt.subplots(2, 3, figsize=(16, 16))

            # True 
            self.heatmap_decoder(x_sample, y_true_sample, ax[0, 0], char_thresh, affinity_thresh, "True box")
 
            ax[0, 1].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
            ax[0, 1].imshow(y_true_sample[:, :, 0], cmap="jet", alpha=0.6)
            ax[0, 1].set_title("True Character Heatmap")
            ax[0, 1].axis("off")
 
            ax[0, 2].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
            ax[0, 2].imshow(y_true_sample[:, :, 1], cmap="hot", alpha=0.6)
            ax[0, 2].set_title("True Affinity Heatmap")
            ax[0, 2].axis("off")
 
            # Predictions
            self.heatmap_decoder(x_sample, y_pred_sample, ax[1, 0], char_thresh, affinity_thresh, "Predicted box")
 
            ax[1, 1].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
            ax[1, 1].imshow(y_pred_sample[:, :, 0], cmap="jet", alpha=0.6)
            ax[1, 1].set_title("Predicted Character Heatmap")
            ax[1, 1].axis("off")
 
            ax[1, 2].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
            ax[1, 2].imshow(y_pred_sample[:, :, 1], cmap="hot", alpha=0.6)
            ax[1, 2].set_title("Predicted Affinity Heatmap")
            ax[1, 2].axis("off")
 
        else:
            # Predictions
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))
 
            self.heatmap_decoder(x_sample, y_pred_sample, ax[0], char_thresh, affinity_thresh, "Predicted box")
 
            ax[1].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
            ax[1].imshow(y_pred_sample[:, :, 0], cmap="jet", alpha=0.6)
            ax[1].set_title("Predicted Character Heatmap")
            ax[1].axis("off")
 
            ax[2].imshow(x_sample, cmap="gray", vmin=0, vmax=1)
            ax[2].imshow(y_pred_sample[:, :, 1], cmap="hot", alpha=0.6)
            ax[2].set_title("Predicted Affinity Heatmap")
            ax[2].axis("off")
 
        plt.tight_layout()
        if sample_check:
            plt.show(block=False)
            return np.array(x_sample), np.array(y_pred_sample)
        else:
            plt.show(block=True)
            return 0, 0

    def benchmark(self, n_rounds=10, n_warmups=1):
        do_validation = self.do_validation
        live_metrics  = self.live_metrics

        print(f"\n{'='*60}")
        print(f"{'Benchmark: Network':^60}")
        print(f"{'='*60}")
        print(f"  rounds        : {n_rounds}")
        print(f"  input shape   : {self.input_size}")
        print(f"  output shape  : {self.output_size}")
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params) if hasattr(x, "size"))
        print(f"  n params      : {n_params:,}")
        devices = jax.devices()
        print(f"  device        : {devices[0].platform}")
        print(f"  n_workers     : {self.loader.n_workers}")
        print(f"  n_chunks      : {self.loader.n_chunks}")
        print(f"  n_batches     : {self.loader.n_batches}")
        print(f"{'─'*70}")

        print(f"{f'Pre jitting {n_warmups} times':^60}")
        t_warmup_start = time.perf_counter()
        for _ in range(n_warmups):
            self.warmup_model()
        t_warmup_end = time.perf_counter()
        print(f"Finished pre jit in {t_warmup_end - t_warmup_start:.2f}s")
        print(f"{'─'*70}")

        gpu = jax.devices('gpu')[0]

        def prefetch(result_holder):
            buf_idx, features, targets = self.loader.acquire_buffer()
            result_holder['buf_idx'] = buf_idx
            result_holder['X']       = jax.device_put(features, device=gpu)
            result_holder['y']       = jax.device_put(targets,  device=gpu)

        holder = {}
        prefetch(holder)
        self.loader.release_buffer(holder['buf_idx'])
        X_dev, y_dev = holder['X'], holder['y']

        print(f"{'─'*70}")
        print(f"  {'round':>5}  {'prefetch(s)':>11}  {'train(s)':>8}  {'val(s)':>6}  {'metrics(s)':>10}  {'total(s)':>8}  {'it/s':>6}")
        print(f"{'─'*70}")

        benchmark_times = {
            "prefetch": [],
            "train":    [],
            "val":      [],
            "metrics":  [],
            "total":    [],
            "its":      [],
        }

        split      = max(1, int(self.loader.n_chunks * 0.8)) if do_validation else self.loader.n_chunks
        start_time = time.perf_counter()

        for i in range(n_rounds):
            t_round_start = time.perf_counter()

            next_holder = {}
            t = threading.Thread(target=prefetch, args=(next_holder,))
            t.start()

            t0 = time.perf_counter()
            self.state, all_losses_train = NeuralNetwork.train_epoch_jitted(
                self.state,
                X_dev[:split], y_dev[:split],
                self.num_layers,
                self.layer_configs_static, self.loss_config, self.layer_forward_int,
                self.optimizer,
            )
            jax.block_until_ready(all_losses_train)
            t1 = time.perf_counter()
            benchmark_times["train"].append(t1 - t0)

            if do_validation:
                all_losses_val = NeuralNetwork.val_epoch_jitted(
                    self.state,
                    X_dev[split:], y_dev[split:],
                    self.num_layers,
                    self.layer_configs_static, self.loss_config, self.layer_forward_int,
                )
                jax.block_until_ready(all_losses_val)
            else:
                all_losses_val = None
            t2 = time.perf_counter()
            benchmark_times["val"].append(t2 - t1)

            t.join()
            t3 = time.perf_counter()
            benchmark_times["prefetch"].append(t3 - t2)

            self.loader.release_buffer(next_holder['buf_idx'])
            X_dev, y_dev = next_holder['X'], next_holder['y']

            if live_metrics:
                log_lambdas   = jax.device_get(self.state.params["log_lambdas"])
                train_metrics = NeuralNetwork.compute_metrics(all_losses_train, log_lambdas, self.loss_config)
                val_metrics   = NeuralNetwork.compute_metrics(all_losses_val,   log_lambdas, self.loss_config)
                train_metrics = jax.device_get(train_metrics)
                val_metrics   = jax.device_get(val_metrics)
            t4 = time.perf_counter()
            benchmark_times["metrics"].append(t4 - t3)

            total = t4 - t_round_start
            benchmark_times["total"].append(total)
            benchmark_times["its"].append(1.0 / total if total > 0 else 0.0)

            print(f"  {i+1:>5}  {(t3-t2):>11.3f}  {(t1-t0):>8.3f}  {(t2-t1):>6.3f}  {(t4-t3):>10.3f}  {total:>8.3f}  {1.0/total:>6.3f}")

        end_time = time.perf_counter()
        print(f"{'─'*70}")
        print(f"Finished {n_rounds} rounds in {(end_time - start_time):.2f}s")
        print(f"{'─'*70}")
        print(f"{'Averages and stds':^60}")
        print(f"{'─'*70}")
        for k, v in benchmark_times.items():
            arr = np.array(v)
            print(f"{k:>20}: {arr.mean():.4f}s ± {arr.std():.4f}s")
        print(f"{'─'*70}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    from configs.default import CFG_TRAIN

    loader = MultiProcessLoader(CFG_TRAIN)
    loader.start_workers()
    model = NeuralNetwork(CFG_TRAIN, loader=loader)
    try:
        model.train()
        print("Training is done")
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        model.plot_training_history(log_scale=False)
        buf_idx, features, targets = loader.acquire_buffer()
        for i in range(model.train_batch_size):
            model.plot_predictions(features[0, i], targets[0, i])
        loader.release_buffer(buf_idx)
        loader.stop_workers()