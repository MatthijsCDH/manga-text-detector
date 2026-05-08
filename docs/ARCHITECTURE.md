# Architecture

The model follows a U-Net-like encoder-decoder structure with a transformer bottleneck.

The encoder extracts visual features while gradually reducing spatial resolution. The bottleneck combines dilated convolutions and a transformer encoder to capture wider context. The decoder then upsamples the feature maps back to the original image resolution using skip connections from earlier encoder layers.

The final output is a dense two-channel heatmap prediction. One channel represents character/text regions, while the other represents affinity regions between nearby text components.

## Input and output

The model expects grayscale input images in NHWC format:
Input:  (B, H, W, 1)
Output: (B, H, W, 2)

The two ouput channels are 
0 Character heatmap
1 Affinity heatmap

## Architecture diagram

<p align="center">
  <img src="Architecture.SVG" alt="Manga Text Detector architecture" width="850">
</p>

## Encoder
The encoder consists of three convolutional blocks. Each block contains two 3×3 convolution layers with ReLU activation, followed by a MaxPool 2×2 layer. As the encoder goes deeper, the spatial resolution decreases while the number of feature channels increases. Due to this increase, the model will extract more abstract features from the image that correspond to text-related features. The outputs of selected encoder layers are reused later as skip connections in the decoder. These skip connections help the model recover spatial detail that may be lost during downsampling.  

## Bottleneck

## Decoder

## Layer-by-layer table

The table below shows the full architecture with output shapes. Shapes use NHWC format: `(batch, height, width, channels)`.

| # | Layer | Output shape |
|---:|---|---|
| Input | Input image | `(B, H, W, 1)` |
| 0 | Conv 3×3, 32, stride 1, padding 1, ReLU | `(B, H, W, 32)` |
| 1 | Conv 3×3, 32, stride 1, padding 1, ReLU | `(B, H, W, 32)` |
| 2 | MaxPool 2×2, stride 2 | `(B, H/2, W/2, 32)` |
| 3 | Conv 3×3, 64, stride 1, padding 1, ReLU | `(B, H/2, W/2, 64)` |
| 4 | Conv 3×3, 64, stride 1, padding 1, ReLU | `(B, H/2, W/2, 64)` |
| 5 | MaxPool 2×2, stride 2 | `(B, H/4, W/4, 64)` |
| 6 | Conv 3×3, 128, stride 1, padding 1, ReLU | `(B, H/4, W/4, 128)` |
| 7 | Conv 3×3, 128, stride 1, padding 1, ReLU | `(B, H/4, W/4, 128)` |
| 8 | MaxPool 2×2, stride 2 | `(B, H/8, W/8, 128)` |
| 9 | Conv 3×3, 256, stride 1, padding 1, ReLU | `(B, H/8, W/8, 256)` |
| 10 | Dilated Conv 3×3, 256, dilation 2, padding 2, ReLU | `(B, H/8, W/8, 256)` |
| 11 | Dilated Conv 3×3, 256, dilation 4, padding 4, ReLU | `(B, H/8, W/8, 256)` |
| 12 | MaxPool 2×2, stride 2 | `(B, H/16, W/16, 256)` |
| 13 | Flatten spatial | `(B, H/16 · W/16, 256)` |
| 14 | 2D positional embedding | `(B, H/16 · W/16, 256)` |
| 15 | Transformer encoder, 8 heads, FFN 1024, dropout 0.1 | `(B, H/16 · W/16, 256)` |
| 16 | Unflatten spatial | `(B, H/16, W/16, 256)` |
| 17 | Bilinear upsampling ×2 | `(B, H/8, W/8, 256)` |
| 18 | Concatenate skip from layer 11 | `(B, H/8, W/8, 512)` |
| 19 | Conv 3×3, 256, stride 1, padding 1, ReLU | `(B, H/8, W/8, 256)` |
| 20 | Conv 3×3, 256, stride 1, padding 1, ReLU | `(B, H/8, W/8, 256)` |
| 21 | Bilinear upsampling ×2 | `(B, H/4, W/4, 256)` |
| 22 | Concatenate skip from layer 7 | `(B, H/4, W/4, 384)` |
| 23 | Conv 3×3, 128, stride 1, padding 1, ReLU | `(B, H/4, W/4, 128)` |
| 24 | Conv 3×3, 128, stride 1, padding 1, ReLU | `(B, H/4, W/4, 128)` |
| 25 | Bilinear upsampling ×2 | `(B, H/2, W/2, 128)` |
| 26 | Concatenate skip from layer 4 | `(B, H/2, W/2, 192)` |
| 27 | Conv 3×3, 64, stride 1, padding 1, ReLU | `(B, H/2, W/2, 64)` |
| 28 | Conv 3×3, 64, stride 1, padding 1, ReLU | `(B, H/2, W/2, 64)` |
| 29 | Bilinear upsampling ×2 | `(B, H, W, 64)` |
| 30 | Concatenate skip from layer 1 | `(B, H, W, 96)` |
| 31 | Conv 3×3, 32, stride 1, padding 1, ReLU | `(B, H, W, 32)` |
| 32 | Conv 3×3, 32, stride 1, padding 1, ReLU | `(B, H, W, 32)` |
| 33 | Conv 1×1, 2, stride 1, padding 0, sigmoid | `(B, H, W, 2)` |

For the default input size `H = 1248` and `W = 864`, the final output is:

```text
(B, 1248, 864, 2)
