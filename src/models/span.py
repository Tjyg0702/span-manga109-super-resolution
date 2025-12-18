"""
SPAN: Swift Parameter-free Attention Network for Efficient Super-Resolution.
Reference: Wan et al., CVPR Workshops (NTIRE) 2024
Official repo: https://github.com/hongyuanyu/SPAN

Key insight: Attention weights derived from feature statistics via symmetric 
activation (sigmoid), requiring zero additional parameters. O(n) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """Conv with adaptive padding."""
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


class Conv3XC(nn.Module):
    """
    Re-parameterizable 3x3 convolution block.
    Training: uses decomposed convs (1x1 -> 3x3 -> 1x1) + skip.
    Inference: fuses to single 3x3 conv.
    """
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super().__init__()
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, kernel_size=1, padding=0, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        """Fuse decomposed convs into single 3x3 for inference."""
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        H_pad = W_pad = (3 - 1) // 2
        sk_w = F.pad(sk_w, [H_pad, H_pad, W_pad, W_pad])

        self.eval_conv.weight.data = weight_concat + sk_w
        self.eval_conv.bias.data = bias_concat + sk_b

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)
        
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """
    Swift Parameter-free Attention Block.
    Uses sigmoid attention: (out3 + x) * (sigmoid(out3) - 0.5)
    """
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        # Parameter-free attention
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att


class SPAN(nn.Module):
    """
    Swift Parameter-free Attention Network.
    Default config: 6 SPAB blocks, 48 channels (~860K params for x4).
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        feature_channels: int = 48,
        upscale: int = 4,
        bias: bool = True,
        img_range: float = 255.0,
        rgb_mean: tuple = (0.4488, 0.4371, 0.4040)
    ):
        super().__init__()
        
        self.img_range = img_range
        self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))

        self.conv_1 = Conv3XC(num_in_ch, feature_channels, gain1=2, s=1)
        
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        # Upsampler: PixelShuffle
        self.upsampler = nn.Sequential(
            conv_layer(feature_channels, num_out_ch * (upscale ** 2), kernel_size=3),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, _ = self.block_1(out_feature)
        out_b2, _, _ = self.block_2(out_b1)
        out_b3, _, _ = self.block_3(out_b2)
        out_b4, _, _ = self.block_4(out_b3)
        out_b5, _, _ = self.block_5(out_b4)
        out_b6, out_b5_2, _ = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained weights, handling different checkpoint formats."""
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded weights from {checkpoint_path}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SPAN(feature_channels=48, upscale=4)
    print(f"SPAN parameters: {count_parameters(model):,}")
    
    x = torch.randn(1, 3, 48, 48)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (1, 3, 192, 192), f"Expected (1, 3, 192, 192), got {y.shape}"
    print("✓ Forward pass OK")
