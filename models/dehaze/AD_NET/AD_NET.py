import torch
import torch.nn as nn
import torch.nn.functional as F
# Consider using actual MobileNetV3 blocks if available from torchvision or implemented
# from torchvision.models import mobilenet_v3_small # Not directly used here, but the idea is similar
import numpy as np # Keep for potential utility

# Assuming these are in a 'utils.py' file relative to this script
from .utils import total_variation_loss, PerceptualLoss

# --- Helper Modules (Defined outside the main class for clarity, but used within) ---

# Reusing Channel Attention and Spatial Attention (from CBAM) - they are lightweight
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Use smaller kernel for potential speedup, or keep 7x7
        self.conv = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False) # Changed to 5x5
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

# Lightweight Convolutional Block (Inspired by MobileNetV3's inverted residual)
class LightweightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=3):
        super().__init__()
        hidden_channels = in_channels * expansion_factor
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        # Pointwise expansion (1x1 conv)
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # Depthwise conv (kxfx)
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_channels, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # Pointwise projection (1x1 conv)
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    # Inside LightweightConvBlock class
    def forward(self, x):
        out = self.conv(x)
        # Check if residual connection is intended AND spatial dimensions match
        # self.use_res_connect is already set in __init__ based on initial stride and channels
        if self.use_res_connect and x.shape[2:] == out.shape[2:]:
            return x + out
        else:
            return out


# Efficient Channel-Spatial Attention Block (Custom, lightweight)
class EfficientCSAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention()
        # Optional: Add a 1x1 conv after attention for feature mixing
        self.conv_mix = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Apply channel and spatial attention sequentially
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)

        out = self.conv_mix(out) # Mix features after attention
        return out + x # Residual connection


# --- The New AD_NET Model (Fast and End-to-End) ---

class AD_NET(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_blocks=[2, 3, 3, 2]):
        """
        A fast, end-to-end dehazing network using lightweight CNN blocks and efficient attention.
        It does not rely on the atmospheric scattering model formula.

        Args:
            in_channels (int): Number of input image channels (default is 3 for RGB).
            base_channels (int): Base number of channels in the first encoder stage.
                                 Channel counts in later stages will be multiples of this.
                                 Lower value means faster model.
            num_blocks (list): List of integers specifying the number of LightweightConvBlocks
                               in each encoder/bottleneck stage.
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks

        # Initial convolution to increase channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- Encoder Stages ---
        # Use LightweightConvBlock with stride for downsampling
        # Add EfficientCSAttention after each stage for refinement
        self.encoder_stage1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=2) # H/2
        self.attn1 = EfficientCSAttention(base_channels)

        self.encoder_stage2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2) # H/4
        self.attn2 = EfficientCSAttention(base_channels * 2)

        self.encoder_stage3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks[2], stride=2) # H/8
        self.attn3 = EfficientCSAttention(base_channels * 4)

        # Bottleneck (can be just a few more blocks or attention)
        self.bottleneck = self._make_layer(base_channels * 4, base_channels * 8, num_blocks[3], stride=2) # H/16
        self.attn4 = EfficientCSAttention(base_channels * 8)


        # --- Decoder Stages ---
        # Use ConvTranspose2d for upsampling and combine with skip connections
        # Add EfficientCSAttention after combining features
        self.decoder_stage3 = self._make_decoder_layer(base_channels * 8, base_channels * 4, base_channels * 4) # H/8
        self.attn5 = EfficientCSAttention(base_channels * 4)

        self.decoder_stage2 = self._make_decoder_layer(base_channels * 4, base_channels * 2, base_channels * 2) # H/4
        self.attn6 = EfficientCSAttention(base_channels * 2)

        self.decoder_stage1 = self._make_decoder_layer(base_channels * 2, base_channels, base_channels) # H/2
        self.attn7 = EfficientCSAttention(base_channels)


        # Final output layer
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2), # Upsample H/2 to H
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1), # Output 3 channels
            nn.Sigmoid() # Output values between 0 and 1
        )

        # --- Loss Functions (Can reuse from original AD_NET) ---
        self.PerceptualLoss = PerceptualLoss() # Assuming this is defined in .utils

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """ Helper function to create an encoder/bottleneck stage """
        layers = []
        # First block handles potential stride and channel change
        layers.append(LightweightConvBlock(in_channels, out_channels, stride=stride))
        # Remaining blocks have stride 1 and no channel change
        for _ in range(1, num_blocks):
            layers.append(LightweightConvBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _make_decoder_layer(self, in_channels, skip_channels, out_channels):
        """ Helper function to create a decoder stage """
        # in_channels from previous decoder stage
        # skip_channels from encoder skip connection
        # out_channels for the current decoder stage output
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2), # Upsample
            nn.ReLU(inplace=True),
            # Combine with skip connection - needs to handle channel dimension
            # The next layer will take (out_channels + skip_channels) as input
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1), # Combine and refine
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        """
        Forward pass of the dehazing network.

        Args:
            x (torch.Tensor): Input hazy image tensor (B, C, H, W), range [0, 1].

        Returns:
            torch.Tensor: Dehazed image tensor (B, C, H, W), range [0, 1].
        """
        # x is the hazy image (B, C, H, W) - assuming already normalized to [0, 1]

        x0 = self.initial_conv(x) # H

        # Encoder
        enc1 = self.encoder_stage1(x0) # H/2
        enc1_att = self.attn1(enc1)

        enc2 = self.encoder_stage2(enc1_att) # H/4
        enc2_att = self.attn2(enc2)

        enc3 = self.encoder_stage3(enc2_att) # H/8
        enc3_att = self.attn3(enc3)

        # Bottleneck
        enc4 = self.bottleneck(enc3_att) # H/16
        enc4_att = self.attn4(enc4)

        # Decoder
        # Upsample enc4_att and prepare for concatenation with enc3_att
        dec3_up = self.decoder_stage3[0](enc4_att) # Upsample H/16 to H/8
        # Ensure dimensions match before concatenation
        # ConvTranspose2d should handle this if input H,W are even.
        # If not, interpolation might be safer, or handle padding in ConvTranspose2d
        # Here we use interpolate as a robust fallback
        if dec3_up.shape[2:] != enc3_att.shape[2:]:
             dec3_up = F.interpolate(dec3_up, size=enc3_att.shape[2:], mode='bilinear', align_corners=False)

        # Combine with skip connection and apply remaining decoder layers + attention
        dec3_combined = torch.cat([dec3_up, enc3_att], dim=1) # Skip connection
        dec3 = self.decoder_stage3[1:](dec3_combined) # Apply remaining conv
        dec3_att = self.attn5(dec3)


        # Upsample dec3_att and prepare for concatenation with enc2_att
        dec2_up = self.decoder_stage2[0](dec3_att) # Upsample H/8 to H/4
        if dec2_up.shape[2:] != enc2_att.shape[2:]:
             dec2_up = F.interpolate(dec2_up, size=enc2_att.shape[2:], mode='bilinear', align_corners=False)
        dec2_combined = torch.cat([dec2_up, enc2_att], dim=1) # Skip connection
        dec2 = self.decoder_stage2[1:](dec2_combined) # Apply remaining conv
        dec2_att = self.attn6(dec2)

        # Upsample dec2_att and prepare for concatenation with enc1_att
        dec1_up = self.decoder_stage1[0](dec2_att) # Upsample H/4 to H/2
        if dec1_up.shape[2:] != enc1_att.shape[2:]:
             dec1_up = F.interpolate(dec1_up, size=enc1_att.shape[2:], mode='bilinear', align_corners=False)
        dec1_combined = torch.cat([dec1_up, enc1_att], dim=1) # Skip connection
        dec1 = self.decoder_stage1[1:](dec1_combined) # Apply remaining conv
        dec1_att = self.attn7(dec1)

        # Final Output
        dehaze_img = self.output_conv(dec1_att) # Upsample H/2 to H

        return dehaze_img.clamp(0, 1) # Ensure output is in valid range

    def forward_loss(self, haze_img, clean_img):
        """
        Calculates the loss for training.

        Args:
            haze_img (torch.Tensor): Input hazy image tensor (B, C, H, W), range [0, 1].
            clean_img (torch.Tensor): Ground truth clean image tensor (B, C, H, W), range [0, 1].

        Returns:
            dict: A dictionary containing different loss components and the total loss.
        """
        dehaze_img = self(haze_img) # Get the dehazed output

        # Calculate losses (same as original AD_NET)
        l1 = F.l1_loss(dehaze_img, clean_img)
        perceptual = self.PerceptualLoss(dehaze_img, clean_img)
        tv = total_variation_loss(dehaze_img)

        # Adjust loss weights as needed
        # Example weights: L1 is primary, Perceptual helps with visual quality, TV smooths
        loss_dict = {
            'l1_loss': l1,
            'perceptual_loss': perceptual,
            'tv_loss': tv,
            'total_loss': l1 + 0.5 * perceptual + 0.1 * tv # Example weights
        }
        return loss_dict

