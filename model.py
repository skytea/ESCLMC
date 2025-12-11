from ConvNeXt import *
from Transformer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import numpy as np


class ESCLMC(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 dropout=0.2,
                 ):
        super().__init__()

        vision_heads = vision_width // 64
        self.VisionTransformer = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.ConvNeXt = ConvNeXt(
            depths=[3, 3, 9, 3],
            dims=[128, 256, 512, 1024],
            embed_dim=embed_dim,
            drop_path_rate=dropout,
        )

        self.radio_linear = nn.Sequential(
            # nn.Identity(),
            nn.Linear(96, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.relu = nn.ReLU()
        self.ebed_1 = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)
        self.ebed_0 = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)

    def forward(self, image, radio):
        image = image.type(self.VisionTransformer.conv1.weight.dtype)
        batch, instance, height, width = image.size()
        image = image.view(-1, 1, height, width)
        VisionTransformer_features = self.VisionTransformer(image)
        VisionTransformer_features, _ = torch.max(VisionTransformer_features, dim=0, keepdim=True)
        ConvNeXt_features = self.ConvNeXt(image)

        # normalized features
        VisionTransformer_features = VisionTransformer_features / VisionTransformer_features.norm(dim=1, keepdim=True)
        ConvNeXt_features = ConvNeXt_features / ConvNeXt_features.norm(dim=1, keepdim=True)

        if radio is not None:
            radio = self.radio_linear(radio)
            radio = radio / radio.norm(dim=1, keepdim=True)

        ebed_1 = self.ebed_1 / self.ebed_1.norm(dim=1, keepdim=True)
        ebed_0 = self.ebed_0 / self.ebed_0.norm(dim=1, keepdim=True)

        return VisionTransformer_features, ConvNeXt_features, radio, ebed_1, ebed_0
