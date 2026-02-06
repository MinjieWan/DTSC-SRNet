import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


class EncoderUnit(nn.Module):
    def __init__(self, in_channels, num_res_layers, kernel_size, channel_size):
        super(EncoderUnit, self).__init__()
        padding = kernel_size // 2
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())
        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_res_layers)]
        self.res_layers = nn.Sequential(*res_layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class FeatureEnhancementModule(nn.Module):
    """Feature enhancement module: enhances features before decoding"""

    def __init__(self, in_channels=64, hidden_channels=64, num_res_blocks=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.PReLU(),
            *[ResidualBlock(hidden_channels, 3) for _ in range(num_res_blocks)]
        )

        self.multi_scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Identity()
            ),
            nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.PReLU(),
                ResidualBlock(hidden_channels, 3)
            ),
            nn.Sequential(
                nn.AvgPool2d(4),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.PReLU(),
                ResidualBlock(hidden_channels, 3)
            )
        ])

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1),
            nn.PReLU(),
            ResidualBlock(hidden_channels, 3),
            ResidualBlock(hidden_channels, 3)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels // 8, 1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels // 8, hidden_channels, 1),
            nn.Sigmoid()
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        encoded = self.encoder(x)

        scale_features = []
        for branch in self.multi_scale_branches:
            feat = branch(encoded)
            if feat.shape[2:] != x.shape[2:]:
                feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            scale_features.append(feat)

        fused = torch.cat(scale_features, dim=1)
        fused = self.feature_fusion(fused)

        attention_weights = self.attention(fused)
        enhanced = fused * attention_weights

        output = x + self.residual_scale * enhanced

        return output


class Decoder(nn.Module):
    def __init__(self, dec_config):
        super(Decoder, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(
                in_channels=dec_config["deconv"]["in_channels"],
                out_channels=dec_config["deconv"]["out_channels"],
                kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.PReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(
                in_channels=dec_config["deconv"]["out_channels"],
                out_channels=dec_config["deconv"]["out_channels"],
                kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.PReLU()
        )
        in_channels = 64
        self.refine_blocks = nn.Sequential(
            ResidualBlock(in_channels, 3),
            ResidualBlock(in_channels, 3),
        )
        self.final = nn.Conv2d(
            in_channels=dec_config["final"]["in_channels"],
            out_channels=dec_config["final"]["out_channels"],
            kernel_size=dec_config["final"]["kernel_size"],
            padding=dec_config["final"]["kernel_size"] // 2,
            padding_mode='reflect'
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.refine_blocks(x)
        x = self.final(x)
        return x