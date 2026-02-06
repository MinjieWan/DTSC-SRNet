import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convlstm_units import ResidualConvLSTMUnit
from models.encoder_decoder import ResidualBlock

class DynamicReceptiveFieldSpatialStream(nn.Module):
    """Spatial feature extraction with dynamic receptive field"""

    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()

        self.deform_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

        self.pyramid_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(4),
            nn.AdaptiveAvgPool2d(8)
        ])

        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.PReLU()
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape

        frame_outputs = []
        for t in range(num_frames):
            frame_feat = x[:, t]

            deform_feat = self.deform_conv(frame_feat)

            pyramid_features = []
            for pool in self.pyramid_pool:
                pooled = pool(deform_feat)
                if pooled.shape[2:] != (height, width):
                    pooled = F.interpolate(pooled, size=(height, width),
                                           mode='bilinear', align_corners=False)
                pyramid_features.append(pooled)

            pyramid_concat = torch.cat(pyramid_features, dim=1)
            fused_pyramid = self.pyramid_fusion(pyramid_concat)

            frame_outputs.append(fused_pyramid)

        return torch.stack(frame_outputs, dim=1)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of spatial-temporal features"""

    def __init__(self, channels):
        super().__init__()

        self.temporal_to_spatial_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=2, batch_first=True
        )
        self.spatial_to_temporal_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=2, batch_first=True
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )

    def forward(self, temporal_feat, spatial_feat):
        B, C, H, W = temporal_feat.shape

        temporal_seq = temporal_feat.view(B, C, -1).permute(0, 2, 1)
        spatial_seq = spatial_feat.view(B, C, -1).permute(0, 2, 1)

        temporal_enhanced, _ = self.temporal_to_spatial_attn(
            temporal_seq, spatial_seq, spatial_seq
        )
        spatial_enhanced, _ = self.spatial_to_temporal_attn(
            spatial_seq, temporal_seq, temporal_seq
        )

        temporal_enhanced = temporal_enhanced.permute(0, 2, 1).view(B, C, H, W)
        spatial_enhanced = spatial_enhanced.permute(0, 2, 1).view(B, C, H, W)

        concat_global = torch.cat([
            F.adaptive_avg_pool2d(temporal_enhanced, 1).view(B, C),
            F.adaptive_avg_pool2d(spatial_enhanced, 1).view(B, C)
        ], dim=1)

        fusion_weight = self.fusion_gate(concat_global).view(B, C, 1, 1)

        fused = temporal_enhanced * fusion_weight + spatial_enhanced * (1 - fusion_weight)

        return fused


class GatedFusionModule(nn.Module):
    """Gated mechanism for dual-stream information fusion"""

    def __init__(self, temporal_channels, spatial_channels, out_channels):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Conv2d(temporal_channels + spatial_channels, out_channels // 4, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        self.temporal_transform = nn.Conv2d(temporal_channels, out_channels, 1)
        self.spatial_transform = nn.Conv2d(spatial_channels, out_channels, 1)

        self.residual_conv = nn.Conv2d(temporal_channels, out_channels,
                                       1) if temporal_channels != out_channels else nn.Identity()

    def forward(self, temporal_feat, spatial_feat):
        temporal_trans = self.temporal_transform(temporal_feat)
        spatial_trans = self.spatial_transform(spatial_feat)

        concat_features = torch.cat([temporal_feat, spatial_feat], dim=1)
        gates = self.gate_net(concat_features)

        temporal_gate = gates[:, 0:1]
        spatial_gate = gates[:, 1:2]

        fused = temporal_trans * temporal_gate + spatial_trans * spatial_gate

        residual = self.residual_conv(temporal_feat)
        output = fused + residual

        return output


class MultiScaleTemporalStream(nn.Module):
    """Multi-scale temporal feature extraction"""

    def __init__(self, config):
        super().__init__()
        self.num_layers = config["num_hidden_layers"]
        hidden_channels = config["hidden_channels"]

        self.coarse_lstm = ResidualConvLSTMUnit(
            config["in_channels"], hidden_channels[0], kernel_size=5
        )
        self.medium_lstm = ResidualConvLSTMUnit(
            config["in_channels"], hidden_channels[0], kernel_size=3
        )
        self.fine_lstm = ResidualConvLSTMUnit(
            config["in_channels"], hidden_channels[0], kernel_size=1
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[0] * 3, hidden_channels[0], 1),
            nn.PReLU(),
            ResidualBlock(hidden_channels[0], 3)  # 这里使用了ResidualBlock
        )

    def forward(self, x, alphas, states=None):
        batch_size, num_frames, channels, height, width = x.shape

        coarse_features, medium_features, fine_features = [], [], []
        coarse_state, medium_state, fine_state = states if states else [None, None, None]

        for t in range(num_frames):
            frame_input = x[:, t]

            h_coarse, coarse_state = self.coarse_lstm(frame_input, coarse_state)
            coarse_features.append(h_coarse)

            h_medium, medium_state = self.medium_lstm(frame_input, medium_state)
            medium_features.append(h_medium)

            h_fine, fine_state = self.fine_lstm(frame_input, fine_state)
            fine_features.append(h_fine)

        fused_features = []
        for t in range(num_frames):
            multi_scale = torch.cat([
                coarse_features[t],
                medium_features[t],
                fine_features[t]
            ], dim=1)
            fused = self.fusion_conv(multi_scale)
            fused_features.append(fused)

        fused_features = torch.stack(fused_features, dim=1)
        alphas_expanded = alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        temporal_fused = torch.sum(fused_features * alphas_expanded, dim=1)

        return temporal_fused, [coarse_state, medium_state, fine_state]


class DualStreamFusionModule(nn.Module):
    """Enhanced dual-stream fusion module"""

    def __init__(self, config):
        super().__init__()

        self.temporal_stream = MultiScaleTemporalStream(config)

        self.spatial_stream = DynamicReceptiveFieldSpatialStream(
            in_channels=config["in_channels"],
            out_channels=config["hidden_channels"][-1]
        )

        self.fusion_strategy = nn.ModuleDict({
            'gated': GatedFusionModule(
                config["hidden_channels"][-1],
                config["hidden_channels"][-1],
                config["hidden_channels"][-1]
            ),
            'cross_attention': CrossAttentionFusion(config["hidden_channels"][-1])
        })

        self.fusion_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(config["hidden_channels"][-1] * 2, config["hidden_channels"][-1] // 4, 1),
            nn.PReLU(),
            nn.Conv2d(config["hidden_channels"][-1] // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        self.refinement = nn.Sequential(
            ResidualBlock(config["hidden_channels"][-1], 3),
            ResidualBlock(config["hidden_channels"][-1], 3),
            nn.Conv2d(config["hidden_channels"][-1], config["hidden_channels"][-1], 3, padding=1)
        )

    def forward(self, x, alphas, states=None):
        temporal_features, new_temporal_states = self.temporal_stream(x, alphas, states)

        spatial_features = self.spatial_stream(x)
        spatial_fused = torch.sum(spatial_features * alphas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=1)

        gated_fusion = self.fusion_strategy['gated'](temporal_features, spatial_fused)
        attention_fusion = self.fusion_strategy['cross_attention'](temporal_features, spatial_fused)

        fusion_input = torch.cat([temporal_features, spatial_fused], dim=1)
        fusion_weights = self.fusion_selector(fusion_input)

        w_gated, w_attention = fusion_weights[:, 0:1], fusion_weights[:, 1:2]
        combined_fusion = gated_fusion * w_gated + attention_fusion * w_attention

        final_output = self.refinement(combined_fusion)

        return final_output, new_temporal_states