import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedAlphaPredictor(nn.Module):
    """Improved attention prediction module combining spatial-temporal information"""

    def __init__(self, input_channels=64, hidden_channels=64, num_frames=16):
        super().__init__()
        self.num_frames = num_frames

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_channels,
            num_heads=4,
            batch_first=True
        )

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.quality_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.PReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape

        spatial_features = []
        for t in range(num_frames):
            frame_feat = x[:, t]
            encoded = self.spatial_encoder(frame_feat)
            encoded = encoded.view(batch_size, -1)
            spatial_features.append(encoded)

        spatial_features = torch.stack(spatial_features, dim=1)

        attended_features, attention_weights = self.temporal_attention(
            spatial_features, spatial_features, spatial_features
        )

        quality_scores = []
        for t in range(num_frames):
            quality = self.quality_net(attended_features[:, t])
            quality_scores.append(quality)

        quality_scores = torch.stack(quality_scores, dim=1).squeeze(-1)

        temporal_weights = attended_features.mean(-1)
        final_alphas = F.softmax(temporal_weights * quality_scores, dim=1)

        return final_alphas, quality_scores