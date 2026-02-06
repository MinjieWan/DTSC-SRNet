import torch
import torch.nn as nn
from models.encoder_decoder import EncoderUnit, FeatureEnhancementModule, Decoder
from models.dual_stream import DualStreamFusionModule
from models.attention_predictor import EnhancedAlphaPredictor


class MISRConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.unit1 = EncoderUnit(
            in_channels=config["encoder"]["in_channels"],
            num_res_layers=config["encoder"]["num_res_blocks"],
            kernel_size=config["encoder"]["kernel_size"],
            channel_size=config["encoder"]["channel_size"]
        )

        self.fuse = DualStreamFusionModule(config["fusion"])

        self.feature_enhancement = FeatureEnhancementModule(
            in_channels=config["fusion"]["hidden_channels"][-1],
            hidden_channels=64,
            num_res_blocks=4
        )

        self.decode = Decoder(config["decoder"])

        self.unit2 = EncoderUnit(
            in_channels=config["encoder"]["channel_size"] * 2,
            num_res_layers=config["encoder"]["num_res_blocks"],
            kernel_size=config["encoder"]["kernel_size"],
            channel_size=config["encoder"]["channel_size"]
        )

        self.alpha_predictor = EnhancedAlphaPredictor(
            input_channels=config["encoder"]["channel_size"],
            num_frames=16
        )

    def forward(self, lrs, alphas=None):
        batch_size, num_frames, channels, height, width = lrs.shape

        lrs_flat = lrs.view(batch_size * num_frames, channels, height, width)
        lrs_encoded = self.unit1(lrs_flat)
        lrs_encoded = lrs_encoded.view(batch_size, num_frames, -1, height, width)

        if alphas is not None:
            alphas = alphas.view(-1, num_frames, 1, 1, 1)
        else:
            learned_alphas, quality_scores = self.alpha_predictor(lrs_encoded)
            alphas = learned_alphas.view(-1, num_frames, 1, 1, 1)

        refs, _ = torch.median(lrs[:, :16], 1)
        refs_encoded = self.unit1(refs)
        refs_encoded = refs_encoded.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        out = torch.cat([lrs_encoded, refs_encoded], 2)
        out = out.view(batch_size * num_frames, -1, height, width)
        out = self.unit2(out)
        out = out.view(batch_size, num_frames, -1, height, width)

        out, _ = self.fuse(out, alphas.squeeze())

        enhanced_features = self.feature_enhancement(out)

        final_output = self.decode(enhanced_features)

        if alphas is None:
            return final_output, learned_alphas
        else:
            return final_output, alphas.squeeze()