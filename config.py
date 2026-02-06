import os
from dataclasses import dataclass


@dataclass
class PathConfig:
    """Configuration class for all file paths"""

    # Training data paths
    train_lr_dir = "dataset/lr_outputlarge384"
    train_hr_dir = "data/gtlarge384"

    # Validation data paths
    val_lr_dir = "dataset/lrlarge384"
    val_hr_dir = "dataset/gtlargetest384"

    # Testing data paths
    test_lr_dir = "dataset/justtest"
    test_hr_dir = "dataset/justtestHR"

    # Model saving paths
    model_checkpoint_dir = "checkpoints"
    pretrained_model_path = "checkpoint_epoch_1077.pth"
    final_model_path = "final_unified_model.pth"

    # Test results path
    test_results_dir = "test_results"

    # Logging path
    log_dir = "logs"

    def __post_init__(self):
        """Create directories if they don't exist"""
        directories = [
            self.model_checkpoint_dir,
            self.test_results_dir,
            self.log_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"✓ Directories created/verified")
        print(f"  - Model checkpoints: {self.model_checkpoint_dir}")
        print(f"  - Test results: {self.test_results_dir}")
        print(f"  - Logs: {self.log_dir}")


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""

    # Encoder configuration
    encoder_in_channels = 3
    encoder_num_res_blocks = 2
    encoder_kernel_size = 3
    encoder_channel_size = 64

    # Fusion module configuration
    fusion_in_channels = 64
    fusion_num_hidden_layers = 2
    fusion_hidden_channels = [64, 64]
    fusion_kernel_sizes = [3, 3]
    fusion_spatial_stream_type = "multi_scale"

    # Decoder configuration
    decoder_in_channels = 64
    decoder_out_channels = 64
    decoder_kernel_size = 4
    decoder_stride = 2
    decoder_padding = 1
    decoder_final_in_channels = 64
    decoder_final_out_channels = 3
    decoder_final_kernel_size = 3

    # Attention predictor configuration
    alpha_predictor_input_channels = 64
    alpha_predictor_hidden_channels = 64
    alpha_predictor_num_frames = 16

    # Feature enhancement module configuration
    feature_enhancement_in_channels = 64
    feature_enhancement_hidden_channels = 64
    feature_enhancement_num_res_blocks = 4


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""

    # Training parameters
    num_epochs = 1500
    learning_rate = 0.000015
    batch_size = 1
    start_epoch = 0

    # Loss weights
    loss_alpha = 0.03  # PSNR weight
    loss_beta = 2.5  # SSIM weight
    loss_gamma = 0.62  # LPIPS weight

    # Learning rate adjustment
    patience = 3
    lr_reduction_factor = 0.95

    # Gradient clipping
    max_grad_norm = 1.0

    # Data parameters
    low_res_size = (96, 96)
    high_res_size = (384, 384)
    n_frames = 16

    # Image cropping
    crop_pixels = 4


@dataclass
class Config:
    """Main configuration class"""

    paths: PathConfig = PathConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            "encoder": {
                "in_channels": self.model.encoder_in_channels,
                "num_res_blocks": self.model.encoder_num_res_blocks,
                "kernel_size": self.model.encoder_kernel_size,
                "channel_size": self.model.encoder_channel_size
            },
            "fusion": {
                "in_channels": self.model.fusion_in_channels,
                "num_hidden_layers": self.model.fusion_num_hidden_layers,
                "hidden_channels": self.model.fusion_hidden_channels,
                "kernel_sizes": self.model.fusion_kernel_sizes,
                "spatial_stream_type": self.model.fusion_spatial_stream_type
            },
            "decoder": {
                "deconv": {
                    "in_channels": self.model.decoder_in_channels,
                    "out_channels": self.model.decoder_out_channels,
                    "kernel_size": self.model.decoder_kernel_size,
                    "stride": self.model.decoder_stride,
                    "padding": self.model.decoder_padding
                },
                "final": {
                    "in_channels": self.model.decoder_final_in_channels,
                    "out_channels": self.model.decoder_final_out_channels,
                    "kernel_size": self.model.decoder_final_kernel_size,
                    "padding": self.model.decoder_final_kernel_size // 2
                }
            }
        }


# Global configuration instance
config = Config()