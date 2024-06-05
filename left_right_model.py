import torch
import torch.nn as nn
import torch.optim as optim
from conv_layers import ResBlock
import math
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.init as init
import math

universal_dropout = 0.1
universal_drop_connect = 0.20
eps = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


perspective_transform = transforms.RandomPerspective(distortion_scale=0.2, p=0.3)
def apply_perspective_to_video(video):
    # video shape: (B, N, SL, H, W)
    print("Video input shape:", video.shape)  # Debug: Check the input shape
    transformed_video = []
    for batch in video:
        print("Batch shape:", batch.shape)  # Debug: Check the shape of each batch
        transformed_batch = []
        for frame in batch:
            print("Frame shape before permute:", frame.shape)  # Debug: Check the shape before permute
            # Add a channel dimension since RandomPerspective expects a channel dimension
            frame_with_channel = frame.unsqueeze(1)  # Shape becomes [SL, 1, H, W]
            # Apply the transform to each frame individually
            transformed_frame = torch.stack([perspective_transform(f) for f in frame_with_channel], dim=0)
            # Remove the channel dimension after transformation
            transformed_frame = transformed_frame.squeeze(1)  # Shape back to [SL, H, W]
            transformed_batch.append(transformed_frame)
        if transformed_batch:
            transformed_video.append(torch.stack(transformed_batch))
        else:
            print("No frames transformed in this batch.")
    return torch.stack(transformed_video) if transformed_video else torch.tensor([])

color_jitter_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

def apply_color_jitter_to_video(video):
    # video shape: (B, N, SL, H, W)
    print("Video input shape:", video.shape)  # Debug: Check the input shape
    transformed_video = []
    for batch in video:
        print("Batch shape:", batch.shape)  # Debug: Check the shape of each batch
        transformed_batch = []
        for frame in batch:
            print("Frame shape before transform:", frame.shape)  # Debug: Check the shape before transform
            # Ensure frame has a channel dimension, assuming grayscale (1 channel)
            frame_with_channel = frame.unsqueeze(1)  # Shape becomes (SL, 1, H, W)
            # Apply the color jitter transform to each frame individually with a probability of 0.8
            transformed_frame = torch.stack([
                color_jitter_transform(f) if torch.rand(1).item() < 0.8 else f
                for f in frame_with_channel
            ], dim=0)
            # Ensure the channel dimension is consistent
            transformed_frame = transformed_frame.squeeze(1)  # Shape back to (SL, H, W)
            transformed_batch.append(transformed_frame)
        if transformed_batch:
            transformed_video.append(torch.stack(transformed_batch))
        else:
            print("No frames transformed in this batch.")
    return torch.stack(transformed_video) if transformed_video else torch.tensor([])

gaussian_blur_transform = transforms.GaussianBlur(kernel_size=23)

def apply_gaussian_blur_to_video(video):
    # video shape: (B, N, SL, H, W)
    print("Video input shape:", video.shape)  # Debug: Check the input shape
    transformed_video = []
    for batch in video:
        print("Batch shape:", batch.shape)  # Debug: Check the shape of each batch
        transformed_batch = []
        for frame in batch:
            print("Frame shape before transform:", frame.shape)  # Debug: Check the shape before transform
            # Apply the Gaussian blur transform to each frame individually with a probability of 0.5
            transformed_frame = torch.stack([
                gaussian_blur_transform(f.unsqueeze(0)).squeeze(0) if torch.rand(1).item() < 0.5 else f
                for f in frame
            ], dim=0)
            transformed_batch.append(transformed_frame)
        if transformed_batch:
            transformed_video.append(torch.stack(transformed_batch))
        else:
            print("No frames transformed in this batch.")
    return torch.stack(transformed_video) if transformed_video else torch.tensor([])

color_normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class HemisphereStream(nn.Module):
    def __init__(self, num_blocks=18, slow_temporal_stride=10, fast_temporal_stride=2):
        super(HemisphereStream, self).__init__()
        self.initial_slow_conv = nn.Conv3d(3, 64, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.initial_fast_conv = nn.Conv3d(3, 8, kernel_size=(1, 1, 1), stride=1, padding=0)

        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride

        # Slow pathway
        self.slow_blocks = nn.ModuleList([
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            ResBlock(dim_in=128, dim_out=128, temp_kernel_size=3, stride=1, dim_inner=128),
            ResBlock(dim_in=128, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3), padding=0),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=256, temp_kernel_size=3, stride=1, dim_inner=256),
            ResBlock(dim_in=256, dim_out=512, temp_kernel_size=3, stride=1, dim_inner=512),
            ResBlock(dim_in=512, dim_out=512, temp_kernel_size=3, stride=1, dim_inner=512),
            ResBlock(dim_in=512, dim_out=512, temp_kernel_size=3, stride=1, dim_inner=512)
        ])
        # Fast pathway
        self.fast_blocks = nn.ModuleList([
            #ResBlock(dim_in=fast_channels[i], dim_out=16, temp_kernel_size=3, stride=1, dim_inner=8) for i in range(num_blocks)  # Smaller inner dimension
            ResBlock(dim_in=8, dim_out=8, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=8, dim_out=8, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=8, dim_out=8, temp_kernel_size=3, stride=1, dim_inner=8),
            ResBlock(dim_in=8, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            ResBlock(dim_in=16, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=16, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=16, dim_out=16, temp_kernel_size=3, stride=1, dim_inner=16),
            ResBlock(dim_in=16, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            nn.MaxPool3d(kernel_size=(10, 3, 3), stride=(8, 3, 3), padding=0),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=32, temp_kernel_size=3, stride=1, dim_inner=32),
            ResBlock(dim_in=32, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64),
            ResBlock(dim_in=64, dim_out=64, temp_kernel_size=3, stride=1, dim_inner=64)  
        ])


    def forward(self, x):
        # Apply temporal stride correctly
        slow_input = self.initial_slow_conv(x[:, :, ::self.slow_temporal_stride, :, :])
        fast_input = self.initial_fast_conv(x[:, :, ::self.fast_temporal_stride, :, :])

        for slow_block, fast_block in zip(self.slow_blocks, self.fast_blocks):
            slow_input = slow_block(slow_input)
            fast_input = fast_block(fast_input)
            #print(f"Fast block output shape: {fast_input.shape}")
            #print(f"Slow block output shape: {slow_input.shape}")

        slow_pooled = F.adaptive_avg_pool3d(slow_input, (1, 10, 10))
        fast_pooled = F.adaptive_avg_pool3d(fast_input, (1, 10, 10))

        return slow_pooled, fast_pooled


class DualStream(nn.Module):
    def __init__(self):
        super(DualStream, self).__init__()
        self.shared_hemisphere_stream = HemisphereStream()

        self.slow_flat_num = 51200
        self.fast_flat_num = 6400

        self.dropout = nn.Dropout(universal_dropout)

        self.hemisphere_slow_reduction_mlp = nn.Sequential(
            nn.Linear(self.slow_flat_num, 2048),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(2048, 3188),
            nn.BatchNorm1d(3188),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(3188, 3188),
            nn.BatchNorm1d(3188),
            nn.Linear(3188, 3188)
        )

        self.hemisphere_fast_reduction_mlp = nn.Sequential(
            nn.Linear(self.fast_flat_num, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, 3188),
            nn.BatchNorm1d(3188),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(3188, 3188),
            nn.BatchNorm1d(3188),
            nn.Linear(3188, 3188)
        )

        self.mse_loss = nn.MSELoss()

    def forward(self, x, epoch):
        B, N, SL, C, H, W = x.shape
        x = x.view(B*N, C, SL, H, W)
        if(epoch >= 0):
            left_input = x[:, :, :, :, :W//2]
            right_input = x[:, :, :, :, W//2:]
        else:
             left_input = x[:, :, :, :, :W//2]
             right_input = x[:, :, :, :, :W//2]

             left_input = left_input
             right_input = apply_gaussian_blur_to_video(right_input)
             # Apply transformations to right_input
             #right_input = right_input.permute(0, 2, 1, 3, 4) 
             

        left_slow, left_fast = self.shared_hemisphere_stream(left_input)
        right_slow, right_fast = self.shared_hemisphere_stream(right_input)

        left_fast_flatten = left_fast.view(left_fast.size(0), -1)
        right_fast_flatten = right_fast.view(right_fast.size(0), -1)
        left_slow_flatten = left_slow.view(left_slow.size(0), -1)
        right_slow_flatten = right_slow.view(right_slow.size(0), -1)

        #print(left_slow_flatten.shape)
        #print(left_fast_flatten.shape)

        # Apply reduction MLPs
        left_slow_reduced = self.hemisphere_slow_reduction_mlp(left_slow_flatten)
        right_slow_reduced = self.hemisphere_slow_reduction_mlp(right_slow_flatten)
        # Dynamically create the first Linear layer in the reduction MLPs for fast pathways based on the flattened size

        # Apply reduction MLPs to fast pathways
        left_fast_reduced = self.hemisphere_fast_reduction_mlp(left_fast_flatten)
        right_fast_reduced = self.hemisphere_fast_reduction_mlp(right_fast_flatten)

        left_representation = torch.cat((left_slow_reduced, left_fast_reduced), dim=1)
        right_representation = torch.cat((right_slow_reduced, right_fast_reduced), dim=1)

        left_representation = nn.LayerNorm(left_representation.size()[1:], device=left_representation.device)(left_representation)
        right_representation = nn.LayerNorm(right_representation.size()[1:], device=right_representation.device)(right_representation)

        return left_representation, right_representation
