import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb
from left_right_model import HemisphereStream, DualStream
import cv2, random
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms.functional import resize
import av
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def create_model(rank):
    model = DualStream().to(rank)
    model = DDP(model, device_ids=[rank])
    return model

def setup_transforms():
    return transforms.Compose([
        transforms.Resize((90, 120)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(25, 25), sigma=(30, 30)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(contrast=0.8),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Ensure that all tensors and models are moved to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for computation.")

# Move models to the appropriate device
# Assuming models are defined in this file or imported, they should be moved to the device.
# Example: model.to(device)

# Ensure all tensors created in this file are automatically on the right device
# Example of creating a tensor on the device: tensor = torch.tensor(data, device=device)

config = {
    "data_dir": '../left-right-model/splice',
    "batch_size": 22,
    "learning_rate": 1e-4,
    "epochs": 100,
    "num_workers": 8,
    "pin_memory": True,
    "drop_last": True,
    "world_size": torch.cuda.device_count(),  # Number of GPUs available
}


def save_checkpoint(model, path):
    """
    Save the model checkpoint.

    Args:
    - model: The model to save.
    - path: Path where the checkpoint will be saved.
    """
    torch.save(model.state_dict(), path)

def log_sampled_frames(frames, num_seq=1, seq_len=30, resize_shape=(90, 120)):
    """
    Log a grid of sampled frames from a video sequence to Weights & Biases (wandb).

    Args:
    - frames (torch.Tensor): A tensor of video frames of shape (num_seq, seq_len, C, H, W).
    - num_seq (int): Number of sequences sampled from the video.
    - seq_len (int): Number of frames in each sequence.
    - resize_shape (tuple): Resize shape for each frame, for consistent grid display.

    Raises:
    - ValueError: If the input tensor does not match the expected shape.
    """

    # Validate input tensor shape
    if not isinstance(frames, torch.Tensor) or len(frames.shape) != 5:
        raise ValueError("Frames must be a 5D tensor of shape (num_seq, seq_len, C, H, W).")
    if frames.shape[0] < num_seq or frames.shape[1] < seq_len:
        raise ValueError("Frames tensor does not have enough sequences or frames per sequence.")

    # Select the first frame from each sequence for simplicity
    selected_frames = frames[:, 0]  # This selects the first frame in each sequence


    # Resize frames for consistent display
    selected_frames_resized = torch.stack([resize(frame, resize_shape) for frame in selected_frames])

    # Create a grid of images
    frame_grid = make_grid(selected_frames_resized, nrow=num_seq, normalize=True)

    # Convert the tensor grid to a PIL image
    grid_image = to_pil_image(frame_grid)

    # Log the grid image to wandb
    wandb.log({"sampled_frames": [wandb.Image(grid_image, caption="Sampled Frames")]})

class TheDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_percentage=0.20):
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mp4')]
        random.shuffle(self.video_files)
        num_files_to_use = int(len(self.video_files) * use_percentage)
        self.video_files = self.video_files[:num_files_to_use]

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_frames = read_video_frames(video_path, self.transform, seq_len=30, num_seq=1)
        return {'video': video_frames}

def read_video_frames(video_path, transform=None, num_seq=1, seq_len=30):
    container = av.open(video_path)
    stream = container.streams.video[0]

    frames = []
    max_start_index = max(0, stream.frames - (num_seq * seq_len))
    start_index = random.randint(0, min(max_start_index, 210))
    end_index = start_index + num_seq * seq_len

    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= start_index and frame_count < end_index:
            img = frame.to_image()  # Convert to PIL Image
            if transform:
                img = transform(img)
            frames.append(img)
        frame_count += 1
        if frame_count >= end_index:
            break

    container.close()

    # Check if frames are already tensors, if not convert them
    if not isinstance(frames[0], torch.Tensor):
        frames = [to_tensor(frame) for frame in frames]

    # Ensure we have the correct number of frames
    if len(frames) != num_seq * seq_len:
        raise ValueError(f"Expected {num_seq * seq_len} frames, but got {len(frames)}")

    frames_tensor = torch.stack(frames, dim=0).view(num_seq, seq_len, 3, *frames[0].shape[1:])

    return frames_tensor

def variance_loss(x, rank, eps=1e-4):
    # Calculate the standard deviation along the batch dimension
    std = torch.sqrt(x.var(dim=0) + eps)
    # Compute the variance loss
    loss = torch.mean(F.relu(1 - std))
    if rank == 0:
        wandb.log({"variance_loss": loss.item()})
    return loss

def invariance_loss(z1, z2, rank):
    # Compute the mean squared error between the two sets of representations
    loss = F.mse_loss(z1, z2)
    if rank == 0:
        wandb.log({"invariance_loss": loss.item()})
    return loss

def off_diagonal(x):
        # Return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def covariance_loss(x, rank):
        x = x - x.mean(dim=0)
        #normval = torch.norm(x)
        #x = x/normval # testline
        cov_matrix = (x.T @ x) / (x.size(0) - 1)
        off_diag = off_diagonal(cov_matrix)
        if rank == 0:
            wandb.log({"covariance_loss": off_diag.pow(2).sum().item() / x.size(1)})
        return off_diag.pow(2).sum() / x.size(1)

def vicreg_loss(z1, z2, rank, lambda_var=1, mu_inv=1, nu_cov=0.04):
    var_loss = variance_loss(z1, rank) + variance_loss(z2, rank)
    inv_loss = invariance_loss(z1, z2, rank)
    cov_loss = covariance_loss(z1, rank) + covariance_loss(z2, rank)
    loss = mu_inv * inv_loss + lambda_var * var_loss + nu_cov * cov_loss
    return loss

def train(rank, world_size):
    setup(rank, world_size)
    model = create_model(rank)
    if rank == 0:
        wandb.init(project="left-right", config=config)

    # Setup transformations and data loaders
    transform = transforms.Compose([
        transforms.Resize((90, 120)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(11, 11), sigma=(5, 5)),
        transforms.RandomApply([
            transforms.Grayscale(num_output_channels=3)
        ], p=0.7),
        transforms.ColorJitter(contrast=2.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = TheDataset(root_dir=config['data_dir'], transform=transform)
    train_size = int(0.70 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_filenames = [full_dataset.video_files[idx] for idx in train_dataset.indices]
    val_filenames = [full_dataset.video_files[idx] for idx in val_dataset.indices]

    with open('train_files.txt', 'w') as f:
        for filename in train_filenames:
            f.write(f"{filename}\n")

    with open('val_files.txt', 'w') as f:
        for filename in val_filenames:
            f.write(f"{filename}\n")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=config['drop_last'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], sampler=val_sampler, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=config['drop_last'])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            inputs = batch['video'].to(rank)
            optimizer.zero_grad()
            left_output, right_output = model(inputs, epoch)
            
            # Assuming outputs and targets are structured for your specific loss function
            # Adjust the following line according to your model's specific output and target requirements
            loss = vicreg_loss(left_output, right_output, rank)
            
            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            total_loss += loss.item()

            if rank == 0:
                wandb.log({"Step Loss": loss.item(), "Learning Rate": current_lr, "Step": i})
                log_sampled_frames(inputs[0])

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)
        
        if rank == 0:
            print(f"Epoch {epoch+1}, Average Training Loss: {average_loss}")
            wandb.log({"Epoch Average Loss": average_loss, "Epoch": epoch})

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch['video'].to(rank)
                left_output, right_output = model(inputs, epoch)
                loss = vicreg_loss(left_output, right_output, rank)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        if rank == 0:
            print(f"Epoch {epoch+1}, Average Validation Loss: {average_val_loss}")
            wandb.log({"Epoch Average Validation Loss": average_val_loss})

        if (epoch + 1) % 1 == 0 and rank == 0:
            checkpoint_path = f'model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)




