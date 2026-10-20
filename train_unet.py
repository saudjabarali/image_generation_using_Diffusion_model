import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

device = torch.device("cpu")

torch.set_num_threads(8)

T = 100
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


# --------------------------------------------------
# 2. Load MNIST
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# --------------------------------------------------
# 3. Define the noise schedule
# --------------------------------------------------

import math

# Cosine noise schedule
steps = torch.arange(T + 1, dtype=torch.float32)

alpha_bars_all = torch.cos(
    ((steps / T + 0.008) / 1.008) * math.pi / 2
) ** 2

alpha_bars_all = (
    alpha_bars_all / alpha_bars_all[0]
)

betas = 1 - (
    alpha_bars_all[1:] / alpha_bars_all[:-1]
)

betas = betas.clamp(0.0001, 0.999)

alphas = 1 - betas

alpha_bars = torch.cumprod(alphas, dim=0)

print("First alpha_bar:", alpha_bars[0].item())
print("Last alpha_bar:", alpha_bars[-1].item())


# --------------------------------------------------
# 4. Create a timestep embedding
# --------------------------------------------------

class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.embedding = nn.Embedding(T, dim)

    def forward(self, t):
        return self.embedding(t)


# --------------------------------------------------
# 5. Define a small convolutional U-Net
# --------------------------------------------------

class MiniUNet(nn.Module):

    def __init__(self):
        super().__init__()

        time_dim = 32

        self.time_embedding = TimeEmbedding(time_dim)

        # Encoder
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

        # Bottleneck
        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1
        )

        # Decoder
        self.up = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.conv4 = nn.Conv2d(
            64, 32, kernel_size=3, padding=1
        )

        self.output = nn.Conv2d(
            32, 1, kernel_size=3, padding=1
        )

        self.time_to_32 = nn.Linear(time_dim, 32)
        self.time_to_64 = nn.Linear(time_dim, 64)

    def forward(self, x, t):

        # Embed the timestep
        time_features = self.time_embedding(t)

        # Encoder
        x1 = F.relu(self.conv1(x))

        x2 = self.conv2(x1)

        x2 = x2 + self.time_to_64(
            time_features
        )[:, :, None, None]

        x2 = F.relu(x2)

        # Bottleneck
        x3 = F.relu(self.conv3(x2))

        # Upsampling
        x4 = self.up(x3)

        x4 = x4 + self.time_to_32(
            time_features
        )[:, :, None, None]

        # Skip connection
        x4 = torch.cat([x4, x1], dim=1)

        x4 = F.relu(self.conv4(x4))

        # Predict noise
        return self.output(x4)


# --------------------------------------------------
# 6. Initialize model and optimizer
# --------------------------------------------------

model = MiniUNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for images, _ in dataloader:

        images = images.to(device)

        batch_size = images.size(0)

        # Random timestep for each image
        t = torch.randint(
            0, T, (batch_size,)
        )

        # Random Gaussian noise
        noise = torch.randn_like(images)

        # Select cumulative alpha values
        a_bar = alpha_bars[t].view(
            batch_size, 1, 1, 1
        )

        # Add noise to the images
        noisy_images = (
            torch.sqrt(a_bar) * images
            + torch.sqrt(1 - a_bar) * noise
        )

        # Predict noise
        predicted_noise = model(
            noisy_images, t
        )

        # Calculate MSE
        loss = criterion(
            predicted_noise, noise
        )

        # Update parameters
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# 8. Save trained model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "unet_cosine_model.pth"
)

print("U-Net training complete!")
print("Saved: unet_cosine_model.pth")