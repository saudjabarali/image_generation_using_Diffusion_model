import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

device = torch.device("cpu")

T = 100


# --------------------------------------------------
# 2. Noise schedule
# --------------------------------------------------

betas = torch.linspace(0.0001, 0.02, T)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(alphas, dim=0)

print("First alpha_bar:", alpha_bars[0].item())
print("Last alpha_bar:", alpha_bars[-1].item())

print(
    "Original signal remaining at final timestep:",
    alpha_bars[-1].item()
)

print(
    "Noise contribution at final timestep:",
    (1 - alpha_bars[-1]).item()
)
# --------------------------------------------------
# 3. Timestep embedding
# --------------------------------------------------

class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.embedding = nn.Embedding(T, dim)

    def forward(self, t):
        return self.embedding(t)


# --------------------------------------------------
# 4. Recreate the trained U-Net
# --------------------------------------------------

class MiniUNet(nn.Module):

    def __init__(self):
        super().__init__()

        time_dim = 32

        self.time_embedding = TimeEmbedding(time_dim)

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1
        )

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

        time_features = self.time_embedding(t)

        x1 = F.relu(self.conv1(x))

        x2 = self.conv2(x1)

        x2 = x2 + self.time_to_64(
            time_features
        )[:, :, None, None]

        x2 = F.relu(x2)

        x3 = F.relu(self.conv3(x2))

        x4 = self.up(x3)

        x4 = x4 + self.time_to_32(
            time_features
        )[:, :, None, None]

        x4 = torch.cat([x4, x1], dim=1)

        x4 = F.relu(self.conv4(x4))

        return self.output(x4)


# --------------------------------------------------
# 5. Load trained model
# --------------------------------------------------

model = MiniUNet().to(device)

model.load_state_dict(
    torch.load(
        "unet_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Trained U-Net loaded successfully!")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load unseen MNIST test images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=True
)

images, _ = next(iter(test_loader))
images = images.to(device)

batch_size = images.size(0)

print("\nNoise-prediction diagnostics:")

for t_value in [0, 25, 50, 75, 99]:

    t = torch.full(
        (batch_size,),
        t_value,
        dtype=torch.long
    )

    noise = torch.randn_like(images)

    a_bar = alpha_bars[t].view(
        batch_size, 1, 1, 1
    )

    noisy_images = (
        torch.sqrt(a_bar) * images
        + torch.sqrt(1 - a_bar) * noise
    )

    predicted_noise = model(
        noisy_images,
        t
    )

    model_mse = torch.mean(
        (predicted_noise - noise) ** 2
    )

    baseline_mse = torch.mean(noise ** 2)

    print(
        f"t={t_value:2d} | "
        f"Model MSE={model_mse.item():.4f} | "
        f"Baseline MSE={baseline_mse.item():.4f}"
    )

# --------------------------------------------------
# 6. Generate images
# --------------------------------------------------

@torch.no_grad()
def generate_images(num_images=16):

    # Start with random Gaussian noise
    x = torch.randn(
        num_images, 1, 28, 28,
        device=device
    )

    for t in reversed(range(T)):

        timestep = torch.full(
            (num_images,),
            t,
            device=device,
            dtype=torch.long
        )

        # Predict the noise
        predicted_noise = model(x, timestep)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        # Estimate the previous state
        x = (
            x - (
                beta_t / torch.sqrt(1 - alpha_bar_t)
            ) * predicted_noise
        ) / torch.sqrt(alpha_t)

        # Add posterior sampling noise
        if t > 0:

            alpha_bar_prev = alpha_bars[t - 1]

            posterior_variance = (
                beta_t * (1 - alpha_bar_prev)
                / (1 - alpha_bar_t)
            )

            x = x + torch.sqrt(
                posterior_variance
            ) * torch.randn_like(x)

    # Convert normalized values to displayable pixels
    x = ((x + 1) / 2).clamp(0, 1)

    # Display generated images
    fig, axes = plt.subplots(
        4, 4, figsize=(6, 6)
    )

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            x[i, 0].cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


generate_images()