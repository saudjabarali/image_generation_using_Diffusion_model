import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


device = torch.device("cpu")
T = 100


# Recreate the training noise schedule
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# Recreate the trained model architecture
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Embedding(T, dim)

    def forward(self, t):
        return self.embedding(t)


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


# Load trained weights
model = MiniUNet().to(device)

model.load_state_dict(
    torch.load(
        "unet_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# Load MNIST test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image, label = dataset[0]

image = image.unsqueeze(0)


# Inspect multiple noise levels
timesteps = [0, 25, 50, 75, 99]

fig, axes = plt.subplots(
    3, len(timesteps),
    figsize=(12, 7)
)

with torch.no_grad():

    for col, t_value in enumerate(timesteps):

        t = torch.tensor([t_value])

        noise = torch.randn_like(image)

        a_bar = alpha_bars[t_value]

        noisy_image = (
            torch.sqrt(a_bar) * image
            + torch.sqrt(1 - a_bar) * noise
        )

        predicted_noise = model(noisy_image, t)

        estimated_clean = (
            noisy_image
            - torch.sqrt(1 - a_bar) * predicted_noise
        ) / torch.sqrt(a_bar)

        # Original image
        axes[0, col].imshow(
            ((image[0, 0] + 1) / 2).clamp(0, 1),
            cmap="gray"
        )

        # Noisy image
        axes[1, col].imshow(
            ((noisy_image[0, 0] + 1) / 2).clamp(0, 1),
            cmap="gray"
        )

        # Estimated clean image
        axes[2, col].imshow(
            ((estimated_clean[0, 0] + 1) / 2).clamp(0, 1),
            cmap="gray"
        )

        axes[0, col].set_title(f"t={t_value}")

        for row in range(3):
            axes[row, col].axis("off")


axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Noisy")
axes[2, 0].set_ylabel("Estimated clean")

plt.tight_layout()
plt.show()

print("Actual MNIST digit label:", label)