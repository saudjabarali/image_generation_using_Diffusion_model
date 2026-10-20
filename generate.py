import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

T = 100

# Recreate the same noise schedule used during training
betas = torch.linspace(0.0001, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# Recreate the exact same model architecture
class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = nn.Embedding(T, 32)

        self.network = nn.Sequential(
            nn.Linear(784 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )

    def forward(self, x, t):
        x = x.view(x.size(0), 784)
        time_features = self.time_embedding(t)
        combined = torch.cat([x, time_features], dim=1)
        return self.network(combined)


# Load the previously trained parameters
model = NoisePredictor().to(device)

model.load_state_dict(
    torch.load(
        "diffusion_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Trained model loaded successfully!")

# Diagnostic: evaluate noise prediction on fresh MNIST images
from torchvision import datasets, transforms

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

images, _ = next(iter(
    torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True
    )
))

images = images.to(device)
batch_size = images.size(0)

# Test a range of noise levels
for t_value in [0, 25, 50, 75, 99]:
    t = torch.full(
        (batch_size,),
        t_value,
        device=device,
        dtype=torch.long
    )

    noise = torch.randn_like(images)
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)

    noisy_images = (
        torch.sqrt(a_bar) * images
        + torch.sqrt(1 - a_bar) * noise
    )

    predicted_noise = model(noisy_images, t)
    predicted_noise = predicted_noise.view_as(noise)

    mse = torch.mean(
        (predicted_noise - noise) ** 2
    )

    zero_baseline = torch.mean(noise ** 2)

    print(
        f"t={t_value:2d} | "
        f"Model MSE={mse.item():.4f} | "
        f"Zero-prediction baseline={zero_baseline.item():.4f}"
    )

# Generate images
@torch.no_grad()
def generate_images(num_images=16):
    model.eval()

    # Start from random Gaussian noise
    x = torch.randn(
        num_images, 1, 28, 28,
        device=device
    )
    snapshots = {}

    for t in reversed(range(T)):
        timestep = torch.full(
            (num_images,),
            t,
            device=device,
            dtype=torch.long
        )

        # Predict noise
        predicted_noise = model(x, timestep)
        predicted_noise = predicted_noise.view(
            num_images, 1, 28, 28
        )

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        # DDPM reverse-process mean
        x = (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t))
            * predicted_noise
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

        if t in [99, 75, 50, 25, 0]:
            snapshots[t] = x[0, 0].detach().cpu().clone()


    # Convert normalized image values to [0, 1]
    x = ((x + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    for ax, t in zip(axes, [99, 75, 50, 25, 0]):
        image = ((snapshots[t] + 1) / 2).clamp(0, 1)

        ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t = {t}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


generate_images()