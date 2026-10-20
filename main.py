import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Select the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2. Load MNIST
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
    batch_size=128,
    shuffle=True
)

# 3. Define the diffusion noise schedule
T = 100

betas = torch.linspace(0.0001, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# 4. Define the neural network
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

        combined = torch.cat(
            [x, time_features],
            dim=1
        )

        return self.network(combined)


model = NoisePredictor().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()


# 5. Train the model
epochs = 5

for epoch in range(epochs):
    total_loss = 0.0

    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)

        # Select a random timestep for each image
        t = torch.randint(
            0, T, (batch_size,), device=device
        )

        # Generate random Gaussian noise
        noise = torch.randn_like(images)

        # Get the appropriate noise schedule values
        a_bar = alpha_bars[t].view(-1, 1, 1, 1)

        # Create noisy images
        noisy_images = (
            torch.sqrt(a_bar) * images
            + torch.sqrt(1 - a_bar) * noise
        )

        # Predict the added noise
        predicted_noise = model(noisy_images, t)

        # Calculate loss
        loss = criterion(
            predicted_noise,
            noise.view(batch_size, 784)
        )

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# 6. Save the trained model
torch.save(model.state_dict(), "diffusion_model.pth")

print("Training complete!")

