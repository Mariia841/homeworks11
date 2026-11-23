import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ДАТАСЕТ 
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# ==========================================
# АРХІТЕКТУРИ CNN: 
# ==========================================
class BaseCNN(nn.Module):
    """CNN без нормалізації та Dropout."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ModifiedCNN(nn.Module):
    """CNN з BatchNorm2d після кожного Conv2d та Dropout(p=0.3) після ReLU."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# НАВЧАННЯ З КОНТРОЛЕМ ГРАДІЄНТІВ ТА ФІКСАЦІЄЮ ВТРАТ
# ==========================================
def train_model(model, loader, epochs=10, lr=1e-3, max_grad_norm=1.0, use_grad_clip=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    grad_norms_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_grad_norms = []

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if use_grad_clip:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                batch_grad_norms.append(total_norm.item())

            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(loader.dataset)
        epoch_losses.append(avg_loss)

        avg_norm = np.mean(batch_grad_norms) if batch_grad_norms else 0.0
        grad_norms_history.append(avg_norm)

        print(
            f"[{model.__class__.__name__}] Епоха {epoch:02d}/{epochs} — "
            f"Loss: {avg_loss:.4f}"
            + (f" | Середня норма градієнта до clip: {avg_norm:.4f}" if use_grad_clip else "")
        )

    return epoch_losses, grad_norms_history


epochs = 10

print("=== Навчання моделі БЕЗ нормалізації ===")
base_model = BaseCNN(num_classes=10)
base_losses, _ = train_model(base_model, train_loader, epochs=epochs, max_grad_norm=1.0, use_grad_clip=True)

print("\n=== Навчання моделі ІЗ BatchNorm2d та Dropout(0.3) ===")
mod_model = ModifiedCNN(num_classes=10)
mod_losses, mod_grad_norms = train_model(mod_model, train_loader, epochs=epochs, max_grad_norm=1.0, use_grad_clip=True)


# ==========================================
# ПОРІВНЯЛЬНИЙ ГРАФІК
# ==========================================
plt.figure(figsize=(9, 5))
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, base_losses, marker="o", label="Модель без нормалізації")
plt.plot(epochs_range, mod_losses, marker="s", label="Модель із BatchNorm та Dropout")

plt.title("Порівняння втрат за епохами (clip_grad_norm max=1.0)")
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.xticks(epochs_range)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()