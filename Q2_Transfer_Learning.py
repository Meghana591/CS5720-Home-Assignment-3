import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Image transformations
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. Load CIFAR-10 dataset
# --------------------------------------------------

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Use a smaller subset to make training faster
train_dataset = torch.utils.data.Subset(
    train_dataset,
    range(5000)
)

test_dataset = torch.utils.data.Subset(
    test_dataset,
    range(1000)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

num_classes = 10


# --------------------------------------------------
# 4. Function to create pretrained ResNet18
# --------------------------------------------------

def create_model():
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Replace final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# --------------------------------------------------
# 5. Count trainable parameters
# --------------------------------------------------

def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# --------------------------------------------------
# 6. Train function
# --------------------------------------------------

def train_model(model, epochs=3):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    loss_history = []

    start_time = time.time()

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        loss_history.append(epoch_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {epoch_loss:.4f}"
        )

    training_time = time.time() - start_time

    return loss_history, training_time


# --------------------------------------------------
# 7. Test accuracy
# --------------------------------------------------

def test_accuracy(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# ==================================================
# EXPERIMENT A — FEATURE EXTRACTION
# ==================================================

print("\n==============================")
print("EXPERIMENT A: FEATURE EXTRACTION")
print("==============================")

model_a = create_model()

# Freeze convolutional/base layers
for param in model_a.parameters():
    param.requires_grad = False

# Make only the final classifier trainable
for param in model_a.fc.parameters():
    param.requires_grad = True

model_a = model_a.to(device)

trainable_a = count_trainable_parameters(model_a)

print("Trainable parameters:", trainable_a)

loss_a, time_a = train_model(
    model_a,
    epochs=3
)

accuracy_a = test_accuracy(model_a)

print("Training time:", time_a, "seconds")
print("Test accuracy:", accuracy_a, "%")


# ==================================================
# EXPERIMENT B — FINE-TUNING
# ==================================================

print("\n==============================")
print("EXPERIMENT B: FINE-TUNING")
print("==============================")

model_b = create_model()

# Freeze all layers first
for param in model_b.parameters():
    param.requires_grad = False

# Unfreeze the last convolutional block
for param in model_b.layer4.parameters():
    param.requires_grad = True

# Train the final classifier
for param in model_b.fc.parameters():
    param.requires_grad = True

model_b = model_b.to(device)

trainable_b = count_trainable_parameters(model_b)

print("Trainable parameters:", trainable_b)

loss_b, time_b = train_model(
    model_b,
    epochs=3
)

accuracy_b = test_accuracy(model_b)

print("Training time:", time_b, "seconds")
print("Test accuracy:", accuracy_b, "%")


# ==================================================
# 8. Comparison table
# ==================================================

print("\n==============================")
print("COMPARISON")
print("==============================")

print(
    f"{'Method':<25}"
    f"{'Trainable Parameters':<25}"
    f"{'Training Time':<20}"
    f"{'Accuracy'}"
)

print(
    f"{'Frozen Feature Extractor':<25}"
    f"{trainable_a:<25}"
    f"{time_a:.2f} sec{'':<10}"
    f"{accuracy_a:.2f}%"
)

print(
    f"{'Fine-Tuned Network':<25}"
    f"{trainable_b:<25}"
    f"{time_b:.2f} sec{'':<10}"
    f"{accuracy_b:.2f}%"
)


# ==================================================
# 9. Plot training loss
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 4),
    loss_a,
    marker="o",
    label="Frozen Feature Extractor"
)

plt.plot(
    range(1, 4),
    loss_b,
    marker="o",
    label="Fine-Tuned Network"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")

plt.legend()
plt.grid(True)

plt.savefig("training_loss_comparison.png")

plt.show()