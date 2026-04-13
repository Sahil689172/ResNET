import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

# =====================
# LOAD DATA
# =====================
train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

print("Train size:", len(train_data))
print("Test size:", len(test_data))

# =====================
# LOAD RESNET MODEL
# =====================
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Change output layer (3 classes)
model.fc = nn.Linear(model.fc.in_features, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =====================
# TRAINING SETUP
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =====================
# TRAIN LOOP
# =====================
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =====================
# EVALUATION
# =====================
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# =====================
# METRICS
# =====================
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# =====================
# SPECIFICITY
# =====================
specificity_list = []

for i in range(len(cm)):
    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
    fp = np.sum(cm[:, i]) - cm[i, i]

    specificity = tn / (tn + fp + 1e-6)
    specificity_list.append(specificity)

print("Specificity:", np.mean(specificity_list))

# =====================
# AUC SCORE
# =====================
try:
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print("AUC Score:", auc)
except:
    print("AUC could not be computed")