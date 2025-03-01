import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata
file_path = "C:/Data/DJ/azcopydata/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx"
df = pd.read_excel(file_path)

# Define image directory (same folder as the Excel sheet)
image_dir = os.path.dirname(file_path)

# Convert categorical columns to string before encoding
df = df.astype(str)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['midas_gender', 'midas_fitzpatrick', 'midas_ethnicity', 'midas_race']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Convert numerical columns to float and handle errors
numerical_cols = ['midas_age', 'length_(mm)', 'width_(mm)']
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Normalize numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(0))

# Define target variable
df['target'] = df['midas_melanoma'].map({'yes': 1, 'no': 0})

# Remove rows with missing labels
df = df.dropna(subset=['target']).reset_index(drop=True)

# Debugging: Check unique label values and types
print("Unique label values:", df["target"].unique())
print("Label data type:", df["target"].dtype)

# Remove rows where the image file is missing
def file_exists(filename):
    possible_paths = [
        os.path.join(image_dir, filename),
        os.path.join(image_dir, filename.replace('.jpg', '.jpeg')),
        os.path.join(image_dir, filename.replace('.jpeg', '.jpg'))
    ]
    return any(os.path.exists(path) for path in possible_paths)

df = df[df['midas_file_name'].apply(file_exists)].reset_index(drop=True)

# Define dataset class
class MRAMIDASDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image with flexible file extension handling
        img_filename = row['midas_file_name']
        img_path = os.path.join(self.image_dir, img_filename)
        
        if not os.path.exists(img_path):
            img_path_jpeg = img_path.replace('.jpg', '.jpeg')
            img_path_jpg = img_path.replace('.jpeg', '.jpg')
            if os.path.exists(img_path_jpeg):
                img_path = img_path_jpeg
            elif os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            else:
                raise FileNotFoundError(f"Image not found: {img_filename}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        metadata_values = row[categorical_cols + numerical_cols].values.astype(float)
        metadata = torch.tensor(metadata_values, dtype=torch.float32)
        
        label = int(row['target'])
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, metadata, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = MRAMIDASDataset(df, image_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define CNN for image processing
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 128)
    
    def forward(self, x):
        return self.model(x)

# Define MLP for metadata processing
class MetadataModel(nn.Module):
    def __init__(self, input_size):
        super(MetadataModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc(x)

# Define Multimodal Model
class MultimodalModel(nn.Module):
    def __init__(self, image_model, metadata_model):
        super(MultimodalModel, self).__init__()
        self.image_model = image_model
        self.metadata_model = metadata_model
        self.classifier = nn.Linear(128 + 32, 2)
    
    def forward(self, image, metadata):
        img_features = self.image_model(image)
        meta_features = self.metadata_model(metadata)
        combined = torch.cat((img_features, meta_features), dim=1)
        return self.classifier(combined)

# Instantiate models
image_model = ImageModel().to(device)
metadata_model = MetadataModel(input_size=len(categorical_cols) + len(numerical_cols)).to(device)
model = MultimodalModel(image_model, metadata_model).to(device)

# Define learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

print(f"Using device: {device}")

print(summary(model, [(3, 224, 224), (len(categorical_cols) + len(numerical_cols),)]))

exit()
# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    total_loss, correct, total = 0, 0, 0
    
    model.train()
    for images, metadata, labels in train_loader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(torch.int64).to(device)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_accuracy = correct / total
    
    # Validation step
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(torch.int64).to(device)
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_accuracy = val_correct / val_total
    scheduler.step(val_loss)  # Adjust learning rate based on validation loss
    
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")
