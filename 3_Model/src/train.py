import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Helper function to find and crop the end of a song
def crop_end_of_song(waveform, sample_rate, duration=3.0):
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, f_min=2000.0, f_max=9000.0
    )
    mel_spec = mel_transform(waveform)
    
    energy_envelope = torch.sum(mel_spec, dim=1).squeeze(0)
    threshold = energy_envelope.mean() + (0.5 * energy_envelope.std())
    active_frames = torch.where(energy_envelope > threshold)[0]
    
    samples_to_keep = int(duration * sample_rate)
    if len(active_frames) == 0:
        if waveform.shape[1] > samples_to_keep:
            return waveform[:, -samples_to_keep:]
        return waveform
        
    last_active_frame = active_frames[-1].item()
    last_active_sample = last_active_frame * 512
    
    padding = int(0.2 * sample_rate) 
    end_sample = min(waveform.shape[1], last_active_sample + padding)
    start_sample = max(0, end_sample - samples_to_keep)
    
    return waveform[:, start_sample:end_sample]

# PyTorch Dataset
class YellowhammerDataset(Dataset):
    def __init__(self, metadata_file, data_dir, mode="5-class", is_train=True, random_state=42):
        self.data_dir = data_dir
        self.is_train = is_train
        
        # Load complete metadata
        df = pd.read_csv(metadata_file)
        
        # Filter and set class balance boundaries
        if mode == "3-class":
            allowed_regions = ["east_germany", "netherlands_belgium", "southern_sweden"]
            df = df[df["target_region"].isin(allowed_regions)].reset_index(drop=True)
            samples_per_class = 31
            balanced_df = (
                df.groupby("target_region")
                .sample(n=samples_per_class, random_state=random_state)
                .reset_index(drop=True)
            )
            self.classes = sorted(balanced_df["target_region"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        elif mode == "4-class":
            allowed_regions = ["east_germany", "netherlands_belgium", "southern_sweden", "south_poland"]
            df = df[df["target_region"].isin(allowed_regions)].reset_index(drop=True)
            samples_per_class = 22
            balanced_df = (
                df.groupby("target_region")
                .sample(n=samples_per_class, random_state=random_state)
                .reset_index(drop=True)
            )
            self.classes = sorted(balanced_df["target_region"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        elif mode == "5-class":
            samples_per_class = 20
            balanced_df = (
                df.groupby("target_region")
                .sample(n=samples_per_class, random_state=random_state)
                .reset_index(drop=True)
            )
            self.classes = sorted(balanced_df["target_region"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        elif mode == "binary":
            # Island (UK) vs Continental (Germany, Netherlands, Poland, Sweden)
            # Balance at 20 samples per class: 20 UK, and 5 from each of the 4 continental regions
            uk_df = df[df["target_region"] == "southern_uk"].sample(n=20, random_state=random_state)
            
            continental_regions = ["east_germany", "netherlands_belgium", "southern_sweden", "south_poland"]
            continental_rows = []
            for r in continental_regions:
                r_df = df[df["target_region"] == r].sample(n=5, random_state=random_state)
                continental_rows.append(r_df)
            cont_df = pd.concat(continental_rows)
            
            uk_df["binary_label"] = 1
            cont_df["binary_label"] = 0
            
            balanced_df = pd.concat([uk_df, cont_df]).reset_index(drop=True)
            self.classes = ["continental", "island"]
            self.class_to_idx = {"continental": 0, "island": 1}
        else:
            raise ValueError("mode must be '3-class', '4-class', '5-class', or 'binary'")
            
        # Stratified Split (80% Train, 20% Val)
        train_rows = []
        val_rows = []
        # Group by target_region to keep geographical splits equal
        for _, group in balanced_df.groupby("target_region"):
            group = group.sample(frac=1, random_state=random_state).reset_index(drop=True)
            split_idx = int(len(group) * 0.8)
            train_rows.append(group.iloc[:split_idx])
            val_rows.append(group.iloc[split_idx:])
            
        self.active_df = pd.concat(train_rows).reset_index(drop=True) if is_train else pd.concat(val_rows).reset_index(drop=True)
        self.mode = mode
        
        # Setup pre-processing transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=128, f_min=2000.0, f_max=9000.0
        )
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)
        
    def __len__(self):
        return len(self.active_df)
        
    def __getitem__(self, idx):
        row = self.active_df.iloc[idx]
        file_path = os.path.join(self.data_dir, row["target_region"], f"{row['id']}.mp3")
        if self.mode == "binary":
            label = int(row["binary_label"])
        else:
            label = self.class_to_idx[row["target_region"]]
        
        # Audio loading
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to 22.05kHz Mono
        if sample_rate != 22050:
            resampler = T.Resample(sample_rate, 22050)
            waveform = resampler(waveform)
            sample_rate = 22050
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Preprocessing (End-Of-Song Crop)
        waveform = crop_end_of_song(waveform, sample_rate, duration=3.0)
        
        # Pad waveform if it's too short (less than 3 seconds)
        target_len = int(3.0 * sample_rate)
        if waveform.shape[1] < target_len:
            pad_len = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
        # Audio-level augmentation (Random Time shift)
        if self.is_train:
            shift = random.randint(-int(0.3 * sample_rate), int(0.3 * sample_rate))
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
            
        # Generate Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = T.AmplitudeToDB()(mel_spec)[0]
        
        # Median Subtraction Denoising
        median_noise = torch.median(log_mel_spec, dim=1, keepdim=True)[0]
        log_mel_spec = log_mel_spec - median_noise
        log_mel_spec = torch.clamp(log_mel_spec, min=0)
        
        # Spectrogram-level augmentation (SpecAugment)
        if self.is_train:
            log_mel_spec = log_mel_spec.unsqueeze(0)
            log_mel_spec = self.freq_mask(log_mel_spec)
            log_mel_spec = self.time_mask(log_mel_spec)
            log_mel_spec = log_mel_spec.squeeze(0)
            
        return log_mel_spec.unsqueeze(0), label # Output shape: (1, n_mels, time)

# Lightweight CNN Architecture
class LightweightCNN(nn.Module):
    def __init__(self, num_classes):
        super(LightweightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 16))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / total, correct / total

def print_baseline_metrics(val_dataset):
    print("\n--- BASELINE MODEL ESTIMATION ---")
    # Read labels directly from pandas dataframe to avoid loading/processing audio files
    if val_dataset.mode == "binary":
        labels = val_dataset.active_df["binary_label"].tolist()
    else:
        labels = val_dataset.active_df["target_region"].map(val_dataset.class_to_idx).tolist()
        
    total = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Majority class baseline
    majority_count = max(counts) if len(counts) > 0 else 0
    majority_acc = majority_count / total if total > 0 else 0.0
    
    # Random uniform guess baseline
    num_classes = len(val_dataset.classes)
    random_acc = 1.0 / num_classes
    
    print(f"Baseline Classifier: Uniform Random Guessing / Dummy Classifier")
    print(f"Number of Classes: {num_classes} ({', '.join(val_dataset.classes)})")
    print(f"Uniform Random Guessing Accuracy: {random_acc:.2%}")
    print(f"Majority Class Accuracy: {majority_acc:.2%}")
    print("---------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Avian Geographic Dialect Classifier Baseline")
    parser.add_argument("--mode", type=str, default="5-class", choices=["3-class", "4-class", "5-class", "binary"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    metadata_file = os.path.join("..", "data", "metadata.csv")
    data_dir = os.path.join("..", "data")
    
    print(f"Initializing Dataset (Mode: {args.mode})...")
    train_dataset = YellowhammerDataset(metadata_file, data_dir, mode=args.mode, is_train=True)
    val_dataset = YellowhammerDataset(metadata_file, data_dir, mode=args.mode, is_train=False)
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    
    print_baseline_metrics(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = len(train_dataset.classes)
    model = LightweightCNN(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_loss = float("inf")
    print("\nStarting Training Baseline...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{args.mode}.pt")
            
        print(f"Epoch [{epoch:02d}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2%}) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2%})")
              
    # Plotting Loss and Accuracy curves
    epochs_range = range(1, args.epochs + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, val_accs, label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plot_path = f"learning_curves_{args.mode}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaved learning curves to {plot_path}")
    
    # Generate Confusion Matrix on Validation Set
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    model.load_state_dict(torch.load(f"best_model_{args.mode}.pt"))
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.item(), p.item()] += 1
                
    print("\n=== CONFUSION MATRIX (VAL SET) ===")
    print("Class mapping: ", {i: c for i, c in enumerate(train_dataset.classes)})
    print(confusion)

if __name__ == "__main__":
    main()
