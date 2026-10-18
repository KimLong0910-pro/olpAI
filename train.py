import json
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification, AdamW

warnings.filterwarnings("ignore")

# 0. Setup
LOCAL_DATA_DIR = Path("./dataset")
if not LOCAL_DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Please run download_data.py first."
    )

OLD_CSV = LOCAL_DATA_DIR / "train.csv"
NEW_CSV = LOCAL_DATA_DIR / "fine_tune.csv"
CHECKPOINT = LOCAL_DATA_DIR / "model"
SR = 16_000
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

config = json.loads((CHECKPOINT / "config.json").read_text())
old_id2label = {int(k): v for k, v in config["id2label"].items()}
NUM_OLD = len(old_id2label)
print(f"device = {DEVICE} | checkpoint knows {NUM_OLD} classes")

old_df = pd.read_csv(OLD_CSV)
new_df = pd.read_csv(NEW_CSV)
new_df["target"] = new_df["target"]
new_id2label = dict(new_df[["target", "category"]].drop_duplicates().values)
new_id2label = {int(k): v for k, v in sorted(new_id2label.items())}

id2label = {**old_id2label, **new_id2label}
label2id = {v: k for k, v in id2label.items()}
NUM_TOTAL = len(id2label)
print(f"total classes after expansion: {NUM_TOTAL}")

# Feature Extractor and Dataset Class
feature_extractor = ASTFeatureExtractor.from_pretrained(str(CHECKPOINT))

class AudioDataset(Dataset):
    def __init__(self, df, data_dir, feature_extractor):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.fe = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.data_dir / row["path"]
        
        wav, _ = librosa.load(audio_path, sr=SR, mono=True)
        inputs = self.fe(wav, sampling_rate=SR, return_tensors="pt")
        input_values = inputs["input_values"].squeeze(0)
        
        label = row["target"]
        return {"input_values": input_values, "labels": torch.tensor(label, dtype=torch.long)}

# Load Pre-trained AST Model
model = ASTForAudioClassification.from_pretrained(
    str(CHECKPOINT),
    num_labels=NUM_TOTAL,
    ignore_mismatched_sizes=True
).to(DEVICE)

# Prepare Combined Dataset and DataLoader
train_combined_df = pd.concat([
    old_df[old_df['split'] == 'train'], 
    new_df[new_df['split'] == 'train']
], ignore_index=True)

train_dataset = AudioDataset(train_combined_df, LOCAL_DATA_DIR, feature_extractor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer & Loss Function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training Loop
model.train()
print("Starting training loop...")
for epoch in range(1):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["input_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits
        
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("Training finished successfully!")