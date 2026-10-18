import os
import json
import torch
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification, AdamW

# 1. Thiết lập cấu hình cơ bản
LOCAL_DATA_DIR = Path("./dataset") # Thư mục chứa dữ liệu đã giải nén
OLD_CSV = LOCAL_DATA_DIR / "train.csv"
NEW_CSV = LOCAL_DATA_DIR / "fine_tune.csv"
CHECKPOINT = LOCAL_DATA_DIR / "model"

SR = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {DEVICE}")

# 2. Xử lý nhãn (29 lớp tổng cộng)
config = json.loads((CHECKPOINT / "config.json").read_text())
old_id2label = {int(k): v for k, v in config["id2label"].items()}

old_df = pd.read_csv(OLD_CSV)
new_df = pd.read_csv(NEW_CSV)
new_id2label = dict(new_df[["target", "category"]].drop_duplicates().values)
new_id2label = {int(k): v for k, v in sorted(new_id2label.items())}

id2label = {**old_id2label, **new_id2label}
NUM_TOTAL = len(id2label)
print(f"Tổng số lớp sau khi mở rộng: {NUM_TOTAL}")

# 3. Định nghĩa Dataset đọc âm thanh
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

# 4. Tải mô hình AST và mở rộng đầu ra lên 29 lớp
model = ASTForAudioClassification.from_pretrained(
    str(CHECKPOINT),
    num_labels=NUM_TOTAL,
    ignore_mismatched_sizes=True
).to(DEVICE)

# 5. Huấn luyện mô hình (Fine-tune)
train_combined_df = pd.concat([
    old_df[old_df['split'] == 'train'], 
    new_df[new_df['split'] == 'train']
], ignore_index=True)

train_dataset = AudioDataset(train_combined_df, LOCAL_DATA_DIR, feature_extractor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # Giảm batch_size xuống 4 nếu RAM/GPU yếu

optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
print("Bắt đầu quá trình huấn luyện trên VS Code...")
for epoch in range(1): # Chạy 1 epoch để đảm bảo hoàn thành nhanh dưới 10 phút
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

print("Huấn luyện hoàn tất! Mô hình đã sẵn sàng.")