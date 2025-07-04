import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ✅ 고정 설정
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# ✅ Hyperspectral Dataset
class HSINPYDataset(Dataset):
    def __init__(self, folder_path, prefix):
        self.samples = []
        x_files = sorted([f for f in os.listdir(folder_path) if f.startswith(f"X_{prefix}_")])
        for x_file in x_files:
            suffix = x_file.replace(f"X_{prefix}_", "").replace(".npy", "")
            y_file = f"y_{prefix}_{suffix}.npy"
            x_path, y_path = os.path.join(folder_path, x_file), os.path.join(folder_path, y_file)
            if os.path.exists(y_path):
                self.samples.append((x_path, y_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx][0], mmap_mode='r')
        y = np.load(self.samples[idx][1], mmap_mode='r')

        if x.ndim == 4: x = x.squeeze(0)
        if x.ndim == 2: x = np.expand_dims(x, axis=0)

        if y.ndim > 0:
            y = y[0]  # 🎯 벡터형 라벨에서 첫 원소만 사용

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

# ✅ CNN 모델 정의
class HSICNN(nn.Module):
    def __init__(self, in_channels, h, w, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ✅ 메인 실행

def main():
    base_dir = "C:/Users/jykong/Desktop/HIv_test/pros_data"
    train_set = HSINPYDataset(os.path.join(base_dir, "train"), "train")
    val_set = HSINPYDataset(os.path.join(base_dir, "val"), "val")
    test_set = HSINPYDataset(os.path.join(base_dir, "test"), "test")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'None'}")

    # ✅ 입력 shape 자동 추출
    sample_x, _ = train_set[0]
    print(f"📐 Original shape: {sample_x.shape}")

    if sample_x.ndim == 5:
        sample_x = sample_x[0]
    if sample_x.ndim == 4:
        sample_x = sample_x.squeeze(0)
    if sample_x.ndim == 2:
        sample_x = np.expand_dims(sample_x, axis=0)

    C, H, W = sample_x.shape
    print(f"✅ Model input shape → C: {C}, H: {H}, W: {W}")

    # ✅ 클래스 수 추정
    all_labels = []
    for _, y_path in train_set.samples + val_set.samples + test_set.samples:
        y = np.load(y_path, mmap_mode='r')
        if y.ndim > 0:
            y = y[0]
        all_labels.append(int(y))
    num_classes = int(max(all_labels)) + 1
    print(f"🧠 Detected classes: {num_classes}")

    model = HSICNN(C, H, W, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"🔧 Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"🧪 Val Acc: {correct/total:.4f}")

    # ✅ 테스트 정확도
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x).argmax(1).cpu().numpy()
            preds.extend(out)
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    print(f"✅ Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()