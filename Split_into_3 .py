from sklearn.model_selection import train_test_split
import numpy as np
import os

X = np.load("X_img1.npy")  # (N, 1, B, H, W)
y = np.load("y_img1.npy")  # (N,)

# 1️먼저 train+val vs test (20%) 분할
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2️train vs val (from temp set) → 여기서 20%는 0.25로 나눠야 전체에서 20% 비율
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")