import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd

# --- 1. 데이터 불러오기 ---
data = loadmat("img1.mat")
hsi_cube = data['ref']       # (H, W, B)
label_map = data['lbl']      # (H, W)

# --- 2. 전처리 ---
H, W, B = hsi_cube.shape
X = hsi_cube.reshape(-1, B)
y = label_map.reshape(-1)

# --- 유효 데이터 확인 및 완화된 마스크 적용 ---
print("라벨 고유값:", np.unique(y))
print("NaN 개수:", np.isnan(y).sum())

# 마스킹 조건 완화
valid_mask = ~np.isnan(y)
X_valid = X[valid_mask]
y_valid = y[valid_mask]

# --- 결과 확인 ---
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)
print("고유 라벨:", np.unique(y_valid))

# --- 3. 저장 ---
output_prefix = "preprocessed_img1"
np.save(f"{output_prefix}_X.npy", X_valid)
np.save(f"{output_prefix}_y.npy", y_valid)
savemat(f"{output_prefix}.mat", {"X": X_valid, "y": y_valid})
df = pd.DataFrame(X_valid)
df["label"] = y_valid
df.to_csv(f"{output_prefix}.csv", index=False)

print("✅ 저장 완료")