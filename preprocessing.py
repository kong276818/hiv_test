import os
import numpy as np
from sklearn.model_selection import train_test_split

#  데이터 경로
data_dir = "C:/Users/jykong/Desktop/HIV_test/pros_data"

#  폴더 내 모든 .npy 파일 가져오기
all_files = os.listdir(data_dir)

# X와 y 파일만 따로 정리 (예: X_img1.npy, y_img1.npy)
x_files = sorted([f for f in all_files if f.startswith("X_") and f.endswith(".npy")])
y_files = sorted([f for f in all_files if f.startswith("y_") and f.endswith(".npy")])

#  X-y 쌍으로 순회
for x_file, y_file in zip(x_files, y_files):
    # 파일 경로
    x_path = os.path.join(data_dir, x_file)
    y_path = os.path.join(data_dir, y_file)

    # 데이터 불러오기
    X = np.load(x_path)
    y = np.load(y_path)

    #  1단계: train+val vs test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    #  2단계: train vs val (20% of temp → 0.25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    #  저장용 이름 추출 (예: X_img1.npy → img1)
    name_suffix = x_file.replace("X_", "").replace(".npy", "")

    #  분할 결과 저장
    np.save(os.path.join(data_dir, f"X_train_{name_suffix}.npy"), X_train)
    np.save(os.path.join(data_dir, f"X_val_{name_suffix}.npy"), X_val)
    np.save(os.path.join(data_dir, f"X_test_{name_suffix}.npy"), X_test)

    np.save(os.path.join(data_dir, f"y_train_{name_suffix}.npy"), y_train)
    np.save(os.path.join(data_dir, f"y_val_{name_suffix}.npy"), y_val)
    np.save(os.path.join(data_dir, f"y_test_{name_suffix}.npy"), y_test)

    # 상태 출력
    print(f"[{name_suffix}] → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
