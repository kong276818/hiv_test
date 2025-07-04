import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# 폴더 경로 설정
data_dir = "C:/Users/jykong/Desktop/HIV_test/data"

# .mat 파일 리스트 가져오기
mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

print(f"🔍 총 {len(mat_files)}개 파일 발견")

# 하나씩 반복
for fname in mat_files:
    try:
        path = os.path.join(data_dir, fname)
        data = loadmat(path)
        
        # 유효한 key 추출 (__로 시작하는 건 제외)
        keys = [k for k in data.keys() if not k.startswith("__")]
        if not keys:
            print(f"⚠️ {fname}: 유효한 변수 없음")
            continue

        var_name = keys[0]
        img = data[var_name]

        print(f"📁 {fname} → 변수: {var_name}, shape: {img.shape}")

        # 시각화 (2D or 3D)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
            plt.title(f"{fname} - {var_name}")
            plt.colorbar()
            plt.show()
        elif img.ndim == 3:
            # 3차원일 경우: 첫 번째 Band
            plt.imshow(img[:, :, 0], cmap="gray")
            plt.title(f"{fname} - {var_name} (Band 0)")
            plt.colorbar()
            plt.show()
        else:
            print(f"❗ {fname}: 지원하지 않는 차원 {img.ndim}")
    except Exception as e:
        print(f"❌ {fname} 에러: {e}")