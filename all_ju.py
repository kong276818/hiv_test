import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. 파일 불러오기
data = loadmat("img1.mat")
hsi_cube = data['ref']  # (1040, 1392, 31)
label_map = data['lbl']  # (1040, 1392)

# 2. NaN/이상치 확인 (선택)
# np.isnan(hsi_cube).sum() 또는 np.isinf(hsi_cube).sum() 등을 사용 가능

# 3. reshape: 이미지 전체를 픽셀 단위로 펼치기
H, W, B = hsi_cube.shape
X = hsi_cube.reshape(-1, B)      # (1040 * 1392, 31)
y = label_map.reshape(-1)        # (1040 * 1392,)

print("X shape:", X.shape)
print("y shape:", y.shape)

# 4. 마스킹: 유효한 라벨만 남기기 (예: -1, 0 등 제거)
valid_mask = ~np.isnan(y)  # 또는 (y > 0), (y != 0) 등 상황에 따라
X = X[valid_mask]
y = y[valid_mask]

# 5. 데이터 스케일링 (선택적)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# 6. 훈련/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# 7. 모델 훈련 (예: Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. 예측 및 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("✅ Mean Squared Error (MSE):", mse)

# 9. 예측 vs 정답 시각화
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("True Brix")
plt.ylabel("Predicted Brix")
plt.title("Prediction vs Truth")
plt.grid(True)
plt.show()
