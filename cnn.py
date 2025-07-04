import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# --- 1. 데이터 불러오기 ---
X = np.load("preprocessed_img1_X.npy")  # shape: (1447680, 31)
y = np.load("preprocessed_img1_y.npy")  # shape: (1447680,)

# --- 2. 훈련/검증 분할 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 3. 모델 정의 및 학습 ---
model = XGBClassifier(
    device='cuda',        # GPU 사용 (CUDA 환경일 경우)
    n_estimators=100,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# --- 4. 예측 및 평가 ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- 5. 시각화 ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.2)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("Prediction Scatter")
plt.grid(True)
plt.show()
