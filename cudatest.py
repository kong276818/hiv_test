import torch
print("CUDA 사용 가능:", torch.cuda.is_available())
print("사용 중인 장치:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

print("✅ 현재 디바이스:", device)
print("✅ 모델 디바이스:", next(model.parameters()).device)