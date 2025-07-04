import numpy as np
from scipy.io import loadmat

def save_reference_spectrum(mat_path, output_txt="white.txt"):
    """
    .mat 파일의 'ref' 데이터에서 평균 스펙트럼 추출 후 txt 저장
    """
    data = loadmat(mat_path)
    hsi = data["ref"]  # shape: (H, W, B)
    
    # 평균 스펙트럼 계산 (axis=(0,1): 공간평균)
    mean_spectrum = np.mean(hsi, axis=(0, 1))  # shape: (B,)
    
    # 저장
    np.savetxt(output_txt, mean_spectrum, fmt="%.8f")
    print(f"✅ 평균 스펙트럼 저장 완료: {output_txt}")

# 사용 예시
save_reference_spectrum("white.mat", output_txt="white.txt")
save_reference_spectrum("dark.mat", output_txt="dark.txt")