import cv2
import numpy as np
from scipy import stats


def calc_laplacian_var(gray):
    """Laplacian 方差（越大越清晰）"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def calc_sobel_energy(gray):
    """Sobel 能量，检测边缘清晰度"""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return (sobel_x ** 2 + sobel_y ** 2).mean()


def calc_contrast(gray):
    """图像对比度（标准差）"""
    return gray.std()


def calc_noise(gray):
    """噪声估计：使用高斯滤波差分"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_map = gray - blur
    return noise_map.std()


def histogram_features(gray):
    """灰度直方图统计：偏度、峰度、熵"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()

    entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-8))
    skew = stats.skew(hist_norm)
    kurt = stats.kurtosis(hist_norm)

    return entropy, skew, kurt


def extract_features(image):
    """
    主特征提取函数，返回 1D 数组特征
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap = calc_laplacian_var(gray)
    sobel = calc_sobel_energy(gray)
    contrast = calc_contrast(gray)
    noise = calc_noise(gray)
    entropy, skew, kurt = histogram_features(gray)

    return np.array([
        lap,
        sobel,
        contrast,
        noise,
        entropy,
        skew,
        kurt
    ])
