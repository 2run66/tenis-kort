import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def adaptive_threshold_clahe(gray_clahe):
    adaptive = cv2.adaptiveThreshold(
        gray_clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return adaptive

def morphological_cleanup(binary_img):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def show_images(original, clahe, adaptive, cleaned):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(clahe, cmap='gray')
    plt.title("CLAHE")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(adaptive, cmap='gray')
    plt.title("Adaptive Threshold")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cleaned, cmap='gray')
    plt.title("Final Cleaned Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# === Ana Akış ===

image_path = "C:\\kortcizim.png"
image = cv2.imread(image_path)

if image is None:
    print("Görüntü yüklenemedi. Yol doğru mu?")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = apply_clahe(gray)
adaptive = adaptive_threshold_clahe(clahe)
final_mask = morphological_cleanup(adaptive)

# Kaydet
cv2.imwrite("clahe.png", clahe)
cv2.imwrite("adaptive_threshold_clahe.png", adaptive)
cv2.imwrite("final_white_mask.png", final_mask)
print("Görüntüler kaydedildi:\n- clahe.png\n- adaptive_threshold_clahe.png\n- final_white_mask.png")

# Göster
show_images(image, clahe, adaptive, final_mask)
