# core_dwt.py
import cv2
import pywt
import numpy as np
from config import *

def preprocess_image(img_path, size):
    """อ่านและปรับขนาดภาพเป็น Grayscale"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"ไม่พบรูปภาพที่: {img_path}")
    return cv2.resize(img, (size, size))

def embed_watermark(face_img, watermark_img):
    """ฝังลายน้ำลงในภาพใบหน้าด้วย DWT"""
    # ปรับสเกลลายน้ำ
    watermark_normalized = watermark_img / 255.0 # ปรับสเกลให้อยู่ในช่วง [0, 1]

    # 1 Level 2D-DWT
    coeffs2 = pywt.dwt2(face_img, WAVELET)
    LL, (HL, LH, HH) = coeffs2

    # สมการฝังลายน้ำ: HL' = HL + alpha * W
    HL_watermarked = HL + (ALPHA * watermark_normalized * np.max(HL))#คือค่าใหม่ที่ซ่อนลายน้ำแล้ว

    # Inverse DWT รวมภาพใบหน้าที่มีลายน้ำ
    coeffs2_watermarked = LL, (HL_watermarked, LH, HH)
    watermarked_face = pywt.idwt2(coeffs2_watermarked, WAVELET)
    
    return np.clip(watermarked_face, 0, 255).astype(np.uint8), HL

def extract_watermark(watermarked_img, original_hl):
    """สกัดลายน้ำออกจากภาพใบหน้า"""
    coeffs2_wm = pywt.dwt2(watermarked_img, WAVELET)
    LL_wm, (HL_wm, LH_wm, HH_wm) = coeffs2_wm

    # สมการสกัดลายน้ำ: W_ext = (HL' - HL) / (alpha * max(HL))
    extracted_watermark = (HL_wm - original_hl) / (ALPHA * np.max(original_hl))
    extracted_watermark = extracted_watermark * 255.0
    
    return np.clip(extracted_watermark, 0, 255).astype(np.uint8)