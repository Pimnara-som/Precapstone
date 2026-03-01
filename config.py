# config.py
ALPHA = 0.05
WAVELET = 'haar'
IMG_SIZE = 512
WM_SIZE = 256

# ตัวแปรที่ auth.py เรียกใช้
FACE_TOLERANCE = 0.6
SSIM_THRESHOLD = 0.50 # ปรับลดลงเพื่อให้รองรับ Noise จาก DWT และการครอบตัดภาพของ AI