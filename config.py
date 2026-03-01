# config.py
ALPHA = 0.10
WAVELET = 'haar'
IMG_SIZE = 512
WM_SIZE = 256

# ตัวแปรระบบยืนยันตัวตน
FACE_TOLERANCE = 0.6
CNN_SIM_THRESHOLD = 0.65 # เปลี่ยนจาก SSIM เป็นเกณฑ์ความเหมือนของ CNN (แนะนำ 0.65 - 0.70)