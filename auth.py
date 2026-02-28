# auth.py
import face_recognition
from skimage.metrics import structural_similarity as ssim 
from config import FACE_TOLERANCE, SSIM_THRESHOLD 

def verify_face(known_image_path, unknown_image_path):
    """ตรวจสอบว่าใบหน้าตรงกันหรือไม่"""
    try:
        # โหลดรูปภาพ
        known_image = face_recognition.load_image_file(known_image_path)
        unknown_image = face_recognition.load_image_file(unknown_image_path)
        
        # แปลงรูปเป็น Encodings
        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        # เปรียบเทียบ
        results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=FACE_TOLERANCE)
        return results[0]
    except IndexError:
        # กรณีหาใบหน้าในรูปไม่เจอ
        return False
    except Exception as e:
        print(f"Face verification error: {e}")
        return False

def verify_object_key(extracted_wm, original_wm):
    """ตรวจสอบความคล้ายคลึงของ Object Key ด้วย SSIM"""
    score = ssim(extracted_wm, original_wm, data_range=original_wm.max() - original_wm.min())
    print(f"SSIM Score: {score:.4f}") # พิมพ์ค่าให้ดูใน Terminal เพื่อใช้ปรับจูน
    return score >= SSIM_THRESHOLD