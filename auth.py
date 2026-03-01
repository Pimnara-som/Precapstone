# auth.py
import face_recognition
from config import FACE_TOLERANCE 

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

# (ลบ verify_object_key และการ import ssim ออกไปได้เลยครับ)