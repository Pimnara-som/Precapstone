import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

# --- ส่วนแก้ปัญหา pkg_resources ---
try:
    import pkg_resources
except (ImportError, ModuleNotFoundError):
    import setuptools
    sys.modules['pkg_resources'] = setuptools

# --- นำเข้า AI YOLOv8 ---
try:
    from ultralytics import YOLO
except ImportError:
    messagebox.showerror("ขาด Library", "กรุณาติดตั้ง AI ก่อนโดยพิมพ์:\npip install ultralytics")
    sys.exit()

# Import Logic ของโปรเจกต์
from auth import verify_face, verify_object_key
from core_dwt import preprocess_image, embed_watermark, extract_watermark
from config import IMG_SIZE, WM_SIZE

# --- ตัวแปรเก็บที่อยู่ไฟล์ ---
reg_face_path = ""
reg_obj_path = ""
login_img_path = ""

# โหลดโมเดล AI เตรียมไว้เลย (ครั้งแรกที่รันจะมีการดาวน์โหลดไฟล์โมเดลขนาดเล็ก ~6MB อัตโนมัติ)
print("กำลังโหลดโมเดล AI YOLOv8...")
model_ai = YOLO("yolov8n.pt") 
print("โหลด AI สำเร็จ!")

def select_reg_face():
    global reg_face_path
    path = filedialog.askopenfilename(title="เลือกรูปใบหน้า", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        reg_face_path = path
        lbl_reg_face.config(text=f"ใบหน้า: {os.path.basename(path)}", fg="#2E7D32")

def select_reg_obj():
    global reg_obj_path
    path = filedialog.askopenfilename(title="เลือกรูปวัตถุ (Key)", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        reg_obj_path = path
        lbl_reg_obj.config(text=f"วัตถุ: {os.path.basename(path)}", fg="#2E7D32")

def select_login_img():
    global login_img_path
    path = filedialog.askopenfilename(title="เลือกรูปภาพ (ต้องมีใบหน้าและวัตถุ)", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        login_img_path = path
        lbl_login_img.config(text=f"รูปที่เลือก: {os.path.basename(path)}", fg="#2E7D32")

def register_action():
    global reg_face_path, reg_obj_path
    if not reg_face_path or not reg_obj_path:
        messagebox.showwarning("เตือน", "กรุณาเลือกรูปให้ครบทั้ง ใบหน้า และ วัตถุ ครับ")
        return
    
    try:
        face_img = preprocess_image(reg_face_path, IMG_SIZE)
        obj_img = preprocess_image(reg_obj_path, WM_SIZE)
        
        watermarked_face, hl_band = embed_watermark(face_img, obj_img)
        
        cv2.imwrite("db_watermarked_face.png", watermarked_face)
        np.save("db_hl.npy", hl_band)
        cv2.imwrite("db_original_face.jpg", cv2.imread(reg_face_path))
        
        messagebox.showinfo("สำเร็จ", "ลงทะเบียนและฝังลายน้ำ (DWT) เรียบร้อยแล้ว!")
        
        global reg_face_path_temp, reg_obj_path_temp
        reg_face_path = ""
        reg_obj_path = ""
        lbl_reg_face.config(text="ใบหน้า: ยังไม่ได้เลือกไฟล์", fg="gray")
        lbl_reg_obj.config(text="วัตถุ: ยังไม่ได้เลือกไฟล์", fg="gray")
        show_frame(frame_home)
        
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถลงทะเบียนได้:\n{str(e)}")

def login_action():
    global login_img_path
    if not login_img_path:
        messagebox.showwarning("เตือน", "กรุณาเลือกรูปภาพสำหรับเข้าสู่ระบบครับ")
        return
    
    if not os.path.exists("db_watermarked_face.png") or not os.path.exists("db_hl.npy"):
        messagebox.showerror("ล้มเหลว", "ไม่พบข้อมูลในระบบ กรุณาลงทะเบียนก่อน!")
        return

    try:
        # --- ขั้นที่ 1: ตรวจสอบใบหน้า ---
        is_face_match = verify_face("db_original_face.jpg", login_img_path)
        if not is_face_match:
            messagebox.showerror("ปฏิเสธการเข้าถึง", "ใบหน้าไม่ตรงกับที่ลงทะเบียนไว้!")
            return

        # --- ขั้นที่ 1.5: ใช้ AI (YOLO) ค้นหาและตัดวัตถุ (แบบเผื่อขอบและรักษาสัดส่วน) ---
        watermarked_face = cv2.imread("db_watermarked_face.png", cv2.IMREAD_GRAYSCALE)
        original_hl = np.load("db_hl.npy")
        extracted_wm = extract_watermark(watermarked_face, original_hl)
        
        full_scene_img_gray = cv2.imread(login_img_path, cv2.IMREAD_GRAYSCALE)
        
        # ให้ AI วิเคราะห์รูปภาพ
        results = model_ai(login_img_path, verbose=False)
        
        best_ssim_score = -1
        best_cropped_obj = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                
                # ข้ามหน้าคน
                if class_id == 0:
                    continue
                
                # --- ลอจิกใหม่: สร้างกล่องสี่เหลี่ยมจัตุรัสและเผื่อขอบ (Padding) ---
                w = x2 - x1
                h = y2 - y1
                center_x = x1 + w // 2
                center_y = y1 + h // 2
                
                # หาด้านที่ยาวที่สุด แล้วทำกล่องให้เป็นสี่เหลี่ยมจัตุรัส (ป้องกันขวดบวมตอนย่อขยาย)
                max_side = max(w, h)
                pad = int(max_side * 0.25) # เผื่อพื้นที่สีขาวรอบๆ ออกไป 25% ให้เหมือนต้นฉบับ
                half_size = (max_side // 2) + pad

                new_x1 = max(0, center_x - half_size)
                new_y1 = max(0, center_y - half_size)
                new_x2 = min(full_scene_img_gray.shape[1], center_x + half_size)
                new_y2 = min(full_scene_img_gray.shape[0], center_y + half_size)
                
                cropped_ai = full_scene_img_gray[new_y1:new_y2, new_x1:new_x2]
                
                if cropped_ai.size == 0:
                    continue
                
                resized_crop = cv2.resize(cropped_ai, (WM_SIZE, WM_SIZE))
                
                # เทียบ SSIM
                data_range = extracted_wm.max() - extracted_wm.min()
                if data_range == 0: data_range = 255
                score = ssim_metric(extracted_wm, resized_crop, data_range=data_range)
                
                print(f"AI พบวัตถุ Class {class_id} -> เทียบ SSIM ได้: {score:.4f}")
                
                if score > best_ssim_score:
                    best_ssim_score = score
                    best_cropped_obj = resized_crop

        if best_cropped_obj is not None:
            test_obj_img = best_cropped_obj
        else:
            test_obj_img = preprocess_image(login_img_path, WM_SIZE)
        
        cv2.imwrite("debug_AI_cropped_object.jpg", test_obj_img)

        # --- ขั้นที่ 2: ตรวจสอบ Object Key ---
        # **บังคับผ่านถ้าระบบ AI มั่นใจว่าจับวัตถุได้และ SSIM > 0.20**
        # เนื่องจากงาน DWT Extraction ทำให้เกิด Noise เยอะมาก ค่า 0.20-0.30 ถือว่าเป็นตัวเลขปกติของงานวิจัยลักษณะนี้ครับ
        if best_ssim_score >= 0.50:
            is_key_match = True
        else:
            is_key_match = False
        
        if is_key_match:
            messagebox.showinfo("สำเร็จ", f"ยืนยันตัวตนสำเร็จ!\nAI ตรวจพบ Object Key ถูกต้อง (SSIM: {best_ssim_score:.2f})")
            login_img_path = ""
            lbl_login_img.config(text="รูปที่เลือก: ยังไม่ได้เลือกไฟล์", fg="gray")
            show_frame(frame_home)
        else:
            messagebox.showerror("ปฏิเสธการเข้าถึง", f"Object Key ไม่ถูกต้อง!\n(คะแนน SSIM สูงสุดที่ได้: {best_ssim_score:.2f})")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("ข้อผิดพลาด", f"ระบบเกิดขัดข้อง:\n{str(e)}")
# --- การตั้งค่า GUI ---
root = tk.Tk()
root.title("SUT Pre-CapStone: DWT Watermarking + AI")
root.geometry("450x500")

def show_frame(frame):
    frame.tkraise()

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

frame_home = tk.Frame(root)
frame_register = tk.Frame(root)
frame_login = tk.Frame(root)

for frame in (frame_home, frame_register, frame_login):
    frame.grid(row=0, column=0, sticky='nsew')

# === 1. หน้าหลัก (Home) ===
tk.Label(frame_home, text="ระบบยืนยันตัวตน 2 ขั้นตอน (DWT + AI)", font=("Arial", 16, "bold")).pack(pady=60)
tk.Button(frame_home, text="📝 ลงทะเบียน (Register)", font=("Arial", 12), command=lambda: show_frame(frame_register), bg="#2196F3", fg="white", width=25, height=2).pack(pady=10)
tk.Button(frame_home, text="✅ เข้าสู่ระบบ (Login)", font=("Arial", 12), command=lambda: show_frame(frame_login), bg="#4CAF50", fg="white", width=25, height=2).pack(pady=10)

# === 2. หน้าลงทะเบียน (Register) ===
tk.Label(frame_register, text="หน้าลงทะเบียน", font=("Arial", 14, "bold")).pack(pady=20)
tk.Button(frame_register, text="👤 1. เลือกรูปใบหน้า", command=select_reg_face, width=25).pack(pady=5)
lbl_reg_face = tk.Label(frame_register, text="ใบหน้า: ยังไม่ได้เลือกไฟล์", fg="gray")
lbl_reg_face.pack(pady=2)
tk.Button(frame_register, text="🔑 2. เลือกรูปวัตถุ (Key)", command=select_reg_obj, width=25).pack(pady=5)
lbl_reg_obj = tk.Label(frame_register, text="วัตถุ: ยังไม่ได้เลือกไฟล์", fg="gray")
lbl_reg_obj.pack(pady=2)
tk.Frame(frame_register, height=2, bd=1, relief="sunken", width=350).pack(pady=20)
tk.Button(frame_register, text="ยืนยันการลงทะเบียน", command=register_action, bg="#2196F3", fg="white", width=20, height=2).pack(pady=5)
tk.Button(frame_register, text="⬅️ กลับหน้าหลัก", command=lambda: show_frame(frame_home), width=15).pack(pady=10)

# === 3. หน้าเข้าสู่ระบบ (Login) ===
tk.Label(frame_login, text="หน้าเข้าสู่ระบบ", font=("Arial", 14, "bold")).pack(pady=20)
tk.Label(frame_login, text="* กรุณาแนบ 1 รูปที่มีทั้ง ใบหน้า และ วัตถุ(Key) *", fg="#D32F2F", font=("Arial", 10)).pack(pady=5)
tk.Button(frame_login, text="📸 เลือกรูปภาพเข้าสู่ระบบ", command=select_login_img, width=25).pack(pady=10)
lbl_login_img = tk.Label(frame_login, text="รูปที่เลือก: ยังไม่ได้เลือกไฟล์", fg="gray")
lbl_login_img.pack(pady=2)
tk.Frame(frame_login, height=2, bd=1, relief="sunken", width=350).pack(pady=20)
tk.Button(frame_login, text="ยืนยันเข้าสู่ระบบ", command=login_action, bg="#4CAF50", fg="white", width=20, height=2).pack(pady=5)
tk.Button(frame_login, text="⬅️ กลับหน้าหลัก", command=lambda: show_frame(frame_home), width=15).pack(pady=10)

show_frame(frame_home)
root.mainloop()