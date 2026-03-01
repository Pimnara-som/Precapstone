import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

# --- แก้ปัญหา pkg_resources ---
try:
    import pkg_resources
except (ImportError, ModuleNotFoundError):
    import setuptools
    sys.modules['pkg_resources'] = setuptools

# --- นำเข้า AI YOLOv8 และ CNN (PyTorch) ---
try:
    from ultralytics import YOLO
    import torch
    from torchvision import models, transforms
    import torch.nn.functional as F
except ImportError:
    messagebox.showerror("ขาด Library", "กรุณาติดตั้ง AI และ PyTorch ก่อนโดยพิมพ์:\npip install ultralytics torch torchvision")
    sys.exit()

# Import Logic ของโปรเจกต์ (เรียกใช้จากไฟล์เดิมได้เลย)
from auth import verify_face
from core_dwt import preprocess_image, embed_watermark, extract_watermark
from config import IMG_SIZE, WM_SIZE,CNN_SIM_THRESHOLD

# --- ตัวแปรเก็บที่อยู่ไฟล์ ---
reg_face_path = ""
reg_obj_path = ""
login_img_path = ""

# 1. โหลดโมเดล YOLO สำหรับ 'หาตำแหน่งวัตถุ'
print("กำลังโหลดโมเดล AI YOLOv8...")
model_yolo = YOLO("yolov8n.pt") 

# 2. โหลดโมเดล CNN สำหรับ 'เปรียบเทียบความเหมือนแทน SSIM'
print("กำลังโหลดโมเดล CNN (ResNet18)...")
cnn_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn_model.eval() # เซ็ตเป็นโหมดใช้งาน (ไม่ต้องเทรน)

# ตัวแปลงสเกลรูปภาพให้เข้ากับมาตรฐาน CNN
cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_cnn_features(img_array):
    """ฟังก์ชันแปลงรูปภาพเป็น Vector จุดเด่นด้วย CNN"""
    if len(img_array.shape) == 2: # ถ้าเป็น Grayscale ให้แปลงเป็น RGB ปลอมๆ ให้ CNN เข้าใจ
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        
    img_tensor = cnn_transform(img_array).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(img_tensor)
    return features

# ---------------------------------------------------------
# ฟังก์ชัน GUI ฝั่งลงทะเบียน (ไม่ต้องแก้ ใช้ลอจิกเดิม)
# ---------------------------------------------------------
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
        
        reg_face_path = ""
        reg_obj_path = ""
        lbl_reg_face.config(text="ใบหน้า: ยังไม่ได้เลือกไฟล์", fg="gray")
        lbl_reg_obj.config(text="วัตถุ: ยังไม่ได้เลือกไฟล์", fg="gray")
        show_frame(frame_home)
        
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถลงทะเบียนได้:\n{str(e)}")

# ---------------------------------------------------------
# ฟังก์ชัน GUI ฝั่งเข้าสู่ระบบ (อัปเกรดระบบตรวจจับด้วย CNN)
# ---------------------------------------------------------
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

        # --- ขั้นที่ 1.5: สกัดลายน้ำ DWT ออกมาเตรียมไว้ ---
        watermarked_face = cv2.imread("db_watermarked_face.png", cv2.IMREAD_GRAYSCALE)
        original_hl = np.load("db_hl.npy")
        extracted_wm = extract_watermark(watermarked_face, original_hl)
        
        # [จุดเปลี่ยนสำคัญ] แปลงภาพลายน้ำที่ได้ให้กลายเป็น CNN Vector ทิ้งไว้เลย
        wm_features = extract_cnn_features(extracted_wm)
        
        full_scene_img_color = cv2.imread(login_img_path) # ใช้ภาพสีในการเปรียบเทียบ
        
        # ให้ YOLO ช่วยหากรอบวัตถุให้
        results = model_yolo(login_img_path, verbose=False)
        
        best_sim_score = -1.0
        best_cropped_obj = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                
                if class_id == 0: # ข้ามหน้าคน
                    continue
                
                # ลอจิกตีเส้นและเผื่อขอบ Padding ของคุณ Som
                w = x2 - x1
                h = y2 - y1
                center_x = x1 + w // 2
                center_y = y1 + h // 2
                max_side = max(w, h)
                pad = int(max_side * 0.25)
                half_size = (max_side // 2) + pad

                new_x1 = max(0, center_x - half_size)
                new_y1 = max(0, center_y - half_size)
                new_x2 = min(full_scene_img_color.shape[1], center_x + half_size)
                new_y2 = min(full_scene_img_color.shape[0], center_y + half_size)
                
                cropped_color = full_scene_img_color[new_y1:new_y2, new_x1:new_x2]
                if cropped_color.size == 0:
                    continue
                
                # [จุดเปลี่ยนสำคัญ] ใช้ CNN สกัด Vector ออกมาจากกรอบที่ YOLO ตัดมาให้
                crop_features = extract_cnn_features(cropped_color)
                
                # เทียบความเหมือนด้วย Cosine Similarity (ค่า 1 คือเหมือนเป๊ะ, 0 คือไม่เหมือน)
                sim_score = F.cosine_similarity(wm_features, crop_features).item()
                
                print(f"AI พบวัตถุ Class {class_id} -> ได้ CNN Similarity Score: {sim_score:.4f}")
                
                if sim_score > best_sim_score:
                    best_sim_score = sim_score
                    best_cropped_obj = cropped_color

        if best_cropped_obj is not None:
            cv2.imwrite("debug_AI_cropped_object.jpg", best_cropped_obj)

        # --- ขั้นที่ 2: ตรวจสอบความถูกต้อง ---
        # CNN Cosine Similarity ทนทานต่อ Noise ของ DWT ได้ดีกว่ามาก 
        # ตั้งเกณฑ์ที่ 0.65 - 0.70 ถือว่ายืดหยุ่นและปลอดภัยครับ
        # --- ขั้นที่ 2: ตรวจสอบความถูกต้อง ---
        if best_sim_score >= CNN_SIM_THRESHOLD:  # <--- เปลี่ยนเป็นใช้ตัวแปรจาก config
            messagebox.showinfo("สำเร็จ", f"ยืนยันตัวตนสำเร็จ!\nระบบ CNN ยืนยัน Object Key ถูกต้อง\n(ความเหมือน: {best_sim_score:.2f})")
            login_img_path = ""
            lbl_login_img.config(text="รูปที่เลือก: ยังไม่ได้เลือกไฟล์", fg="gray")
            show_frame(frame_home)
        else:
            messagebox.showerror("ปฏิเสธการเข้าถึง", f"Object Key ไม่ถูกต้อง!\n(คะแนนความเหมือนสูงสุด: {best_sim_score:.2f})")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("ข้อผิดพลาด", f"ระบบเกิดขัดข้อง:\n{str(e)}")

# ---------------------------------------------------------
# การตั้งค่า GUI
# ---------------------------------------------------------
root = tk.Tk()
root.title("SUT Pre-CapStone: DWT Watermarking + AI (CNN Edition)")
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
tk.Label(frame_home, text="ระบบยืนยันตัวตน 2 ขั้นตอน (DWT + AI CNN)", font=("Arial", 14, "bold")).pack(pady=60)
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