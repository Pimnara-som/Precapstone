# main.py
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np

# --- ส่วนแก้ปัญหา pkg_resources สำหรับ Python 3.12 ---
try:
    import pkg_resources
except (ImportError, ModuleNotFoundError):
    import setuptools
    sys.modules['pkg_resources'] = setuptools

# Import Logic ของโปรเจกต์คุณ
from auth import verify_face, verify_object_key
from core_dwt import preprocess_image, embed_watermark, extract_watermark
from config import IMG_SIZE, WM_SIZE

# ตัวแปรเก็บที่อยู่ไฟล์
face_path = ""
obj_path = ""

def select_face():
    global face_path
    path = filedialog.askopenfilename(title="เลือกรูปใบหน้า", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        face_path = path
        label_face_status.config(text=f"ใบหน้า: {os.path.basename(path)}", fg="#2E7D32")

def select_obj():
    global obj_path
    path = filedialog.askopenfilename(title="เลือกรูปวัตถุ (Key)", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        obj_path = path
        label_obj_status.config(text=f"วัตถุ: {os.path.basename(path)}", fg="#2E7D32")

def register_action():
    if not face_path or not obj_path:
        messagebox.showwarning("เตือน", "กรุณาเลือกรูปให้ครบทั้ง ใบหน้า และ วัตถุ ครับ")
        return
    
    try:
        # 1. โหลดและปรับขนาดรูปตาม config
        face_img = preprocess_image(face_path, IMG_SIZE)
        obj_img = preprocess_image(obj_path, WM_SIZE)
        
        # 2. ฝังลายน้ำด้วย DWT
        watermarked_face, hl_band = embed_watermark(face_img, obj_img)
        
        # 3. เซฟข้อมูลลงเครื่อง (จำลองเป็น Database สำหรับใช้ยืนยันตัวตน)
        cv2.imwrite("db_watermarked_face.png", watermarked_face) # เก็บรูปใบหน้าที่ฝังลายน้ำแล้ว
        np.save("db_hl.npy", hl_band)                            # เก็บค่า HL สำหรับใช้สกัดลายน้ำ
        cv2.imwrite("db_original_face.jpg", cv2.imread(face_path)) # เก็บใบหน้าต้นฉบับไว้เทียบ Face Recog
        
        messagebox.showinfo("สำเร็จ", "ลงทะเบียนและฝังลายน้ำ (DWT) เรียบร้อยแล้ว!")
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถลงทะเบียนได้:\n{str(e)}")

def login_action():
    if not face_path or not obj_path:
        messagebox.showwarning("เตือน", "กรุณาเลือกรูปให้ครบทั้ง ใบหน้า และ วัตถุ ครับ")
        return
    
    if not os.path.exists("db_watermarked_face.png") or not os.path.exists("db_hl.npy"):
        messagebox.showerror("ล้มเหลว", "ไม่พบข้อมูลในระบบ กรุณาลงทะเบียนก่อน!")
        return

    try:
        # --- ขั้นที่ 1: ตรวจสอบใบหน้า (Face Recognition) ---
        is_face_match = verify_face("db_original_face.jpg", face_path)
        if not is_face_match:
            messagebox.showerror("ปฏิเสธการเข้าถึง", "ใบหน้าไม่ตรงกับที่ลงทะเบียนไว้!")
            return

        # --- ขั้นที่ 2: สกัดและตรวจสอบลายน้ำ (DWT Extraction & SSIM) ---
        watermarked_face = cv2.imread("db_watermarked_face.png", cv2.IMREAD_GRAYSCALE)
        original_hl = np.load("db_hl.npy")
        
        # สกัด Object Key ออกมาจากภาพที่ฝังลายน้ำ
        extracted_wm = extract_watermark(watermarked_face, original_hl)
        
        # โหลดภาพวัตถุที่อัปโหลดมาใหม่เพื่อเปรียบเทียบ
        test_obj_img = preprocess_image(obj_path, WM_SIZE)
        
        # ตรวจสอบด้วย SSIM
        is_key_match = verify_object_key(extracted_wm, test_obj_img)
        
        if is_key_match:
            messagebox.showinfo("สำเร็จ", "ยืนยันตัวตนสำเร็จ!\nใบหน้าและ Object Key ถูกต้อง")
        else:
            messagebox.showerror("ปฏิเสธการเข้าถึง", "Object Key ไม่ถูกต้อง (ลายน้ำไม่ตรงกัน)!")
            
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"ระบบเกิดขัดข้อง:\n{str(e)}")

# --- สร้าง GUI ---
root = tk.Tk()
root.title("SUT Pre-CapStone: DWT Watermarking")
root.geometry("400x450")

tk.Label(root, text="ระบบยืนยันตัวตน 2 ขั้นตอน (DWT)", font=("Arial", 14, "bold")).pack(pady=20)

# ส่วนเลือกไฟล์
tk.Button(root, text="👤 เลือกรูปใบหน้า", command=select_face, width=20).pack(pady=5)
label_face_status = tk.Label(root, text="ใบหน้า: ยังไม่ได้เลือกไฟล์", fg="gray")
label_face_status.pack(pady=2)

tk.Button(root, text="🔑 เลือกรูปวัตถุ (Key)", command=select_obj, width=20).pack(pady=5)
label_obj_status = tk.Label(root, text="วัตถุ: ยังไม่ได้เลือกไฟล์", fg="gray")
label_obj_status.pack(pady=2)

# เส้นแบ่ง
tk.Frame(root, height=2, bd=1, relief="sunken", width=350).pack(pady=20)

# ส่วนปุ่มทำงาน
frame_buttons = tk.Frame(root)
frame_buttons.pack()

tk.Button(frame_buttons, text="📝 ลงทะเบียน\n(ฝังลายน้ำ)", command=register_action, 
          bg="#2196F3", fg="black", width=15, height=2).grid(row=0, column=0, padx=10)

tk.Button(frame_buttons, text="✅ เข้าสู่ระบบ\n(ตรวจสอบ)", command=login_action, 
          bg="#4CAF50", fg="black", width=15, height=2).grid(row=0, column=1, padx=10)

root.mainloop()