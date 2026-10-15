from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pyzbar.pyzbar import decode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scan")
async def scan_barcode(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "ไม่สามารถอ่านไฟล์รูปภาพได้"}

        # เปลี่ยนเป็นภาพขาวดำ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # วิธีที่ 1: อ่านจากภาพเทาปกติ
        barcodes = decode(gray)
        
        # วิธีที่ 2: ถ้าไม่เจอ ให้ใช้ Thresholding (ตัดแสงและเงา ทำให้พื้นหลังขาวสนิท แท่งบาร์โค้ดดำสนิท)
        if not barcodes:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            barcodes = decode(thresh)
            
        # วิธีที่ 3: ถ้ายังไม่เจออีก ให้ลองย่อขนาดภาพลง 50% (แก้ปัญหาภาพจากกล้องมือถือที่ละเอียดเกินไปจนเห็น Noise)
        if not barcodes:
            height, width = gray.shape
            resized = cv2.resize(gray, (width // 2, height // 2))
            barcodes = decode(resized)

        if not barcodes:
            return {"status": "error", "message": "สแกนไม่สำเร็จ: ไม่พบบาร์โค้ดในรูปภาพ กรุณาถ่ายให้ชัดเจนขึ้น"}
        
        barcode_data = barcodes[0].data.decode('utf-8')
        return {"status": "success", "data": barcode_data}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}