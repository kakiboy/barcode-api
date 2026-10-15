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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray)
        
        if not barcodes:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            barcodes = decode(gray)

        if not barcodes:
            return {"status": "error", "message": "สแกนไม่สำเร็จ: ไม่พบบาร์โค้ดในรูปภาพ"}
        
        barcode_data = barcodes[0].data.decode('utf-8')
        return {"status": "success", "data": barcode_data}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}