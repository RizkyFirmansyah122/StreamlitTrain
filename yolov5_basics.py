# from yolov5 import YOLO
# import numpy

# model = YOLO('D:/SEMETER5/StreamlitTrain/best.pt', 'v5')

# detection_output = model.predict(source='D:/SEMETER5/cric/gambar Cric/00b1e59ebc3e7be500ef7548207d44e2.png')

# print(detection_output)

# print(detection_output[0].numpy())

import sys
from pathlib import WindowsPath
import torch

# Tambahkan direktori YOLOv5 ke path Python
sys.path.append(str(WindowsPath('D://SEMETER5//StreamlitTrain//yolov5')))

# Muat model YOLOv5 dari file yang disimpan lokal
model = torch.hub.load(WindowsPath('D://SEMETER5//StreamlitTrain//yolov5'), 'custom', WindowsPath('D://SEMETER5//StreamlitTrain//best.pt'), source='local', autoshape=False, device='cpu', force_reload=True)

# Prediksi pada gambar
img_path = "D://SEMETER5//cric//gambar Cric//00b1e59ebc3e7be500ef7548207d44e2.png"
results = model(img_path)

# Tampilkan hasil deteksi
results.print()  # Print hasil deteksi ke terminal
results.show()   # Tampilkan gambar dengan bounding box
