import pandas as pd
import qrcode
import os

data = pd.read_csv("urls.csv")

os.makedirs("qr_dataset/phishing", exist_ok=True)
os.makedirs("qr_dataset/legitimate", exist_ok=True)

for _, row in data.iterrows():
    img = qrcode.make(row['url'])
    label = row['label']
    img.save(f"qr_dataset/{label}/qr_{row['id']}.png")

print("QR codes generated successfully")
