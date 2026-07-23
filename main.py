import os
import base64
import pandas as pd
from openai import OpenAI
from jiwer import cer

# Koneksi ke LM Studio (sesuai screenshot: http://192.168.56.1:1234 atau localhost)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Path folder dataset sesuai struktur Kakak
IMAGE_DIR = r"dataset/indonesian license plate dataset/images/test"
LABEL_DIR = r"dataset/indonesian license plate dataset/labels/test"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

results = []

# NAMA MODEL WAJIB SAMA PERSIS DENGAN DI LM STUDIO
MODEL_NAME = "google/gemma-3-4b" 

valid_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Ditemukan {len(valid_images)} gambar di folder test. Memulai inferensi...")

for img_name in valid_images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # Ambil ground truth
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, base_name + ".txt")
    
    ground_truth = ""
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            ground_truth = f.read().strip()
    else:
        ground_truth = base_name.split("_")[0]

    base64_image = encode_image(img_path)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the license plate number shown in this image? Respond only with the plate number."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error pada {img_name}: {e}")
        prediction = "ERROR"

    # Hitung Character Error Rate (CER)
    gt_clean = ground_truth if ground_truth else ""
    pred_clean = prediction if prediction else ""
    
    cer_score = cer(gt_clean, pred_clean) if gt_clean else 1.0

    results.append({
        "image": img_name,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "CER_score": cer_score
    })
    print(f"File: {img_name} | GT: {ground_truth} | Pred: {prediction} | CER: {cer_score:.2f}")

# Simpan ke CSV sesuai instruksi UAS[cite: 1]
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("Selesai! Hasil disimpan ke results.csv")