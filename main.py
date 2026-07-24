import os
import base64
import csv
from openai import OpenAI
from jiwer import cer

# 1. Inisialisasi client OpenAI untuk terhubung ke local server LM Studio
# Pastikan port LM Studio menyala di 1234
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# 2. Definisikan direktori dataset test
image_dir = "dataset/Indonesian License Plate Dataset/images/test"
label_dir = "dataset/Indonesian License Plate Dataset/labelswithLP/test"
output_csv = "results.csv"

# Fungsi untuk encode gambar ke base64 agar bisa dikirim ke VLM
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def parse_label_file(txt_path):
    """
    Parse file label YOLO: setiap baris berformat
        class_id x_center y_center width height plate_text
    Mengembalikan list of dict: [{"plate": str, "area": float}, ...]
    Area (w*h) dipakai untuk menentukan plat mana yang paling
    menonjol jika ada lebih dari satu plat dalam satu gambar.
    """
    plates = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                # baris tidak sesuai format yang diharapkan, lewati
                continue
            try:
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            plate_text = parts[-1]  # teks plat = token terakhir
            plates.append({"plate": plate_text, "area": w * h})
    return plates


def get_ground_truth_text(txt_path):
    """
    Ambil teks plat nomor bersih dari file label.
    Jika ada beberapa plat terdeteksi di satu gambar, pilih yang
    bounding box-nya paling besar (plat paling menonjol/dekat kamera)
    sebagai ground truth utama untuk dibandingkan dengan jawaban VLM.
    """
    plates = parse_label_file(txt_path)
    if not plates:
        return "", []
    main_plate = max(plates, key=lambda p: p["area"])["plate"]
    all_plates = [p["plate"] for p in plates]
    return main_plate, all_plates


def normalize_plate(text):
    """Normalisasi teks plat untuk perbandingan CER yang adil:
    huruf besar semua + hapus spasi, karena spasi hanyalah gaya
    penulisan (mis. 'BG1352AE' vs 'BG 1352 AE') dan bukan kesalahan
    karakter OCR yang sesungguhnya."""
    return text.upper().replace(" ", "")


results = []

# Ambil semua file gambar di folder test
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"Memproses {len(image_files)} gambar dari dataset...")

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)

    # Cari file label ground truth yang bersesuaian (.txt)
    base_name = os.path.splitext(img_file)[0]
    txt_path = os.path.join(label_dir, f"{base_name}.txt")

    if not os.path.exists(txt_path):
        print(f"Warning: Label untuk {img_file} tidak ditemukan.")
        continue

    ground_truth, all_plates = get_ground_truth_text(txt_path)

    if not ground_truth:
        print(f"Warning: Ground truth untuk {img_file} kosong/tidak valid.")
        continue

    if len(all_plates) > 1:
        print(f"Info: {img_file} punya {len(all_plates)} plat terdeteksi {all_plates}, "
              f"dipakai yang terbesar: {ground_truth}")

    # Encode gambar ke base64
    base64_image = encode_image(img_path)

    try:
        # Kirim request ke model VLM lokal di LM Studio
        response = client.chat.completions.create(
            model="google/gemma-3-4b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the license plate number shown in this image? Respond only with the plate number text without any other words."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.0
        )

        prediction = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error pada gambar {img_file}: {e}")
        prediction = "ERROR"

    # Hitung Character Error Rate (CER) antara ground_truth dan prediction
    # Dinormalisasi (uppercase, tanpa spasi) supaya perbandingan adil
    cer_score = cer(normalize_plate(ground_truth), normalize_plate(prediction))

    # Simpan hasil
    results.append({
        "image": img_file,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "CER_score": cer_score
    })

    print(f"Selesai: {img_file} | GT: {ground_truth} | Pred: {prediction} | CER: {cer_score:.4f}")

# 3. Tulis hasil lengkap ke file results.csv
with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["image", "ground_truth", "prediction", "CER_score"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nSelesai! Seluruh hasil evaluasi berhasil disimpan ke {output_csv}")