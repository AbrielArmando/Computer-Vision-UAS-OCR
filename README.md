# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (VLM)

Proyek UAS mata kuliah **Computer Vision (RE604)** — Program Studi Teknik Robotika, Politeknik Negeri Batam.

Program ini melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan Indonesia menggunakan **Visual Language Model (VLM)** yang dijalankan secara lokal melalui **LM Studio**, diintegrasikan dengan Python melalui OpenAI-compatible API.

## Deskripsi Singkat

Setiap gambar dari dataset dikirim ke model multimodal (`google/gemma-3-4b`) yang berjalan di LM Studio dengan prompt:

> "What is the license plate number shown in this image? Respond only with the plate number text without any other words."

Hasil prediksi model kemudian dibandingkan dengan ground truth menggunakan metrik **Character Error Rate (CER)**.

## Dataset

[Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset) (folder `test`).

Struktur yang digunakan:
```
dataset/Indonesian License Plate Dataset/
├── images/test/            # gambar kendaraan (.jpg)
└── labelswithLP/test/      # label YOLO (.txt): class_id x y w h plate_text
```

Setiap baris pada file label berisi bounding box (format YOLO) diikuti teks plat nomor sebagai token terakhir. Jika satu gambar memiliki lebih dari satu plat terdeteksi, program memilih plat dengan **bounding box terbesar** sebagai ground truth utama (diasumsikan sebagai plat yang paling menonjol/dekat kamera, sehingga paling mungkin dibaca oleh VLM).

## Metrik Evaluasi: Character Error Rate (CER)

```
CER = (S + D + I) / N
```

- S = jumlah karakter salah substitusi
- D = jumlah karakter yang dihapus
- I = jumlah karakter yang disisipkan
- N = jumlah karakter pada ground truth

Perhitungan CER dilakukan menggunakan library `jiwer`, dengan teks ground truth dan prediksi dinormalisasi terlebih dahulu (huruf kapital semua, tanpa spasi) agar perbedaan format penulisan spasi tidak dihitung sebagai kesalahan karakter.

## Instalasi

1. Install dependensi Python:
   ```bash
   pip install openai jiwer
   ```

2. Install dan jalankan [LM Studio](https://lmstudio.ai/), lalu:
   - Download model multimodal, misalnya `google/gemma-3-4b` (atau model VLM lain seperti `llava`, `bakllava`).
   - Jalankan Local Server di LM Studio pada port `1234` (default).

3. Download dataset dari Kaggle dan ekstrak ke folder `dataset/` sesuai struktur di atas.

## Cara Menjalankan

```bash
python main.py
```

Program akan:
1. Membaca seluruh gambar pada folder `dataset/Indonesian License Plate Dataset/images/test`.
2. Mengirim tiap gambar ke model VLM lokal via LM Studio.
3. Membandingkan hasil prediksi dengan ground truth menggunakan CER.
4. Menyimpan seluruh hasil ke `results.csv`.

## Format Output (`results.csv`)

| Kolom | Deskripsi |
|---|---|
| `image` | Nama file gambar |
| `ground_truth` | Teks plat nomor asli (dari label dataset) |
| `prediction` | Hasil OCR dari VLM |
| `CER_score` | Character Error Rate antara ground truth dan prediksi |

## Ringkasan Hasil

- Total gambar dievaluasi: **100**
- Rata-rata CER: **±0.33**
- Model berhasil membaca beberapa plat dengan sempurna (CER = 0), namun juga terdapat beberapa kasus gagal total (CER > 1) di mana model menghasilkan teks yang jauh berbeda dari plat asli — kemungkinan disebabkan oleh kualitas gambar, sudut pengambilan, atau keterbatasan kemampuan model VLM kecil (`gemma-3-4b`) dalam membaca teks pada resolusi rendah.

Penjelasan lengkap beserta contoh kasus sukses dan gagal tersedia pada video penjelasan proyek (link disertakan pada pengumpulan e-learning).

## Struktur Repository

```
.
├── main.py           # Script utama: inferensi VLM + evaluasi CER
├── results.csv        # Hasil evaluasi lengkap
└── README.md
```

## Referensi

- [LM Studio Python SDK — Image Input](https://lmstudio.ai/docs/python/llm-prediction/image-input)
- [jiwer — Python library for CER/WER](https://github.com/jitsi/jiwer)
