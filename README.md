## 📋 Fitur Utama
* **Integrasi VLM**: Menghubungkan Python dengan model multimodal lokal di LM Studio (menggunakan API OpenAI-compatible).
* **Automated Inference**: Memproses gambar secara otomatis dari direktori dataset test.
* **Evaluasi CER**: Menghitung *Character Error Rate* (CER) untuk mengukur tingkat akurasi prediksi model terhadap *ground truth*[cite: 1].
* **Logging CSV**: Menyimpan hasil evaluasi lengkap ke dalam file `results.csv` dengan kolom: `image`, `ground_truth`, `prediction`, dan `CER_score`[cite: 1].

## 🛠️ Persyaratan Sistem & Instalasi
1. **Python** (versi terbaru terinstal di sistem).
2. **LM Studio** yang menjalankan server lokal dengan model multimodal (misalnya *gemma-3-4b* atau *llava*)[cite: 1].
3. Install dependencies yang dibutuhkan:
   ```bash
   🚀 Cara Menjalankan Program
   pip install -r requirements.txt
Pastikan aplikasi LM Studio sudah aktif dan local server-nya menyala di port 1234.

Pastikan struktur folder dataset sudah sesuai.

Jalankan script utama di terminal:
python main.py
Kalau sudah di-paste, tinggal klik tombol hijau **"Commit changes..."** di sebelah kanan atas buat nyimpen perubahannya[cite: 1]!
