# Informatics Employee AI Dashboard 🚀

Proyek ini adalah aplikasi web berbasis Flask yang dikembangkan untuk memenuhi tugas UTS Artificial Intelligence. Aplikasi ini berfungsi untuk memprediksi performa karyawan dan segmentasi data menggunakan berbagai model Machine Learning dan Deep Learning.

## ✨ Fitur Utama
- **Prediksi Performa Karyawan**: Menggunakan model ANN, RNN (LSTM), dan Linear Regression untuk memprediksi skor performa.
- **Prediksi Resign (Klasifikasi)**: Menggunakan model Backpropagation untuk memprediksi probabilitas karyawan mengundurkan diri.
- **Segmentasi Karyawan**: Menggunakan algoritma K-Means untuk mengelompokkan karyawan berdasarkan karakteristik produktivitas.
- **Dashboard Interaktif**: Visualisasi metrik evaluasi model (MAE, RMSE, R2, Akurasi) menggunakan Chart.js.
- **Modern UI/UX**: Desain premium dengan tema *Glassmorphism*, dukungan *Dark/Light Mode*, dan animasi AOS.

## 🧠 Model yang Digunakan
Berdasarkan hasil analisis pada `notebooks/analisis.ipynb`, berikut adalah model yang diimplementasikan:
1. **Linear Regression**: Digunakan sebagai baseline untuk tugas regresi.
2. **Artificial Neural Network (ANN)**: Jaringan saraf tiruan untuk prediksi skor numerik.
3. **RNN (LSTM)**: Model Deep Learning yang dioptimalkan untuk data urutan (model terbaik dengan MAE terendah).
4. **Backpropagation**: Neural Network untuk tugas klasifikasi biner (Resign/Tidak).
5. **K-Means Clustering**: Algoritma unsupervised learning untuk pengelompokan data menjadi 3 klaster utama.

## 📊 Hasil Evaluasi Model
| Model | Metrik | Nilai |
|-------|--------|-------|
| RNN / LSTM | MAE | 0.0128 |
| ANN | MAE | 0.1531 |
| Linear Regression | R² Score | 0.9679 |
| Backpropagation | Akurasi | 90.14% |

## 🛠️ Teknologi
- **Backend**: Python, Flask
- **Frontend**: HTML5, Vanilla CSS, JavaScript, Bootstrap 5
- **Libraries**: TensorFlow, Scikit-Learn, Pandas, NumPy, Joblib, Chart.js, AOS (Animate on Scroll)
- **Deployment**: Railway / Heroku (via Procfile)

## 🚀 Cara Menjalankan Secara Lokal
1. **Clone repository**:
   ```bash
   git clone https://github.com/ReidOates/UTS-Artificial-Intellegence.git
   cd UTS-Artificial-Intellegence
   ```
2. **Buat virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan aplikasi**:
   ```bash
   python app/main.py
   ```
   Buka browser dan akses `http://127.0.0.1:5000`.

## 📂 Struktur Direktori
- `app/`: Berisi kode utama aplikasi Flask (routes, templates, static).
- `data/`: Berisi dataset yang digunakan (CSV).
- `models/`: Berisi file model yang sudah dilatih (.h5, .pkl, .joblib).
- `notebooks/`: Berisi proses analisis data dan pelatihan model (`analisis.ipynb`).
- `requirements.txt`: Daftar pustaka Python yang diperlukan.

---
**Dibuat oleh [ReidOates] untuk Tugas UTS Artificial Intelligence.**
