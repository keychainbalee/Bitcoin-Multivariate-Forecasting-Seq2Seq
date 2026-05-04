# 📈 Multivariate Crypto Forecasting (Bitcoin) - 24 Hours Horizon

<!--**Status Proyek:** LULUS BINTANG 5 (Advanced) 🌟🌟🌟🌟🌟-->  
**Final Test MAE:** `0.00582` (Target: `< 0.015`)

Proyek ini adalah submission akhir untuk kelas **Deep Learning Tingkat Lanjut** di Dicoding. Tujuan dari proyek ini adalah membangun model *Deep Learning* tingkat lanjut untuk memprediksi harga *Close* Bitcoin selama 24 jam ke depan menggunakan data historis multivariat (2017 - 2023).

---

## ✨ Fitur Unggulan & Pemenuhan Kriteria (Advanced)
Proyek ini mengimplementasikan teknik-teknik *Advanced Deep Learning*, meliputi:
- **Arsitektur Kompleks:** Model Baseline LSTM (Functional API) & Seq2Seq LSTM dengan Autoregressive / Teacher Forcing (Model Subclassing).
- **Custom Layers:** Pembuatan ulang layer `Dense`, `Multi-Head Attention`, dan `Dropout` dari nol beserta fungsi `get_config()` untuk serialisasi.
- **Custom Training Loop:** Pelatihan model menggunakan `tf.GradientTape` dengan optimasi `@tf.function` (Graph Execution) untuk akselerasi performa (Turbo Mode).
- **Custom Loss & Callback:** Implementasi *Weighted MAE Loss* dan logika *Threshold Early Stopping* serta *Reduce Learning Rate on Plateau* secara kustom.
- **Data Pipeline:** Menggunakan `tf.data.Dataset` untuk optimasi memori dan kecepatan pemrosesan *batch*.

---

## 📓 Penjelasan Struktur Notebook (Cell by Cell)

Keseluruhan *pipeline* proyek dibagi ke dalam 13 sel (cell) eksekusi utama di dalam Jupyter Notebook / Google Colab:

### 📥 Bagian 1: Persiapan & Eksplorasi Data
* **Sel 1: Import Libraries**  
  Memuat semua modul yang dibutuhkan seperti TensorFlow, Pandas, NumPy, Matplotlib, Seaborn, dan Statsmodels.
* **Sel 2: Load Dataset**  
  Mengunduh dan memuat dataset Crypto Multivariate (khusus Bitcoin) ke dalam Pandas DataFrame.
* **Sel 3: Exploratory Data Analysis (EDA) & Correlation Heatmap**  
  Menganalisis distribusi data dan membuat visualisasi *Heatmap* untuk memastikan minimal 3 fitur yang dipilih memiliki korelasi yang baik dengan target (Close).
* **Sel 4: ACF, PACF & Time Series Decomposition**  
  Melakukan uji *Autocorrelation*, *Partial Autocorrelation*, serta membedah komponen tren dan musiman (*seasonality*) untuk menentukan *Window Size* yang optimal.
* **Sel 5: Feature Engineering**  
  Membuat fitur baru menggunakan *Rolling Statistic* (seperti *Rolling Mean* atau *Rolling Std*) untuk menambah sinyal prediksi pada model.

### ⚙️ Bagian 2: Preprocessing & Pipeline
* **Sel 6: Data Splitting, Scaling & `tf.data.Dataset`**  
  Membagi data (Train, Val, Test) secara sekuensial untuk menghindari *Data Leakage*, melakukan standarisasi/normalisasi, dan membungkusnya dalam fungsi *windowing* menggunakan pipeline `tf.data.Dataset` dengan `BATCH_SIZE = 256`.

### 🧠 Bagian 3: Pembangunan Arsitektur Model (Custom)
* **Sel 7: Definisi Custom Layers**  
  Membangun kelas `MyCustomDense`, `MyCustomAttention`, dan `MyCustomDropout(rate=0.1)` dari modul dasar TensorFlow, lengkap dengan metode `get_config()` agar model dapat disimpan ke format `.keras`.
* **Sel 8: Model Baseline LSTM (Functional API)**  
  Merakit arsitektur *baseline* menggunakan lapisan standar digabung dengan *Custom Dense* dan *Custom Attention*.
* **Sel 9: Model Seq2Seq LSTM (Model Subclassing)**  
  Merakit arsitektur tingkat lanjut dengan *Encoder-Decoder LSTM*. Model ini mengimplementasikan *Teacher Forcing* saat *training* dan perulangan prediksi 24 langkah (*horizon*) dengan injeksi *Custom Dropout* untuk mencegah *overfitting*.

### 🚀 Bagian 4: Custom Training & Evaluasi
* **Sel 10: Hyperparameters, Metrics & Custom Loss**  
  Mendefinisikan *Adam Optimizer*, *Stateful Metrics*, dan fungsi *Custom Loss* (`weighted_mae_loss`) yang memberikan pinalti/bobot lebih besar jika model salah menebak di jam-jam yang lebih jauh.
* **Sel 11: Custom Training Loop (Turbo Mode)**  
  Inti dari proses pelatihan! Menggunakan `tf.GradientTape` untuk menghitung gradien secara manual. Dilengkapi dengan:
  - `@tf.function` untuk mempercepat komputasi.
  - *Dynamic Progress Bar* (`tqdm`).
  - *Threshold Early Stopping* (berhenti otomatis jika *Val Loss* < 0.015).
  - Penyimpanan berkas ke `model_baseline_LSTM.keras` dan `best_model_seq2seq_LSTM.keras`.
* **Sel 12: Inference, Visualisasi & Final Evaluation**  
  Melakukan *Autoregressive Inference* pada data uji (Test). Menyajikan *Line Chart* perbandingan aktual vs prediksi, tabel selisih nilai, dan perhitungan **Final MAE murni** sebagai penentu kelulusan.
* **Sel 13: Export Dependencies**  
  Menjalankan perintah `!pip freeze > requirements.txt` untuk memastikan reprodusibilitas *environment*.

---

## 🛠️ Cara Menjalankan (How to Run)
1. Buka `Nama_Submission_Akhir_DLTM.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Pastikan Anda memiliki akses ke internet untuk mengunduh dataset pada blok awal.
3. Disarankan menggunakan akselerator **GPU (T4)** untuk mempercepat proses *training*. Jika menggunakan CPU, *Custom Training Loop* tetap berjalan cepat berkat optimasi `@tf.function` (*Turbo Mode*).
4. Jalankan sel secara berurutan (*Run All*). Notebook dijamin berjalan lancar tanpa *error* dari Sel 1 hingga Sel 13.

---
*Dibuat untuk Submission Dicoding Academy*
*Di Kembangkan Oleh Muhammad Iqbal Saputra*
