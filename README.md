# 📈 Advanced Multivariate Time Series Forecasting: Bitcoin Price Prediction

**Status Proyek:** LULUS BINTANG 5 (Level: Advanced / Mastery) 🌟🌟🌟🌟🌟  
**Fokus Model:** Seq2Seq LSTM dengan Autoregressive & Self-Attention 

Proyek ini merupakan submission akhir untuk kelas **Deep Learning untuk Time Series (DLTM)** di Dicoding. Proyek ini berfokus pada prediksi harga *Close* Bitcoin selama 24 jam ke depan (*multi-step horizon*) menggunakan data historis multivariat dari tahun 2017 hingga 2023.

---

## 🚀 Pencapaian Kriteria "Advanced" (Level Mahir)

Proyek ini telah disusun sedemikian rupa untuk memenuhi dan melampaui standar *best practice* di industri *Machine Learning*, dengan implementasi tingkat lanjut meliputi:

### 1. Re-implementasi Multi-Head Attention dari Nol (Matematika Murni)
Tidak menggunakan layer bawaan `tf.keras.layers.MultiHeadAttention`. Komponen *Attention* dibangun **sepenuhnya dari nol (from scratch)**. Ini mencakup pembuatan proyeksi *Query, Key, Value* menggunakan layer Dense kustom, pemisahan *heads*, dan perhitungan *Scaled Dot-Product Attention* secara manual menggunakan perkalian matriks dasar (`tf.matmul`) dan `tf.nn.softmax`.

### 2. OOP Custom Callbacks (Dynamic LR & Early Stopping)
Mekanisme penghentian pelatihan dan penjadwalan *Learning Rate* tidak lagi menggunakan struktur prosedural (if-else biasa), melainkan diimplementasikan menggunakan arsitektur **Object-Oriented Programming (OOP)** dengan mewarisi kelas `tf.keras.callbacks.Callback`. Hal ini memastikan *log* tercatat rapi, terstruktur, dan transparan di setiap akhir *epoch*.

### 3. Arsitektur Model Subclassing & Custom Loss
Model **Seq2Seq LSTM** dirancang menggunakan metode *Model Subclassing* (mewarisi `tf.keras.Model`) untuk fleksibilitas maksimal, dilengkapi pendekatan *Teacher Forcing* saat fase pelatihan. Fungsi loss menggunakan **Weighted MAE (Custom Loss)**, di mana error prediksi pada jam yang lebih jauh ke depan (misal jam ke-24) diberi penalti bobot yang lebih besar.

### 4. Eksplorasi Data Lanjutan & Pipeline Efisien
Melakukan *Time Series Decomposition* (mendapatkan komponen *Trend* & *Seasonality*) dan uji **ACF & PACF** menggunakan `statsmodels` untuk menjustifikasi penetapan *Window Size = 24*. Data dimuat secara sangat efisien dan aman dari kebocoran (*data leakage*) menggunakan `tf.data.Dataset` untuk **kedua model** (Baseline dan Seq2Seq).

---

## 📓 Struktur Eksekusi Notebook (Cell by Cell)

Keseluruhan kode dijalankan secara berurutan tanpa *error* dengan struktur:

*   **Bagian 1: Eksplorasi & Preprocessing Data**
    *   Load Dataset & Visualisasi *Heatmap* Korelasi.
    *   Eksplorasi EDA Lanjutan: Plot *Time Series Decomposition* dan uji lag *ACF/PACF*.
    *   Feature Engineering: *Rolling Statistic*.
    *   Pembagian data secara sekuensial & Pipeline `tf.data.Dataset`.
*   **Bagian 2: Arsitektur & Pelatihan Baseline**
    *   Definisi kelas `MyCustomDense` dan arsitektur *Baseline LSTM* (Functional API).
    *   Pelatihan Baseline menggunakan `model.fit()` dengan *pipeline dataset*.
*   **Bagian 3: Arsitektur Model Mahir (Seq2Seq & Custom Attention)**
    *   Definisi matematika murni `MyCustomAttentionFromScratch`.
    *   Definisi arsitektur *Seq2Seq LSTM* melalui *Model Subclassing*.
*   **Bagian 4: OOP Custom Callback & Pelatihan Kustom (Turbo Mode)**
    *   Implementasi `MyAdvancedCallback` dan `weighted_mae_loss`.
    *   Eksekusi pelatihan dengan `tf.GradientTape` dan optimasi `@tf.function`.
*   **Bagian 5: Evaluasi Akhir**
    *   Inference dan visualisasi (*Line Chart* & Tabel Komparasi) untuk **Kedua Model**.
    *   Perhitungan Final MAE yang menembus target penilaian.

---

## 🛠️ Cara Menjalankan Notebook
1. Buka file `Nama_Submission_Akhir_DLTM.ipynb`.
2. Pastikan terhubung dengan internet untuk mengunduh dataset pada sel awal.
3. Notebook ini dapat dijalankan dengan efisien di CPU maupun GPU (T4 direkomendasikan).
4. Jalankan sel secara berurutan (*Run All*). 

---
*Dikembangkan oleh Muhammad Iqbal Saputra untuk Submission Dicoding Academy.*
