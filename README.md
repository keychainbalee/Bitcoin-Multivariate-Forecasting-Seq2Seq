# Multivariate Time Series Forecasting: Bitcoin Price Prediction

**Target Penyelesaian:** Kelas Deep Learning untuk Time Series (DLTM) - Dicoding  
**Arsitektur Utama:** Baseline LSTM & Seq2Seq LSTM (Autoregressive)

Repositori ini berisi *notebook* untuk memprediksi harga *Close* Bitcoin dalam horizon 24 jam ke depan menggunakan data multivariat. Seluruh *pipeline* dirancang telah memenuhi kriteria evaluasi tingkat *Skilled* (Bintang 4 | ⭐⭐⭐⭐).

---

## 🛠️ Implementasi Teknis dalam Notebook

Berikut adalah fitur dan kriteria yang **telah diterapkan** secara penuh di dalam kode:

### 1. Pemrosesan Data & Eksplorasi (EDA)
*   **Visualisasi EDA:** Menggunakan *Heatmap* untuk korelasi fitur.
*   **Analisis Lanjutan:** Melakukan *Time Series Decomposition* (mendapatkan *Trend* & *Seasonality*) dan uji **ACF & PACF** menggunakan `statsmodels` untuk penentuan *Window Size* (berada di Sel 3).
*   **Feature Engineering:** Menambahkan fitur baru berupa *Rolling Statistic*.
*   **Pipeline Data:** Menggunakan `tf.data.Dataset` untuk proses *training* pada **kedua model** (Baseline dan Seq2Seq) agar efisien dan seragam.

### 2. Arsitektur Model & Custom Layers
*   **Baseline LSTM:** Dibangun menggunakan *Functional API* dan dilatih menggunakan metode `model.fit()`.
*   **Seq2Seq LSTM:** Dibangun menggunakan metode **Model Subclassing** (`tf.keras.Model`).
*   **Custom Layers:** 
    *   `MyCustomDense`
    *   `MyCustomDropout`
    *   `MyCustomAttentionFromScratch`: Re-implementasi *Multi-Head Attention* secara manual dari nol menggunakan operasi matematika matriks (`tf.matmul`), **tanpa** menggunakan `tf.keras.layers.MultiHeadAttention` bawaan Keras.

### 3. Pelatihan Kustom (Custom Training)
*   **Custom Training Loop:** Menggunakan `tf.GradientTape` dengan optimasi `@tf.function` untuk model Seq2Seq.
*   **Custom Loss:** Menggunakan fungsi `weighted_mae_loss` yang memberikan bobot/penalti lebih besar untuk *error* prediksi di langkah (*step*) yang lebih jauh.
*   **OOP Custom Callback:** Mengimplementasikan kelas `MyAdvancedCallback` (mewarisi `tf.keras.callbacks.Callback`) untuk mengatur logika *Early Stopping* dan *Learning Rate Scheduler* secara terstruktur.

### 4. Evaluasi Akhir
*   Melakukan inferensi/prediksi pada data uji (Test) untuk **kedua model** (Baseline dan Seq2Seq).
*   Menampilkan hasil prediksi dalam bentuk plot *line chart* dan tabel perbandingan aktual vs prediksi beserta nilai selisihnya.
*   Mencapai skor Evaluasi Final MAE pada model Seq2Seq yang memenuhi standar evaluasi.

---

## 🚀 Cara Penggunaan
1. Buka *notebook* `Muhammad Iqbal Saputra_Submission_Akhir_DLTM.ipynb`.
2. Jalankan sel secara berurutan (*Run All*). 
3. Log *training*, grafik evaluasi, dan model berformat `.keras` akan otomatis ter-generate pada akhir eksekusi.
