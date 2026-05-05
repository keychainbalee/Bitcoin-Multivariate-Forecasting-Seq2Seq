# 📈 Multivariate Crypto Forecasting (Bitcoin) - 24 Hours Horizon

**Status Proyek:** LULUS BINTANG 4 (Skilled) 🌟🌟🌟🌟
**Final Test MAE (Seq2Seq):** `0.00773` (Syarat Advanced: `< 0.015`)

Proyek ini merupakan submission akhir untuk kelas **Deep Learning untuk Time Series (DLTM)** di Dicoding. Proyek ini berfokus pada prediksi harga *Close* Bitcoin selama 24 jam ke depan (*multi-step horizon*) menggunakan data historis multivariat dari tahun 2017 hingga 2023.

---

## ✅ Pemenuhan Kriteria Evaluasi Dicoding

Proyek ini telah disusun secara komprehensif untuk memenuhi seluruh kriteria evaluasi mulai dari tingkat *Basic* hingga *Advanced*:

### Kriteria 1: Persiapan Data & Model Baseline (Terpenuhi)
- Menggunakan minimal 3 fitur input dan divisualisasikan korelasi antar fiturnya menggunakan **Heatmap**.
- Memastikan tidak ada *data leakage* dalam pembagian dataset dan proses normalisasi.
- Membuat *pipeline* data yang efisien menggunakan `tf.data.Dataset`.
- Melakukan dekomposisi data target dan menentukan *window size* berdasarkan hasil uji **ACF dan PACF**.
- Melakukan *feature engineering* dengan menambahkan fitur **Rolling Statistic**.
- **[REVISI SELESAI]** Membangun dan melatih model LSTM Baseline menggunakan metode bawaan `model.fit()`, dibuktikan dengan log training yang tersimpan di notebook.

### Kriteria 2: Arsitektur Model Kustom (Terpenuhi)
- Membangun model **LSTM Baseline** (menggunakan Functional API).
- Membangun model **Seq2Seq LSTM** dengan pendekatan *Teacher Forcing* dan *Autoregressive* (menggunakan Model Subclassing).
- Membuat ulang *layer* dari nol (Custom Layer): `MyCustomDense`, `MyCustomAttention`, dan **tambahan custom layer** `MyCustomDropout`.
- Mengaplikasikan *Custom Layers* tersebut ke dalam arsitektur model dan mengimplementasikan fungsi `get_config()` untuk keperluan penyimpanan format `.keras`.

### Kriteria 3: Pelatihan Kustom & Evaluasi (Terpenuhi)
- Membangun *Custom Training Loop* menggunakan `tf.GradientTape` (dioptimasi dengan *Turbo Mode* `@tf.function` untuk performa komputasi).
- Membuat **Custom Loss MAE** (`weighted_mae_loss`) yang memberikan bobot *error* lebih besar pada prediksi langkah yang lebih jauh.
- Membuat logika **Custom Callback** secara manual dari awal: *Early Stopping* (berdasarkan *threshold* target) dan *Reduce Learning Rate on Plateau*.
- **[REVISI SELESAI]** Melakukan inferensi pada data Test untuk **KEDUA MODEL** (Baseline LSTM dan Seq2Seq LSTM).
- **[REVISI SELESAI]** Menampilkan hasil inferensi kedua model dalam bentuk visualisasi *Line Chart* dan *Tabel Komparasi* (Aktual vs Prediksi vs Selisih).
- Mencapai performa **Final MAE murni < 0.015** pada data Test untuk model Seq2Seq.

---

## 📓 Struktur Eksekusi Notebook

Keseluruhan kode dijalankan secara berurutan tanpa *error* dengan struktur sebagai berikut:

1. **Bagian 1: Eksplorasi & Preprocessing Data**
   - *Load Dataset*, Visualisasi *Heatmap*, Uji Dekomposisi & ACF/PACF.
   - Pembuatan fitur *Rolling Statistic*.
   - Standarisasi (*Scaler*) yang di-*fit* hanya pada data *train* untuk mencegah kebocoran data.
   - Pembungkusan data ke dalam format `tf.data.Dataset`.

2. **Bagian 2: Arsitektur & Pelatihan Baseline**
   - Definisi kelas untuk *Custom Layers*.
   - Pembentukan arsitektur *Baseline LSTM* (Functional API).
   - Eksekusi pelatihan *Baseline* dengan `model.fit()`.

3. **Bagian 3: Arsitektur & Pelatihan Seq2Seq (Custom Loop)**
   - Pembentukan arsitektur *Seq2Seq LSTM* (Subclassing).
   - Eksekusi pelatihan dengan *Custom Training Loop* (menampilkan metrik *epoch*, *train loss*, dan *val loss* secara dinamis).
   - Otomatisasi penyimpanan model terbaik (`.keras`).

4. **Bagian 4: Evaluasi Akhir**
   - Pencetakan grafik prediksi dan tabel selisih untuk **Model Baseline**.
   - Pencetakan grafik prediksi dan tabel selisih untuk **Model Seq2Seq**.
   - Perhitungan rata-rata skor evaluasi MAE Final.
   - *Generate* berkas `requirements.txt`.

---

## 🛠️ Cara Menjalankan Notebook
1. Buka file `Nama_Submission_Akhir_DLTM.ipynb`.
2. Pastikan terhubung dengan internet untuk mengunduh dataset pada sel awal.
3. Notebook ini dapat dijalankan menggunakan *Runtime CPU* secara efisien berkat pemanfaatan Graph Execution (`@tf.function`), namun menggunakan GPU (T4) direkomendasikan.
4. Jalankan sel berurutan dari atas ke bawah (*Run All*). Seluruh proses telah diverifikasi bebas dari *error*.

---
*Dibuat untuk Submission Dicoding Academy*
*Di Kembangkan Oleh Muhammad Iqbal Saputra*
