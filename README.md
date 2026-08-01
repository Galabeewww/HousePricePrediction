# Tutorial Menjalankan Program — Tugas Besar Rekayasa Fitur

## 1. Apakah kedua file terhubung?

**Ya.** Keduanya terhubung lewat satu file perantara: `train.csv`.

```
00_generate_dataset.py  --- membuat --->  train.csv  --- dibaca oleh --->  01_feature_engineering_modeling.py
   (skrip pembuat data)                  (data)                          (skrip utama: FE + modeling)
```

- `00_generate_dataset.py` **tidak melakukan analisis apa pun**. Tugasnya cuma satu: membuat file `train.csv` (data simulasi bergaya Ames Housing) di folder yang sama tempat ia dijalankan.
- `01_feature_engineering_modeling.py` adalah **file utama** yang benar-benar mengerjakan tugas besar (pra-pemrosesan, rekayasa fitur, pemodelan, evaluasi, visualisasi). File ini **membaca `train.csv`** di baris paling awal lewat `pd.read_csv("train.csv")`.

Artinya: **file 01 tidak akan bisa jalan kalau `train.csv` belum ada** di folder yang sama. Ada dua cara menyediakan `train.csv`:

| Cara                                   | Kapan dipakai                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| Jalankan `00_generate_dataset.py` dulu | Kalau kamu belum punya dataset asli dari Kaggle, atau hanya ingin menguji/coba kode |
| Taruh `train.csv` asli dari Kaggle     | Direkomendasikan untuk laporan final yang dikumpulkan ke dosen                      |

Kedua cara menghasilkan file dengan nama dan skema kolom yang sama persis, jadi **file 01 tidak perlu diubah kodenya sama sekali** — tinggal jalankan.

---

## 2. Persiapan (sekali saja)

### 2.1 Pastikan Python terpasang

Buka terminal / command prompt, ketik:

```bash
python3 --version
```

Kalau muncul versi (mis. `Python 3.10.x`), Python sudah siap. Kalau belum ada, unduh dari [python.org](https://www.python.org/downloads/).

### 2.2 Pasang library yang dibutuhkan

Jalankan sekali di terminal:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Kalau muncul error izin akses, tambahkan `--user`:

```bash
pip install --user pandas numpy scikit-learn matplotlib
```

### 2.3 Kumpulkan file dalam satu folder

Buat satu folder khusus, misalnya `tugas-besar-rekayasa-fitur`, lalu masukkan **kedua file `.py`** ke dalamnya:

```
tugas-besar-rekayasa-fitur/
├── 00_generate_dataset.py
└── 01_feature_engineering_modeling.py
```

Semua file hasil (dataset, grafik, tabel) nanti otomatis tersimpan di folder yang sama ini.

---

## 3. Menjalankan program — langkah demi langkah

Buka terminal, masuk ke folder tadi:

```bash
cd path/ke/tugas-besar-rekayasa-fitur
```

### OPSI 1

### Langkah 1 — Buat dataset

```bash
python3 00_generate_dataset.py
```

**Yang terjadi:** skrip membuat 1.200 baris data rumah simulasi dan menyimpannya sebagai `train.csv` di folder yang sama. Kamu akan melihat output seperti:

```
BERHASIL: train.csv dibuat -> 1200 baris, 33 kolom
...
Langkah selanjutnya: jalankan
    python3 01_feature_engineering_modeling.py
```

> Kalau kamu sudah punya `train.csv` **asli** dari Kaggle, lewati langkah ini — cukup taruh file itu di folder yang sama dengan nama persis `train.csv`.

### Langkah 2 — Jalankan pipeline utama

```bash
python3 01_feature_engineering_modeling.py
```

**Yang terjadi**, tahap demi tahap (progress tercetak sebagai `[1/6]` sampai `[6/6]`):

1. **Memuat data** dari `train.csv`
2. **Pra-pemrosesan** — mengisi nilai yang hilang (missing value)
3. **Rekayasa fitur** — membuat fitur baru, binning, encoding
4. **Split & scaling** — bagi data latih/uji, standarisasi, pilih 20 fitur terbaik
5. **Pemodelan & evaluasi** — melatih Linear Regression & Random Forest, hitung RMSE/MAE/R²
6. **Visualisasi** — membuat 3 grafik hasil

### Langkah 3 — Cek hasilnya

Setelah selesai, folder akan berisi file baru ini:

| File                           | Isi                                             |
| ------------------------------ | ----------------------------------------------- |
| `model_results.csv`            | Tabel RMSE, MAE, R² untuk tiap model            |
| `feature_importances.csv`      | Skor pentingnya tiap fitur (dari Random Forest) |
| `plot_actual_vs_predicted.png` | Grafik harga aktual vs prediksi                 |
| `plot_model_comparison.png`    | Grafik perbandingan RMSE & R² antar model       |
| `plot_feature_importance.png`  | Grafik top 10 fitur terpenting                  |

Buka file-file `.png` dan `.csv` tersebut untuk melihat hasilnya. Angka dan grafik inilah yang sudah dimasukkan ke laporan Word.

### OPSI 2 (Program Dieksekusi menggunakan streamlit)

### Langkah 1 — Jalankan Program

```bash
python -m streamlit run 01_feature_engineering_modeling.py
```

**Yang terjadi:** Pada mode ini, program dijalankan melalui antarmuka web Streamlit. Jika kamu sudah memiliki file train.csv asli dari Kaggle, cukup letakkan file pada kolom unggah train.csv, lalu program akan langsung mengeluarkan hasil dari file train.csv yang sudah dijalankan.

### Langkah 2 — Cek hasilnya

Setelah selesai, maka akan keluar output berupa:

| File                           | Isi                                             |
| ------------------------------ | ----------------------------------------------- |
| `model_results.csv`            | Tabel RMSE, MAE, R² untuk tiap model            |
| `feature_importances.csv`      | Skor pentingnya tiap fitur (dari Random Forest) |
| `plot_actual_vs_predicted.png` | Grafik harga aktual vs prediksi                 |
| `plot_model_comparison.png`    | Grafik perbandingan RMSE & R² antar model       |
| `plot_feature_importance.png`  | Grafik top 10 fitur terpenting                  |

Buka atau download file-file `.png` dan `.csv` tersebut untuk melihat hasilnya. Angka dan grafik inilah yang sudah dimasukkan ke laporan Word.

---

## 4. Cara mengganti dengan dataset ASLI dari Kaggle (opsional, direkomendasikan)

1. Buka https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data (perlu akun Kaggle, gratis).
2. Unduh file `train.csv`.
3. Salin file itu ke folder `tugas-besar-rekayasa-fitur/`, **timpa** `train.csv` yang lama (hasil simulasi).
4. Jalankan ulang **hanya langkah 2** (tidak perlu jalankan `00_generate_dataset.py` lagi):
   ```bash
   python3 01_feature_engineering_modeling.py
   ```
5. Semua tabel, metrik, dan grafik akan otomatis dihitung ulang berdasarkan data asli. Salin angka & grafik baru itu ke laporan Word untuk hasil final yang paling akurat.

---

## 5. Troubleshooting (masalah umum)

| Pesan error                                                                                                              | Penyebab                                                                                                                                                                                                  | Solusi                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: train.csv`                                                                                           | File 01 dijalankan sebelum file 00, atau beda folder                                                                                                                                                      | Jalankan `00_generate_dataset.py` dulu di folder yang sama                                                                                                      |
| `ModuleNotFoundError: No module named 'sklearn'`                                                                         | Library belum terpasang                                                                                                                                                                                   | Jalankan `pip install pandas numpy scikit-learn matplotlib`                                                                                                     |
| `python3: command not found`                                                                                             | Python belum terpasang / pakai perintah `python` bukan `python3` (umum di Windows)                                                                                                                        | Coba `python 00_generate_dataset.py`                                                                                                                            |
| Grafik `.png` tidak muncul / kosong                                                                                      | Belum sempat cek folder setelah program selesai                                                                                                                                                           | Refresh folder file explorer, cek nama file `plot_*.png`                                                                                                        |
| `ValueError: could not convert string to float: 'RL'` (atau nilai teks lain) saat pakai `train.csv` **asli** dari Kaggle | Versi kode sebelumnya hanya mengenali sedikit kolom kategorikal secara manual, padahal data asli Kaggle punya 81 kolom dengan puluhan kolom teks tambahan (`MSZoning`, `Exterior1st`, `Foundation`, dll.) | Sudah diperbaiki — unduh ulang `01_feature_engineering_modeling.py` versi terbaru; sekarang kode **otomatis mendeteksi semua kolom teks**, berapa pun jumlahnya |
| `streamlit: command not found / 'streamlit' is not recognized as an internal or external command`                        | Streamlit belum terpasang atau PATH tidak dikenali                                                                                                                                                        | install `pip install streamlit` lalu jalankan ulang terminal. Jika tetap tidak bisa, aktifkan virtual environment sebelum menjalankan perintah                  |
| `Warning missing ScriptRunContext saat jalankan dengan python file.py`                                                   | Streamlit tidak boleh dijalankan langsung dengan python, harus lewat command khusus                                                                                                                       | gunakan `streamlit run 01_feature_engineering_modeling.py`                                                                                                      |
| `Aplikasi tidak terbuka di browser`                                                                                      | Port default (8501) tidak otomatis terbuka, atau firewall memblokir                                                                                                                                       | Buka manual link `http://localhost:8501` di browser. Jika bentrok port, jalankan dengan `streamlit run 01_feature_engineering_modeling.py --server.port 8502`   |

---

## 6. Ringkasan alur singkat

```
1. pip install pandas numpy scikit-learn matplotlib
2. python3 00_generate_dataset.py                 -> hasil: train.csv
3. python3 01_feature_engineering_modeling.py      -> hasil: model_results.csv, feature_importances.csv, plot_*.png
4. (opsional) ganti train.csv dengan data asli Kaggle, ulangi langkah 3
```
