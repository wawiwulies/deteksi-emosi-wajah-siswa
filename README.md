# Deteksi Emosi Wajah Siswa Berbasis CNN

Aplikasi web Streamlit untuk mendeteksi emosi wajah siswa menggunakan model CNN berbasis dataset FER2013.

## Fitur

- Input langsung dari kamera perangkat menggunakan `st.camera_input()`.
- Deteksi wajah menggunakan OpenCV Haar Cascade.
- Preprocessing wajah ke grayscale 48x48, normalisasi 0-1, dan reshape ke `(1, 48, 48, 1)`.
- Prediksi 7 kelas emosi berdasarkan dataset siswa SLB Kota Madiun dan FER2013.
- Tampilan gambar asli, area wajah terdeteksi, hasil emosi, confidence, grafik probabilitas, dan saran singkat.
- Fallback loader untuk file weights yang cocok dengan arsitektur CNN yang disediakan.

## Struktur Project

```text
.
├── app.py
├── requirements.txt
├── README.md
├── models/
├── assets/
└── utils/
    ├── __init__.py
    ├── model_loader.py
    └── preprocessing.py
```

## Persiapan Model

Letakkan file model CNN ke folder `models/`.

> Catatan: file `facial_emotion_recognition.ipynb` adalah notebook, bukan file model untuk prediksi. Notebook perlu dijalankan atau diekspor terlebih dahulu menjadi `.keras` atau `.h5`.

Format yang didukung:

- `.keras`
- `.h5`

Contoh:

```text
models/
└── facial_emotion_cnn_retrained.keras
```

Pada notebook yang digunakan untuk training, checkpoint model tersimpan dengan nama:

```python
path = "Facial_expression_weights.keras"
checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_best_only=True)
```

Jika file tersebut belum ada, jalankan notebook sampai training selesai, lalu salin `Facial_expression_weights.keras` ke folder `models/`.

Aplikasi akan otomatis memakai model prioritas dari folder `models/`, terutama `facial_emotion_cnn_retrained.keras`.

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

## Kelas Emosi

| Index | Label Model | Tampilan Indonesia |
| --- | --- | --- |
| 0 | anger | Marah |
| 1 | disgust | Tidak nyaman |
| 2 | fear | Takut |
| 3 | happiness | Senang |
| 4 | sad | Sedih |
| 5 | surprised | Terkejut |
| 6 | neutral | Netral |

## Catatan Teknis

- Model tidak dilatih ulang oleh aplikasi.
- Model dimuat dari folder `models/` menggunakan `@st.cache_resource`.
- Jika `load_model()` gagal, aplikasi mencoba membangun ulang arsitektur fallback dan memuat file sebagai weights.
- Jika file model belum tersedia, aplikasi menampilkan instruksi penempatan file model.

## Disclaimer

Hasil deteksi ini hanya estimasi berdasarkan ekspresi wajah dan bukan diagnosis psikologis.
