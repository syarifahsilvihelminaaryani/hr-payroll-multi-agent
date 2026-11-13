# 🧭 Integrated Multi-Agent HR & Payroll Simulation System

Platform simulasi dan tata kelola SDM berbasis multi-agen. Menggabungkan analisis data gaji
([Kaggle: `prince7489/employee-salary-dataset`](https://www.kaggle.com/datasets/prince7489/employee-salary-dataset))
dengan LLM dari **Groq API** dan antarmuka **Streamlit**.

## Alur kerja end-to-end

| Modul | Fungsi |
|---|---|
| 1. Payroll Audit & Anomaly Detection | Sanity check data, outlier global (IQR), outlier per grup (modified z-score), penyimpangan dari tren pengalaman, ketimpangan antar-grup dan gender, ringkasan oleh Agen Auditor |
| 2. Hierarchical Negotiation | Agen Analis HR → Agen Manajer Keuangan → Agen Karyawan bernegosiasi sampai konsensus; angka akhir dibatasi plafon anggaran oleh kode (guardrail), bukan hanya prompt |
| 3. Career Growth Advisor | Kuartil gaji karyawan di grupnya, selisih ke level berikutnya, dan roadmap kompetensi dari LLM |

Data mengalir antar modul: temuan audit masuk ke prompt negosiasi, sisa anggaran dihitung kumulatif,
dan hasil negosiasi ikut dipertimbangkan Penasihat Karier.

## Struktur proyek

```
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/
    └── employee_salary_dataset.csv   # opsional, unduh dari Kaggle
```

## Instalasi lokal

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # lalu isi GROQ_API_KEY (opsional)
```

Buat API key gratis di <https://console.groq.com>.

## Menyiapkan dataset

Pilih salah satu:

1. **Upload di aplikasi** — unduh CSV dari halaman Kaggle di atas, lalu unggah lewat sidebar.
2. **File lokal** — simpan sebagai `data/employee_salary_dataset.csv`, lalu pilih "File lokal data/" di sidebar.
3. **Data contoh** — pilih "Data contoh (sintetis)" untuk mencoba tanpa dataset (bukan data Kaggle asli).

Aplikasi menebak peran kolom (gaji, departemen, pengalaman, dst.) secara otomatis.
Jika nama kolom dataset berbeda, koreksi di **sidebar → Pemetaan kolom**. Hanya kolom gaji yang wajib.

## Menjalankan

```bash
streamlit run app.py
```

Buka <http://localhost:8501>, masukkan Groq API Key di sidebar (atau isi `.env`), klik **Uji koneksi API**.

## Pengaturan penting (sidebar)

- **Model**: default `llama-3.3-70b-versatile`. Model lama seperti `llama3-70b-8192` sudah dihentikan Groq
  menurut pengetahuan saya; cek daftar terbaru di <https://console.groq.com/docs/models>
  dan gunakan opsi "(ketik manual)" bila perlu.
- **Plafon pool kenaikan**: persen dari total payroll bulanan yang boleh dipakai untuk kenaikan.
- **Batas kenaikan individu**: batas atas per karyawan.
- **Ambang audit**: sensitivitas deteksi anomali.

## Deployment ke GitHub & Streamlit Community Cloud

1. Buat repositori GitHub, lalu dorong kode:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: multi-agent HR & payroll simulation"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```
   `.gitignore` sudah mengecualikan `.env`, secrets, dan CSV di `data/`.
2. Buka <https://share.streamlit.io>, login dengan GitHub, klik **Create app**,
   pilih repositori, branch `main`, dan file utama `app.py`.
3. (Opsional) Di **Advanced settings → Secrets**, isi:
   ```toml
   GROQ_API_KEY = "isi_key_anda"
   ```
   Jika tidak diisi, setiap pengguna memasukkan key sendiri di sidebar.
4. Klik **Deploy**. Karena CSV tidak ikut di-commit, pengguna mengunggah dataset lewat sidebar
   (atau gunakan data contoh).

## Keamanan

- Jangan commit API key. Gunakan `.env` (lokal) atau Secrets (cloud).
- Data gaji karyawan nyata bersifat sensitif: jangan unggah ke aplikasi publik, dan ingat bahwa isi prompt
  dikirim ke Groq.

## Keterbatasan

- Keluaran LLM bersifat simulatif; keputusan kompensasi nyata harus ditinjau manusia.
- Selisih gaji antar-gender di Modul 1 bersifat deskriptif, tidak mengontrol faktor lain.
- Dataset Kaggle ini kemungkinan sintetis, sehingga temuan tidak mencerminkan kondisi pasar kerja nyata.
