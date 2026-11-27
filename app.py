"""
Integrated Multi-Agent HR & Payroll Simulation System
=====================================================
Platform simulasi dan tata kelola SDM berbasis multi-agen.

Alur end-to-end (data yang sama mengalir antar modul):
    Data gaji  ->  Data Readiness & Quality Check
               ->  Modul 1: Audit & deteksi anomali (+ What-If Simulator kebijakan & anggaran)
               ->  Modul 2: Negosiasi hierarkis multi-agen (HR vs Keuangan vs Karyawan)
               ->  Modul 3: Penasihat karier (kuartil gaji + jalur kompetensi)
               ->  Tab 4  : Ringkasan eksekutif & ekspor (.md / .txt / .csv)
    Riwayat sesi disimpan di st.session_state (dibatasi MAX_HISTORY entri).

Dataset acuan : Kaggle `prince7489/employee-salary-dataset`
LLM engine    : Groq API (library resmi `groq`)
UI            : Streamlit

Jalankan: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# python-dotenv bersifat opsional: dipakai bila ada file .env lokal.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Library groq diimpor defensif agar aplikasi tetap menampilkan pesan yang jelas
# apabila dependency belum terpasang.
try:
    import groq
    from groq import Groq
except ImportError:  # pragma: no cover
    groq = None
    Groq = None


# =============================================================================
# 0. KONFIGURASI & KONSTANTA
# =============================================================================

APP_TITLE = "Integrated Multi-Agent HR & Payroll Simulation System"
DEFAULT_DATA_PATH = Path("data") / "employee_salary_dataset.csv"

# Dugaan nama kolom untuk pemetaan otomatis. Pencocokan dilakukan setelah nama kolom
# dinormalkan (huruf kecil, hanya huruf/angka), sehingga "Monthly Salary", "monthly_salary",
# dan "Monthly-Salary" dianggap sama. Pengguna tetap dapat mengoreksi lewat sidebar.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "salary": ["monthly_salary", "salary", "gaji", "gaji_pokok", "gaji_bulanan", "annual_salary", "ctc",
               "salary_inr", "income", "pendapatan", "upah", "compensation", "base_salary"],
    "department": ["department", "dept", "departemen", "divisi", "division", "bagian"],
    "position": ["job_title", "jobtitle", "position", "role", "designation", "job_role", "title", "jabatan", "posisi"],
    "experience": ["experience_years", "experience", "years_of_experience", "years_experience", "tenure",
                   "masa_kerja", "lama_kerja", "pengalaman"],
    "education": ["education_level", "education", "pendidikan", "degree", "jenjang"],
    "age": ["age", "umur", "usia"],
    "gender": ["gender", "sex", "jenis_kelamin", "kelamin"],
    "city": ["city", "location", "kota", "lokasi", "domisili"],
    "employee_id": ["employee_id", "emp_id", "empid", "id", "id_karyawan", "nik", "nip", "staff_id"],
    "performance": ["performance_score", "performance", "performance_rating", "rating", "kinerja", "kpi",
                    "skor_kinerja"],
}

FIELD_LABELS: dict[str, str] = {
    "salary": "Gaji (wajib)",
    "department": "Departemen",
    "position": "Posisi / Jabatan",
    "experience": "Pengalaman (tahun)",
    "education": "Pendidikan",
    "age": "Usia",
    "gender": "Gender",
    "city": "Kota",
    "employee_id": "ID Karyawan",
    "performance": "Skor performa",
}

# Sembilan kolom penting yang diperiksa pada Data Readiness (position = opsional tambahan).
ESSENTIAL_FIELDS = ["salary", "department", "experience", "education", "age", "gender", "city", "employee_id", "performance"]
NUMERIC_FIELDS = ("salary", "experience", "age", "performance")

NONE_OPTION = "(tidak ada)"

# Model aktif Groq. Default: openai/gpt-oss-120b.
MODEL_OPTIONS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "(ketik manual)"]
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Model gpt-oss memakai token untuk "reasoning"; beri ruang tambahan agar jawaban tidak terpotong/kosong.
REASONING_HEADROOM = 1200

# Riwayat sesi dibatasi agar session_state tetap ringan.
MAX_HISTORY = 15
WIB = timezone(timedelta(hours=7))

# Strategi alokasi pada What-If Simulator.
STRAT_CORRECT = "Koreksi underpaid (naikkan ke median grup)"
STRAT_MERIT = "Merit merata (semua karyawan)"
STRAT_PERF = "Merit berbasis performa"
STRATEGIES = [STRAT_CORRECT, STRAT_MERIT, STRAT_PERF]


def now_str() -> str:
    """Waktu eksekusi (WIB) untuk riwayat dan laporan."""
    return datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S WIB")


# =============================================================================
# 1. LAPISAN DATA: muat, petakan kolom, bersihkan
# =============================================================================

def make_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Membuat data SINTETIS untuk demo bila dataset Kaggle belum diunduh.

    Beberapa baris sengaja disuntik anomali (terlalu tinggi / terlalu rendah)
    agar Modul 1 punya sesuatu untuk dideteksi.
    """
    rng = np.random.default_rng(seed)
    dept_base = {"Engineering": 45000, "Finance": 40000, "HR": 33000, "Marketing": 36000, "Sales": 34000}
    edu_bonus = {"High School": 0, "Bachelor": 4000, "Master": 9000, "PhD": 15000}
    cities = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai"]

    rows = []
    for i in range(n):
        dept = rng.choice(list(dept_base))
        edu = rng.choice(list(edu_bonus))
        exp = int(rng.integers(0, 26))
        age = int(min(60, 22 + exp + rng.integers(0, 8)))
        salary = dept_base[dept] + edu_bonus[edu] + exp * 1800 + rng.normal(0, 4000)
        rows.append(
            {
                "Employee_ID": f"E{i + 1:04d}",
                "Age": age,
                "Gender": rng.choice(["Male", "Female"]),
                "Department": dept,
                "City": rng.choice(cities),
                "Education_Level": edu,
                "Experience_Years": exp,
                "Performance_Score": int(rng.integers(1, 6)),
                "Monthly_Salary": float(round(max(salary, 8000), 0)),
            }
        )
    df = pd.DataFrame(rows)

    # Suntik anomali: 3 gaji terlalu tinggi, 3 terlalu rendah.
    picks = rng.choice(n, size=6, replace=False)
    for idx in picks[:3]:
        df.loc[idx, "Monthly_Salary"] = float(round(df.loc[idx, "Monthly_Salary"] * 2.6, 0))
    for idx in picks[3:]:
        df.loc[idx, "Monthly_Salary"] = float(round(df.loc[idx, "Monthly_Salary"] * 0.35, 0))
    return df


def read_csv_flexible(source) -> pd.DataFrame:
    """Membaca CSV dari path atau file-upload, mencoba beberapa encoding dan pemisah (, ; tab)."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = source.getvalue()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise ValueError("Tidak dapat membaca file CSV (encoding tidak dikenali).")


def _norm(name: object) -> str:
    """Normalisasi nama kolom: huruf kecil, hanya huruf/angka."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_number(x: object) -> float:
    """Mengubah nilai (termasuk teks seperti 'Rp 5.000.000' atau '1,250.50') menjadi float; gagal -> NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = re.sub(r"[^0-9,.\-]", "", str(x))
    if s in ("", "-", ".", ","):
        return np.nan
    if s.count(",") > 1:
        s = s.replace(",", "")
    if s.count(".") > 1:
        s = s.replace(".", "")
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_series(s: pd.Series) -> pd.Series:
    """Konversi satu kolom ke numerik secara toleran."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return s.map(parse_number).astype(float)


def detect_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Menebak kolom mana yang berperan sebagai gaji, departemen, dst.

    Urutan: (1) cocok persis setelah normalisasi, (2) cocok sebagian untuk alias >= 5 karakter.
    Satu kolom CSV tidak dipakai untuk dua peran sekaligus.
    """
    norm_cols: dict[str, str] = {}
    for c in df.columns:
        norm_cols.setdefault(_norm(c), c)

    mapping: dict[str, Optional[str]] = {}
    used: set[str] = set()
    for field, candidates in COLUMN_CANDIDATES.items():
        found: Optional[str] = None
        for cand in candidates:
            n = _norm(cand)
            if n in norm_cols and norm_cols[n] not in used:
                found = norm_cols[n]
                break
        if found is None:
            for cand in candidates:
                n = _norm(cand)
                if len(n) < 5:
                    continue
                found = next((orig for nc, orig in norm_cols.items() if n in nc and orig not in used), None)
                if found:
                    break
        mapping[field] = found
        if found:
            used.add(found)
    return mapping


def prepare_dataframe(raw: pd.DataFrame, mapping: dict[str, Optional[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Membentuk DataFrame standar (kolom: salary, department, ...) dan daftar catatan kualitas data."""
    notes: list[str] = []
    out = pd.DataFrame()
    for field, col in mapping.items():
        if col and col in raw.columns:
            out[field] = raw[col]

    # Gaji wajib numerik.
    out["salary"] = parse_series(out["salary"])
    missing_salary = int(out["salary"].isna().sum())
    if missing_salary:
        notes.append(f"{missing_salary} baris dibuang karena gaji kosong / bukan angka.")
    out = out.dropna(subset=["salary"])

    for numeric_field in ("experience", "age", "performance"):
        if numeric_field in out.columns:
            out[numeric_field] = parse_series(out[numeric_field])

    out = out.reset_index(drop=True)

    # Kolom opsional yang kosong total dibuang agar tidak merusak analisis.
    for col in list(out.columns):
        if col not in ("salary", "employee_id") and out[col].isna().all():
            out = out.drop(columns=[col])
            notes.append(f"Kolom opsional '{FIELD_LABELS.get(col, col)}' kosong seluruhnya sehingga diabaikan.")

    if "employee_id" not in out.columns:
        out["employee_id"] = [f"E{i + 1:04d}" for i in range(len(out))]
        notes.append("Kolom ID karyawan tidak ada; ID dibuat otomatis.")
    else:
        fallback = pd.Series([f"E{i + 1:04d}" for i in range(len(out))], index=out.index)
        out["employee_id"] = out["employee_id"].where(out["employee_id"].notna(), fallback)
    out["employee_id"] = out["employee_id"].astype(str)

    # Sanity check dasar kualitas data.
    dupes = int(raw.duplicated().sum())
    if dupes:
        notes.append(f"Ditemukan {dupes} baris duplikat penuh pada data mentah.")
    if (out["salary"] <= 0).any():
        notes.append(f"{int((out['salary'] <= 0).sum())} baris bergaji nol/negatif.")
    if "experience" in out.columns and "age" in out.columns:
        impossible = out[(out["experience"] > (out["age"] - 15)) & out["age"].notna() & out["experience"].notna()]
        if len(impossible):
            notes.append(f"{len(impossible)} baris dengan pengalaman kerja tidak masuk akal terhadap usia.")
    return out, notes


# =============================================================================
# 1b. DATA READINESS & QUALITY CHECK
# =============================================================================

def column_status_table(raw: pd.DataFrame, mapping: dict[str, Optional[str]]) -> pd.DataFrame:
    """Tabel status ketersediaan tiap kolom penting (ditemukan? kosong? numerik valid?)."""
    rows = []
    for field, label in FIELD_LABELS.items():
        category = "Wajib" if field == "salary" else ("Penting" if field in ESSENTIAL_FIELDS else "Opsional")
        col = mapping.get(field)
        if not col or col not in raw.columns:
            if field == "employee_id":
                status = "⚠️ Akan dibuat otomatis"
            elif category == "Opsional":
                status = "➖ Tidak ada (opsional)"
            else:
                status = "❌ Tidak ditemukan"
            rows.append({"Kolom baku": field, "Label": label, "Kategori": category, "Kolom di CSV": "-",
                         "Tipe": "-", "Kosong (%)": np.nan, "Numerik valid (%)": np.nan, "Status": status})
            continue

        s = raw[col]
        miss = float(s.isna().mean() * 100)
        valid_pct = np.nan
        if field in NUMERIC_FIELDS and s.notna().any():
            valid_pct = float(parse_series(s).notna().sum() / s.notna().sum() * 100)

        if miss >= 99.99:
            status = "❌ Kolom kosong total"
        elif not np.isnan(valid_pct) and valid_pct < 90:
            status = f"⚠️ {100 - valid_pct:.0f}% nilai bukan angka"
        elif miss > 0:
            status = "⚠️ Sebagian kosong"
        else:
            status = "✅ Siap"
        rows.append({"Kolom baku": field, "Label": label, "Kategori": category, "Kolom di CSV": col,
                     "Tipe": str(s.dtype), "Kosong (%)": round(miss, 1),
                     "Numerik valid (%)": round(valid_pct, 1) if not np.isnan(valid_pct) else np.nan,
                     "Status": status})
    return pd.DataFrame(rows)


def readiness_score(status_df: pd.DataFrame) -> float:
    """Skor kesiapan 0-100: gaji bernilai 40 poin, delapan kolom penting lain @7,5 poin, dikali kelengkapan."""
    score = 0.0
    for _, r in status_df.iterrows():
        if r["Kolom baku"] not in ESSENTIAL_FIELDS:
            continue
        weight = 40.0 if r["Kolom baku"] == "salary" else 7.5
        if r["Kolom di CSV"] == "-" or str(r["Status"]).startswith("❌"):
            continue
        completeness = 1.0 - (r["Kosong (%)"] or 0.0) / 100.0
        valid = r["Numerik valid (%)"]
        if not pd.isna(valid):
            completeness *= valid / 100.0
        score += weight * completeness
    return round(score, 1)


def render_data_readiness(
    raw: pd.DataFrame,
    mapping: dict[str, Optional[str]],
    df: Optional[pd.DataFrame] = None,
    notes: Optional[list[str]] = None,
) -> None:
    """Menampilkan ringkasan kesiapan & kualitas data."""
    st.markdown("#### ✅ Data Readiness & Quality Check")
    status_df = column_status_table(raw, mapping)
    score = readiness_score(status_df)

    num_cols = list(raw.select_dtypes(include="number").columns)
    cat_cols = [c for c in raw.columns if c not in num_cols]
    total_cells = max(raw.shape[0] * raw.shape[1], 1)
    total_missing = int(raw.isna().sum().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baris × Kolom", f"{raw.shape[0]:,} × {raw.shape[1]}")
    c2.metric("Sel kosong", f"{total_missing:,}", delta=f"{total_missing / total_cells * 100:.1f}% dari semua sel", delta_color="off")
    c3.metric("Numerik / Kategorikal", f"{len(num_cols)} / {len(cat_cols)}")
    c4.metric("Baris duplikat", f"{int(raw.duplicated().sum()):,}")
    c5.metric("Skor kesiapan", f"{score:.0f} / 100")

    salary_row = status_df[status_df["Kolom baku"] == "salary"].iloc[0]
    if str(salary_row["Status"]).startswith("❌"):
        st.error("Kolom gaji belum tersedia / kosong. Pilih kolom gaji di sidebar > Pemetaan kolom.")
    elif score >= 80:
        st.success("Data siap dipakai untuk semua modul.")
    elif score >= 50:
        st.warning("Data dapat dipakai, tetapi sebagian fitur akan berjalan dengan informasi terbatas (lihat catatan di bawah).")
    else:
        st.warning("Kesiapan data rendah. Periksa pemetaan kolom dan kualitas isi kolom.")

    if df is not None:
        st.caption(f"{len(df):,} baris valid dari {len(raw):,} baris mentah setelah pembersihan.")

    st.markdown("**Status ketersediaan kolom penting**")
    st.dataframe(status_df.drop(columns=["Label"]), hide_index=True)

    # Dampak kolom yang hilang terhadap fitur.
    have = {f for f in FIELD_LABELS if mapping.get(f) and mapping[f] in raw.columns
            and not raw[mapping[f]].isna().all()}
    impacts = []
    if "department" not in have and "position" not in have:
        impacts.append("Tanpa departemen/posisi, benchmark & audit grup memakai seluruh perusahaan sebagai satu grup.")
    if "experience" not in have:
        impacts.append("Tanpa pengalaman, lapis audit 'tren pengalaman' dan grafik gaji-vs-pengalaman tidak aktif.")
    if "performance" not in have:
        impacts.append("Tanpa skor performa, strategi 'Merit berbasis performa' berjalan sebagai merit merata.")
    if "gender" not in have:
        impacts.append("Tanpa gender, tabel selisih gaji antar-gender tidak ditampilkan.")
    if "age" not in have:
        impacts.append("Tanpa usia, pemeriksaan kewajaran pengalaman terhadap usia dilewati.")
    if "employee_id" not in have:
        impacts.append("ID karyawan dibuat otomatis (E0001, E0002, ...).")
    for msg in impacts:
        st.caption(f"ℹ️ {msg}")
    for n in notes or []:
        st.caption(f"🧹 {n}")

    with st.expander("Detail nilai kosong & tipe data per kolom"):
        detail = pd.DataFrame(
            {
                "Tipe": raw.dtypes.astype(str),
                "Jenis": ["numerik" if c in num_cols else "kategorikal" for c in raw.columns],
                "Kosong": raw.isna().sum(),
                "Kosong (%)": (raw.isna().mean() * 100).round(1),
                "Nilai unik": raw.nunique(),
            }
        ).sort_values("Kosong", ascending=False)
        st.dataframe(detail)


# =============================================================================
# 2. MODUL 1 (logika): AUDIT PAYROLL & DETEKSI ANOMALI
# =============================================================================

def run_audit(df: pd.DataFrame, group_col: Optional[str], mz_thresh: float, resid_thresh: float) -> pd.DataFrame:
    """Menjalankan tiga lapis deteksi anomali dan mengembalikan DataFrame beranotasi.

    Lapis 1 - Outlier global  : aturan IQR (Tukey) pada seluruh gaji.
    Lapis 2 - Outlier grup    : modified z-score (median/MAD) di dalam tiap posisi/departemen.
    Lapis 3 - Tren pengalaman : residu regresi linear gaji ~ pengalaman.
    Kolom hasil: `is_anomaly`, `anomaly_reasons`, `exp_resid_z`.
    """
    d = df.copy()
    reasons: dict[int, list[str]] = {i: [] for i in d.index}
    sal = d["salary"]

    for i in d.index[sal <= 0]:
        reasons[i].append("Gaji nol/negatif")

    valid = sal[sal > 0]

    # Lapis 1: IQR global.
    if len(valid) >= 8:
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for i in valid.index[(valid < low) | (valid > high)]:
            reasons[i].append(f"Outlier global (IQR; batas wajar {low:,.0f}-{high:,.0f})")

    # Lapis 2: modified z-score per grup.
    if group_col and group_col in d.columns:
        for name, sub in d[d["salary"] > 0].groupby(group_col):
            if len(sub) < 5:
                continue
            med = sub["salary"].median()
            mad = (sub["salary"] - med).abs().median()
            if mad == 0:
                continue
            mz = 0.6745 * (sub["salary"] - med) / mad
            for i in mz.index[mz.abs() > mz_thresh]:
                reasons[i].append(f"Outlier dalam grup '{name}' (modified z={mz[i]:+.1f})")

    # Lapis 3: residu terhadap tren pengalaman.
    d["exp_resid_z"] = np.nan
    if "experience" in d.columns:
        sub = d[(d["salary"] > 0) & d["experience"].notna()]
        if len(sub) >= 10 and sub["experience"].nunique() > 1:
            slope, intercept = np.polyfit(sub["experience"], sub["salary"], 1)
            resid = sub["salary"] - (slope * sub["experience"] + intercept)
            sd = resid.std()
            if sd > 0:
                z = resid / sd
                d.loc[z.index, "exp_resid_z"] = z
                for i in z.index[z.abs() > resid_thresh]:
                    arah = "di atas" if z[i] > 0 else "di bawah"
                    reasons[i].append(f"Gaji {arah} tren pengalaman (z={z[i]:+.1f})")

    d["anomaly_reasons"] = ["; ".join(reasons[i]) for i in d.index]
    d["is_anomaly"] = d["anomaly_reasons"] != ""
    return d


def compute_group_gaps(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Statistik gaji per grup + selisih median grup terhadap median perusahaan."""
    g = df.groupby(group_col)["salary"].agg(["count", "median", "mean", "min", "max"]).reset_index()
    overall = df["salary"].median()
    g["selisih_vs_median_perusahaan_%"] = (g["median"] / overall - 1) * 100
    return g.sort_values("median", ascending=False).reset_index(drop=True)


def compute_gender_gap(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Median gaji per gender dan selisihnya terhadap median tertinggi."""
    if "gender" not in df.columns:
        return None
    med = df.groupby("gender")["salary"].agg(["count", "median"]).reset_index()
    top = med["median"].max()
    med["selisih_vs_tertinggi_%"] = (med["median"] / top - 1) * 100
    return med


# =============================================================================
# 3. UTILITAS ANALITIK BERSAMA (kuartil, fakta karyawan, anggaran)
# =============================================================================

def fmt_money(value: float, currency: str) -> str:
    """Format angka gaji dengan label mata uang."""
    return f"{currency} {value:,.0f}"


def salary_quartile_info(df: pd.DataFrame, row: pd.Series, group_col: Optional[str]) -> dict:
    """Menentukan kuartil gaji karyawan di dalam grupnya (atau seluruh perusahaan bila grup < 4 orang)."""
    pool = df
    group_name = "seluruh perusahaan"
    if group_col and group_col in df.columns and pd.notna(row.get(group_col)):
        candidate = df[df[group_col] == row[group_col]]
        if len(candidate) >= 4:
            pool = candidate
            group_name = f"{row[group_col]}"

    q1, q2, q3 = pool["salary"].quantile([0.25, 0.5, 0.75])
    s = float(row["salary"])
    if s <= q1:
        label, target = "Q1 (25% terbawah)", q1
    elif s <= q2:
        label, target = "Q2 (di bawah median)", q2
    elif s <= q3:
        label, target = "Q3 (di atas median)", q3
    else:
        label, target = "Q4 (25% teratas)", float(pool["salary"].max())
    return {
        "label": label,
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3),
        "target": float(target),
        "gap": max(float(target) - s, 0.0),
        "group_name": group_name,
        "pool_size": int(len(pool)),
    }


def build_employee_facts(row: pd.Series, df: pd.DataFrame, group_col: Optional[str], currency: str, flag_reason: str) -> str:
    """Merangkum data seorang karyawan menjadi teks fakta untuk prompt agen."""
    lines = [
        f"- ID Karyawan: {row['employee_id']}",
        f"- Gaji bulanan saat ini: {fmt_money(row['salary'], currency)}",
    ]
    for key, label in (
        ("position", "Posisi"),
        ("department", "Departemen"),
        ("experience", "Pengalaman (tahun)"),
        ("education", "Pendidikan"),
        ("age", "Usia"),
        ("performance", "Skor performa"),
        ("city", "Kota"),
    ):
        if key in row.index and pd.notna(row[key]):
            lines.append(f"- {label}: {row[key]}")

    q = salary_quartile_info(df, row, group_col)
    lines.append(f"- Posisi gaji dalam {q['group_name']} (n={q['pool_size']}): {q['label']}")
    lines.append(
        f"- Median grup: {fmt_money(q['q2'], currency)}; Q1: {fmt_money(q['q1'], currency)}; Q3: {fmt_money(q['q3'], currency)}"
    )
    if flag_reason:
        lines.append(f"- Temuan audit payroll: {flag_reason}")
    else:
        lines.append("- Temuan audit payroll: tidak ada anomali")
    return "\n".join(lines)


def budget_state(df: pd.DataFrame, cap_pct: float) -> dict:
    """Menghitung pool anggaran kenaikan gaji (per bulan) dan sisanya."""
    total_payroll = float(df["salary"].sum())
    pool = total_payroll * cap_pct / 100.0
    committed = float(st.session_state.get("committed_raise", 0.0))
    return {
        "total_payroll": total_payroll,
        "pool": pool,
        "committed": committed,
        "remaining": max(pool - committed, 0.0),
    }


# =============================================================================
# 3b. WHAT-IF SIMULATOR (kebijakan & anggaran)
# =============================================================================

def simulate_scenario(df: pd.DataFrame, audit_df: pd.DataFrame, group_col: Optional[str], cfg: dict) -> dict:
    """Simulasi penyesuaian gaji berdasarkan skenario (pool anggaran, batas individu, strategi, merit).

    Langkah:
      1. Hitung kenaikan yang "diinginkan" per karyawan menurut strategi.
      2. Batasi per karyawan oleh `max_individual` (% gaji).
      3. Jika total melebihi pool anggaran, semua kenaikan diskalakan proporsional.
      4. Audit ulang pada gaji baru untuk melihat perubahan jumlah anomali.
    """
    d = df.copy()
    sal = d["salary"].astype(float)
    total_before = float(sal.sum())
    pool = total_before * cfg["cap_pct"] / 100.0

    if group_col and group_col in d.columns:
        gmed = d.groupby(group_col)["salary"].transform("median").fillna(sal.median())
    else:
        gmed = pd.Series(float(sal.median()), index=d.index)

    under = audit_df["is_anomaly"].reindex(d.index).fillna(False) & (sal < gmed) & (sal > 0)
    gap = (gmed - sal).clip(lower=0)
    cap_each = sal * cfg["max_individual"] / 100.0
    merit_pct = float(cfg["merit_pct"])
    strategy = cfg["strategy"]

    perf_missing = False
    if strategy == STRAT_CORRECT:
        want = gap.where(under, 0.0)
    elif strategy == STRAT_MERIT:
        want = sal * merit_pct / 100.0
    else:
        w: object = 1.0
        if "performance" in d.columns and d["performance"].notna().any():
            pmean = d["performance"].mean()
            if pd.notna(pmean) and pmean > 0:
                w = (d["performance"] / pmean).clip(0.5, 1.5).fillna(1.0)
            else:
                perf_missing = True
        else:
            perf_missing = True
        want = sal * merit_pct / 100.0 * w

    want = pd.Series(np.minimum(np.asarray(want, dtype=float), cap_each.to_numpy()), index=d.index).clip(lower=0)
    need = float(want.sum())
    scale = min(1.0, pool / need) if need > 0 else 1.0
    inc = want * scale

    d["gaji_baru"] = sal + inc
    d["kenaikan"] = inc
    d["kenaikan_pct"] = np.where(sal > 0, inc / sal * 100, 0.0)

    after_df = d.copy()
    after_df["salary"] = d["gaji_baru"]
    audit_after = run_audit(after_df, group_col, cfg["mz_thresh"], cfg["resid_thresh"])

    total_after = float(d["gaji_baru"].sum())
    recipients = inc > 0.5
    detail_cols = ["employee_id"] + ([group_col] if group_col and group_col in d.columns else [])
    detail = d[detail_cols + ["salary", "gaji_baru", "kenaikan", "kenaikan_pct"]].copy()
    detail["anomali_sebelum"] = audit_df["is_anomaly"].reindex(d.index).fillna(False).to_numpy()
    detail["anomali_sesudah"] = audit_after["is_anomaly"].to_numpy()

    group_table = None
    if group_col and group_col in d.columns:
        group_table = (
            detail.groupby(group_col)
            .agg(
                n=("employee_id", "count"),
                median_sebelum=("salary", "median"),
                median_sesudah=("gaji_baru", "median"),
                total_sebelum=("salary", "sum"),
                total_sesudah=("gaji_baru", "sum"),
            )
            .reset_index()
        )
        group_table["selisih_median_%"] = (group_table["median_sesudah"] / group_table["median_sebelum"] - 1) * 100

    return {
        "strategy": strategy,
        "merit_pct": merit_pct,
        "cap_pct": cfg["cap_pct"],
        "max_individual": cfg["max_individual"],
        "mz_thresh": cfg["mz_thresh"],
        "resid_thresh": cfg["resid_thresh"],
        "total_before": total_before,
        "total_after": total_after,
        "delta": total_after - total_before,
        "delta_pct": (total_after / total_before - 1) * 100 if total_before else 0.0,
        "pool": pool,
        "need": need,
        "scale": scale,
        "pool_used_pct": (total_after - total_before) / pool * 100 if pool > 0 else 0.0,
        "n_recipients": int(recipients.sum()),
        "avg_raise_pct": float(d.loc[recipients, "kenaikan_pct"].mean()) if recipients.any() else 0.0,
        "anom_before": int(audit_df["is_anomaly"].sum()),
        "anom_after": int(audit_after["is_anomaly"].sum()),
        "perf_missing": perf_missing,
        "detail": detail,
        "group_table": group_table,
    }


# =============================================================================
# 4. LAPISAN INFERENSI: GROQ CLIENT + ERROR HANDLING
# =============================================================================

class LLMError(Exception):
    """Galat ramah-pengguna yang aman ditampilkan di UI."""


def get_client(api_key: str):
    """Membuat klien Groq; melempar LLMError bila library/key bermasalah."""
    if Groq is None:
        raise LLMError("Library `groq` belum terpasang. Jalankan: pip install -r requirements.txt")
    if not api_key or not api_key.strip():
        raise LLMError("Groq API Key masih kosong. Masukkan key di sidebar (atau set GROQ_API_KEY di .env).")
    return Groq(api_key=api_key.strip())


def llm_chat(client, model: str, system: str, user: str, temperature: float = 0.4, max_tokens: int = 500) -> str:
    """Satu panggilan chat completion ke Groq dengan penanganan error yang rinci.

    Untuk model gpt-oss (reasoning), batas token dinaikkan dan reasoning_effort diset 'low'
    agar jawaban tidak terpotong oleh token reasoning.
    """
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "gpt-oss" in model.lower():
        kwargs["max_tokens"] = max_tokens + REASONING_HEADROOM
        kwargs["reasoning_effort"] = "low"
    try:
        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # Versi SDK lama tanpa parameter reasoning_effort.
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()
    except groq.AuthenticationError as exc:
        raise LLMError("API Key ditolak Groq (tidak valid atau sudah dicabut). Periksa kembali key Anda.") from exc
    except groq.RateLimitError as exc:
        raise LLMError("Batas laju (rate limit) Groq tercapai. Tunggu beberapa saat lalu coba lagi.") from exc
    except groq.NotFoundError as exc:
        raise LLMError(f"Model '{model}' tidak ditemukan. Pilih model lain di sidebar.") from exc
    except groq.BadRequestError as exc:
        raise LLMError(
            f"Permintaan ditolak Groq (kemungkinan model '{model}' sudah dihentikan atau input terlalu panjang): {exc}"
        ) from exc
    except groq.APIConnectionError as exc:
        raise LLMError("Tidak dapat terhubung ke Groq. Periksa koneksi internet Anda.") from exc
    except groq.APIStatusError as exc:
        raise LLMError(f"Groq mengembalikan error HTTP {exc.status_code}: {exc}") from exc
    except Exception as exc:  # jaring pengaman terakhir
        raise LLMError(f"Kesalahan tak terduga saat memanggil Groq: {exc}") from exc


def default_api_key() -> str:
    """Mengambil API key dari environment atau Streamlit secrets (untuk deployment)."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


# =============================================================================
# 5. MODUL 2 (logika): NEGOSIASI HIERARKIS MULTI-AGEN
# =============================================================================

AGENTS: dict[str, dict] = {
    "hr": {
        "name": "Agen Analis HR",
        "avatar": "🧑‍💼",
        "tag": "USULAN",
        "system": (
            "Anda adalah Analis HR senior. Tugas Anda mengajukan persentase kenaikan gaji yang adil "
            "berdasarkan DATA (performa, masa kerja, posisi kuartil gaji, temuan audit). "
            "Jangan mengarang data di luar yang diberikan. Gunakan Bahasa Indonesia, maksimal 110 kata. "
            "Akhiri dengan satu baris persis: USULAN: <angka>%"
        ),
    },
    "fin": {
        "name": "Agen Manajer Keuangan",
        "avatar": "💼",
        "tag": "KEPUTUSAN",
        "system": (
            "Anda adalah Manajer Keuangan yang ketat dan menjaga disiplin anggaran payroll. "
            "Anda TIDAK BOLEH menyetujui kenaikan di atas plafon yang diberikan. "
            "Evaluasi usulan terakhir, tolak/counter bila perlu, dan berikan alasan singkat. "
            "Gunakan Bahasa Indonesia, maksimal 110 kata. "
            "Akhiri dengan satu baris persis: KEPUTUSAN: <angka>%"
        ),
    },
    "emp": {
        "name": "Agen Karyawan",
        "avatar": "🙋",
        "tag": "PERMINTAAN",
        "system": (
            "Anda adalah karyawan yang berunding secara profesional dan sopan. Anda ingin gaji yang layak "
            "tetapi realistis terhadap kondisi perusahaan. Jika tawaran Manajer Keuangan wajar, "
            "terima dan nyatakan setuju. Gunakan Bahasa Indonesia, maksimal 110 kata. "
            "Akhiri dengan satu baris persis: PERMINTAAN: <angka>%"
        ),
    },
}


def extract_pct(text: str, tag: str) -> Optional[float]:
    """Mengambil angka persen dari baris bertag, misalnya 'USULAN: 7.5%'."""
    matches = re.findall(rf"{tag}\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*%", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return float(matches[-1].replace(",", "."))


def transcript_to_text(transcript: list[dict]) -> str:
    """Mengubah transkrip menjadi teks untuk konteks prompt."""
    if not transcript:
        return "(belum ada percakapan)"
    return "\n\n".join(f"[Ronde {t['round']} - {t['name']}]\n{t['text']}" for t in transcript)


def parse_mediator_json(text: str) -> Optional[dict]:
    """Mengurai JSON dari keluaran Notulis, toleran terhadap teks tambahan di sekitarnya."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def run_negotiation(
    client,
    model: str,
    temperature: float,
    facts: str,
    ceiling_pct: float,
    rounds: int,
    currency: str,
    salary: float,
) -> dict:
    """Menjalankan simulasi negosiasi dan menampilkan tiap giliran secara real-time.

    Hirarki keputusan:
      1. Analis HR mengusulkan.
      2. Manajer Keuangan menyaring dengan plafon anggaran.
      3. Karyawan merespons / menawar.
      4. Notulis merangkum konsensus; angka akhir DIPAKSA <= plafon (guardrail kode,
         bukan sekadar permintaan di prompt).
    """
    budget_text = (
        f"Plafon kenaikan untuk karyawan ini: maksimal {ceiling_pct:.2f}% "
        f"(setara {fmt_money(salary * ceiling_pct / 100, currency)} per bulan). "
        "Anggaran total perusahaan terbatas."
    )
    transcript: list[dict] = []
    consensus = False

    instructions = {
        "hr": lambda r: "Ajukan usulan awal kenaikan gaji."
        if r == 1
        else "Revisi usulan Anda dengan mempertimbangkan keberatan Keuangan dan permintaan Karyawan.",
        "fin": lambda r: "Evaluasi usulan Analis HR terakhir terhadap plafon anggaran. Jika melebihi plafon, tolak dan counter di bawah/di plafon.",
        "emp": lambda r: "Tanggapi keputusan terakhir Manajer Keuangan. Jika wajar, setujui; jika tidak, ajukan permintaan balik yang realistis.",
    }

    for r in range(1, rounds + 1):
        for key in ("hr", "fin", "emp"):
            agent = AGENTS[key]
            user_prompt = (
                f"DATA KASUS:\n{facts}\n\nBATAS ANGGARAN:\n{budget_text}\n\n"
                f"TRANSKRIP SEJAK AWAL:\n{transcript_to_text(transcript)}\n\n"
                f"GILIRAN ANDA (Ronde {r}/{rounds}): {instructions[key](r)}"
            )
            text = llm_chat(client, model, agent["system"], user_prompt, temperature=temperature, max_tokens=350)
            transcript.append({"round": r, "key": key, "name": agent["name"], "avatar": agent["avatar"], "text": text})

            # Tampilkan langsung (real-time) setelah tiap agen selesai bicara.
            with st.chat_message(agent["name"], avatar=agent["avatar"]):
                st.markdown(f"**{agent['name']}** - Ronde {r}")
                st.markdown(text)

        # Cek konsensus: permintaan karyawan <= keputusan keuangan (toleransi 0,25 poin).
        fin_dec = extract_pct(transcript[-2]["text"], "KEPUTUSAN")
        emp_req = extract_pct(transcript[-1]["text"], "PERMINTAAN")
        emp_text = transcript[-1]["text"].lower()
        if fin_dec is not None and (
            (emp_req is not None and emp_req <= fin_dec + 0.25) or (emp_req is None and "setuju" in emp_text)
        ):
            consensus = True
            break

    # Notulis merangkum konsensus dalam JSON.
    mediator_system = (
        "Anda adalah Notulis Konsensus yang netral. Keluarkan HANYA satu objek JSON valid, tanpa teks lain, "
        'dengan kunci: "status" ("consensus" atau "no_consensus"), "final_raise_pct" (angka), "rationale" (string singkat).'
    )
    mediator_user = (
        f"DATA KASUS:\n{facts}\n\nBATAS ANGGARAN:\n{budget_text}\n\n"
        f"TRANSKRIP:\n{transcript_to_text(transcript)}\n\n"
        f"Tentukan hasil akhir. final_raise_pct tidak boleh melebihi {ceiling_pct:.2f}."
    )
    mediator_text = llm_chat(client, model, mediator_system, mediator_user, temperature=0.1, max_tokens=250)
    parsed = parse_mediator_json(mediator_text)

    last_fin = next((extract_pct(t["text"], "KEPUTUSAN") for t in reversed(transcript) if t["key"] == "fin"), None)
    if parsed and isinstance(parsed.get("final_raise_pct"), (int, float)):
        final_pct = float(parsed["final_raise_pct"])
        status = str(parsed.get("status", "consensus" if consensus else "no_consensus"))
        rationale = str(parsed.get("rationale", ""))
    else:
        final_pct = float(last_fin if last_fin is not None else 0.0)
        status = "consensus" if consensus else "no_consensus"
        rationale = "Ringkasan otomatis gagal diurai; memakai keputusan terakhir Manajer Keuangan."

    # Guardrail kode: tidak boleh negatif dan tidak boleh melewati plafon.
    final_pct = max(0.0, min(final_pct, ceiling_pct))
    return {
        "transcript": transcript,
        "status": status,
        "final_pct": final_pct,
        "rationale": rationale,
        "ceiling_pct": ceiling_pct,
        "salary_before": salary,
        "salary_after": salary * (1 + final_pct / 100),
        "monthly_increase": salary * final_pct / 100,
        "applied": False,
    }


def render_transcript(transcript: list[dict]) -> None:
    """Menampilkan ulang transkrip tersimpan (setelah Streamlit rerun)."""
    for t in transcript:
        with st.chat_message(t["name"], avatar=t["avatar"]):
            st.markdown(f"**{t['name']}** - Ronde {t['round']}")
            st.markdown(t["text"])


# =============================================================================
# 6. MODUL 3 (logika): PENASIHAT KARIER
# =============================================================================

CAREER_SYSTEM = (
    "Anda adalah Penasihat Karier & Pengembangan Kompetensi yang praktis dan realistis. "
    "Gunakan HANYA data yang diberikan dan pengetahuan umum industri; jangan menjanjikan kenaikan gaji pasti. "
    "Tulis dalam Bahasa Indonesia, format Markdown ringkas dengan bagian: "
    "**Posisi Anda Saat Ini**, **3 Kompetensi Prioritas** (beserta alasan), "
    "**Roadmap 0-3 / 3-6 / 6-12 Bulan**, **Indikator Keberhasilan**, **Catatan Realistis**, "
    "dan tutup dengan **Poin Utama** berisi tepat 3 butir bullet singkat (satu kalimat per butir). "
    "Maksimal sekitar 380 kata."
)


def build_career_prompt(facts: str, q: dict, currency: str, negotiation: Optional[dict]) -> str:
    """Menyusun prompt penasihat karier dari fakta karyawan, kuartil, dan hasil negosiasi (bila ada)."""
    extra = ""
    if negotiation:
        extra = (
            f"\nHasil negosiasi kenaikan gaji: {negotiation['final_pct']:.2f}% "
            f"(gaji baru {fmt_money(negotiation['salary_after'], currency)}), status: {negotiation['status']}."
        )
    return (
        f"DATA KARYAWAN:\n{facts}\n\n"
        f"Level gaji berikutnya (batas kuartil atas): {fmt_money(q['target'], currency)}; "
        f"selisih dari gaji saat ini: {fmt_money(q['gap'], currency)}.{extra}\n\n"
        "Susun rekomendasi pengembangan kompetensi agar karyawan ini layak naik ke level gaji berikutnya."
    )


def extract_key_points(text: str) -> list[str]:
    """Mengambil butir 'Poin Utama' dari jawaban penasihat karier (fallback: butir bullet pertama)."""
    bullet = r"^\s*(?:[-*•]|\d+[.)])\s+"
    m = re.search(r"Poin Utama[^\n]*\n(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    block = m.group(1) if m else text
    points = [re.sub(bullet, "", ln).replace("**", "").strip() for ln in block.splitlines() if re.match(bullet, ln)]
    points = [p for p in points if p][:4]
    if not points:
        lines = [ln.strip().lstrip("#").replace("**", "").strip() for ln in text.splitlines() if ln.strip()]
        points = lines[:3]
    return points


# =============================================================================
# 7. RIWAYAT SESI
# =============================================================================

def init_session_state() -> None:
    """Menyiapkan state yang bertahan antar interaksi Streamlit."""
    st.session_state.setdefault("committed_raise", 0.0)  # total kenaikan (per bulan) yang sudah disetujui
    st.session_state.setdefault("negotiations", {})  # emp_id -> hasil negosiasi
    st.session_state.setdefault("careers", {})  # emp_id -> rekomendasi karier
    st.session_state.setdefault("audit_ai", None)  # ringkasan AI audit terakhir
    st.session_state.setdefault("exec_ai", "")  # narasi eksekutif AI terakhir
    st.session_state.setdefault("history", [])  # riwayat analisis/simulasi (maks MAX_HISTORY)
    st.session_state.setdefault("history_counter", 0)


def add_history(kind: str, model: str, summary: str, detail: dict) -> None:
    """Menambah entri riwayat (terbaru di atas) dan memangkas ke MAX_HISTORY."""
    st.session_state["history_counter"] += 1
    st.session_state["history"].insert(
        0,
        {
            "id": st.session_state["history_counter"],
            "waktu": now_str(),
            "jenis": kind,
            "model": model,
            "ringkasan": summary,
            "detail": detail,
        },
    )
    del st.session_state["history"][MAX_HISTORY:]


def render_history() -> None:
    """Tampilan riwayat: daftar, lihat kembali, hapus satu / semua."""
    st.markdown("### 🕘 Riwayat Sesi")
    hist = st.session_state["history"]
    if not hist:
        st.caption("Belum ada riwayat. Jalankan audit, simpan simulasi, negosiasi, atau penasihat karier.")
        return
    st.caption(f"Menyimpan maksimal {MAX_HISTORY} entri terbaru; hilang saat sesi browser berakhir.")
    st.dataframe(
        pd.DataFrame(
            [{"#": h["id"], "Waktu": h["waktu"], "Jenis": h["jenis"], "Model": h["model"], "Ringkasan": h["ringkasan"]} for h in hist]
        ),
        hide_index=True,
    )
    labels = {h["id"]: f"#{h['id']} | {h['waktu']} | {h['jenis']}" for h in hist}
    sel = st.selectbox("Lihat kembali entri", list(labels), format_func=lambda k: labels[k], key="history_select")
    entry = next((h for h in hist if h["id"] == sel), None)
    if entry:
        with st.expander("Detail entri", expanded=True):
            st.write(f"**{entry['jenis']}** | {entry['waktu']} | model: `{entry['model']}`")
            st.write(entry["ringkasan"])
            st.json(entry["detail"], expanded=False)
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Hapus entri terpilih"):
        st.session_state["history"] = [h for h in hist if h["id"] != sel]
        st.rerun()
    if c2.button("🧹 Hapus semua riwayat"):
        st.session_state["history"] = []
        st.rerun()


# =============================================================================
# 8. ANTARMUKA STREAMLIT
# =============================================================================

def sidebar_config() -> dict:
    """Membangun sidebar konfigurasi dan mengembalikan seluruh pengaturan sebagai dict."""
    sb = st.sidebar
    sb.header("⚙️ Konfigurasi")

    # --- Groq ---
    sb.subheader("Groq API")
    api_key = sb.text_input(
        "Groq API Key",
        value=default_api_key(),
        type="password",
        help="Key hanya disimpan di memori sesi browser Anda. Dapatkan di console.groq.com.",
    )
    model_choice = sb.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    if model_choice == "(ketik manual)":
        model = sb.text_input("Nama model", value=DEFAULT_MODEL).strip() or DEFAULT_MODEL
    else:
        model = model_choice
    temperature = sb.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if sb.button("🔌 Uji koneksi API"):
        try:
            client = get_client(api_key)
            reply = llm_chat(client, model, "Balas singkat.", "Jawab dengan kata: OK", temperature=0.0, max_tokens=20)
            sb.success(f"Koneksi berhasil. Balasan model: {reply or '(kosong, tetapi koneksi OK)'}")
        except LLMError as exc:
            sb.error(str(exc))

    # --- Data ---
    sb.subheader("Sumber Data")
    source = sb.radio("Pilih sumber", ["Upload CSV", "File lokal data/", "Data contoh (sintetis)"], index=0)
    uploaded = sb.file_uploader("CSV gaji karyawan", type=["csv"]) if source == "Upload CSV" else None
    currency = sb.text_input("Label mata uang", value="INR")

    # --- Anggaran ---
    sb.subheader("Anggaran & Simulasi")
    cap_pct = sb.slider("Plafon pool kenaikan (% dari total payroll bulanan)", 0.5, 10.0, 3.0, 0.5)
    max_individual = sb.slider("Batas kenaikan individu maksimum (%)", 1.0, 30.0, 12.0, 0.5)
    rounds = sb.slider("Jumlah ronde negosiasi", 1, 5, 3)
    if sb.button("♻️ Reset anggaran terpakai"):
        st.session_state["committed_raise"] = 0.0
        st.session_state["negotiations"] = {}
        sb.success("Anggaran terpakai direset.")

    # --- Audit ---
    sb.subheader("Ambang Audit")
    mz_thresh = sb.slider("Modified z-score (outlier grup)", 2.0, 6.0, 3.5, 0.1)
    resid_thresh = sb.slider("Residu tren pengalaman (std)", 1.5, 4.0, 2.5, 0.1)

    # --- What-If ---
    sb.subheader("🎛️ Skenario What-If")
    scenario_active = sb.checkbox(
        "Aktifkan skenario",
        value=False,
        help="Jika aktif, plafon pool, batas individu, dan sensitivitas anomali di bawah MENGGANTIKAN pengaturan di atas "
        "untuk Audit, Simulator, dan Negosiasi.",
    )
    sc_cap = sb.slider("Skenario: plafon pool kenaikan (%)", 0.0, 15.0, 3.0, 0.5)
    sc_max = sb.slider("Skenario: kenaikan individu maksimum (%)", 0.0, 40.0, 12.0, 0.5)
    sc_sens = sb.slider(
        "Skenario: pengali ambang anomali",
        0.5,
        1.5,
        1.0,
        0.05,
        help="Mengalikan ambang audit. <1 = lebih sensitif (lebih banyak anomali); >1 = lebih longgar.",
    )
    strategy = sb.selectbox("Strategi alokasi (simulator)", STRATEGIES)
    merit_pct = sb.slider("Merit (%) untuk strategi merit", 0.0, 15.0, 3.0, 0.5)
    sb.caption("Strategi & merit hanya dipakai Simulator What-If di Tab 1 (selalu berlaku).")

    eff_cap, eff_max, eff_mz, eff_resid = cap_pct, max_individual, mz_thresh, resid_thresh
    if scenario_active:
        eff_cap, eff_max = sc_cap, sc_max
        eff_mz = float(np.clip(mz_thresh * sc_sens, 1.0, 10.0))
        eff_resid = float(np.clip(resid_thresh * sc_sens, 1.0, 6.0))

    return {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "source": source,
        "uploaded": uploaded,
        "currency": currency,
        "cap_pct": eff_cap,
        "max_individual": eff_max,
        "rounds": rounds,
        "mz_thresh": eff_mz,
        "resid_thresh": eff_resid,
        "scenario_active": scenario_active,
        "base_cap_pct": cap_pct,
        "base_max_individual": max_individual,
        "base_mz_thresh": mz_thresh,
        "base_resid_thresh": resid_thresh,
        "strategy": strategy,
        "merit_pct": merit_pct,
    }


def load_raw_data(cfg: dict) -> Optional[pd.DataFrame]:
    """Memuat data mentah sesuai sumber yang dipilih; mengembalikan None bila belum tersedia."""
    try:
        if cfg["source"] == "Upload CSV":
            if cfg["uploaded"] is None:
                return None
            return read_csv_flexible(cfg["uploaded"])
        if cfg["source"] == "File lokal data/":
            if not DEFAULT_DATA_PATH.exists():
                st.sidebar.warning(f"File {DEFAULT_DATA_PATH} tidak ditemukan.")
                return None
            return read_csv_flexible(DEFAULT_DATA_PATH)
        return make_sample_data()
    except Exception as exc:
        st.sidebar.error(f"Gagal membaca data: {exc}")
        return None


def column_mapping_ui(raw: pd.DataFrame) -> dict[str, Optional[str]]:
    """Expander pemetaan kolom; default dari deteksi otomatis, dapat dikoreksi pengguna."""
    detected = detect_columns(raw)
    options = [NONE_OPTION] + list(raw.columns)
    sig = abs(hash(tuple(map(str, raw.columns)))) % 10**8  # kunci widget berubah bila struktur CSV berubah
    mapping: dict[str, Optional[str]] = {}
    with st.sidebar.expander("🧭 Pemetaan kolom", expanded=detected["salary"] is None):
        for field, label in FIELD_LABELS.items():
            default = detected[field] if detected[field] in options else NONE_OPTION
            choice = st.selectbox(label, options, index=options.index(default), key=f"map_{sig}_{field}")
            mapping[field] = None if choice == NONE_OPTION else choice
    return mapping


def tab_data(raw: pd.DataFrame, mapping: dict, df: pd.DataFrame, notes: list[str]) -> None:
    """Tab Data: readiness check + pratinjau."""
    st.subheader("Data")
    render_data_readiness(raw, mapping, df, notes)
    st.markdown("#### Pratinjau data (setelah pembersihan)")
    st.dataframe(df.head(200), hide_index=True)


def render_whatif(df: pd.DataFrame, audit_df: pd.DataFrame, group_col: Optional[str], cfg: dict, whatif: dict) -> None:
    """Panel What-If: perbandingan sebelum vs sesudah simulasi."""
    cur = cfg["currency"]
    st.markdown("#### 🎛️ Simulator What-If: Kebijakan & Anggaran")
    if cfg["scenario_active"]:
        st.info(
            f"Skenario AKTIF: pool {cfg['cap_pct']:.1f}% | batas individu {cfg['max_individual']:.1f}% | "
            f"ambang z-grup {cfg['mz_thresh']:.2f} | ambang tren {cfg['resid_thresh']:.2f}. "
            "Ubah di sidebar > Skenario What-If."
        )
    else:
        st.caption("Skenario tidak aktif: simulator memakai pengaturan baku sidebar. Aktifkan skenario untuk menguji angka lain.")
    st.caption(f"Strategi: **{whatif['strategy']}**" + (f" | merit {whatif['merit_pct']:.1f}%" if whatif["strategy"] != STRAT_CORRECT else ""))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total payroll awal / bulan", fmt_money(whatif["total_before"], cur))
    k2.metric(
        "Proyeksi setelah penyesuaian",
        fmt_money(whatif["total_after"], cur),
        delta=f"+{fmt_money(whatif['delta'], cur)} ({whatif['delta_pct']:.2f}%)",
        delta_color="off",
    )
    k3.metric("Pool anggaran terpakai", f"{whatif['pool_used_pct']:.1f}%", delta=f"pool {fmt_money(whatif['pool'], cur)}", delta_color="off")
    k4.metric("Anomali sebelum → sesudah", f"{whatif['anom_before']} → {whatif['anom_after']}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Karyawan menerima kenaikan", f"{whatif['n_recipients']:,}")
    r2.metric("Rata-rata kenaikan penerima", f"{whatif['avg_raise_pct']:.2f}%")
    if cfg["scenario_active"]:
        base_audit = run_audit(df, group_col, cfg["base_mz_thresh"], cfg["base_resid_thresh"])
        r3.metric(
            "Anomali: baseline → skenario",
            f"{int(base_audit['is_anomaly'].sum())} → {int(audit_df['is_anomaly'].sum())}",
            help="Dampak perubahan sensitivitas pada jumlah anomali (sebelum penyesuaian gaji).",
        )
    else:
        r3.metric("Faktor skala anggaran", f"{whatif['scale'] * 100:.0f}%")

    if whatif["scale"] < 1.0:
        st.warning(
            f"Kebutuhan kenaikan ({fmt_money(whatif['need'], cur)}) melebihi pool ({fmt_money(whatif['pool'], cur)}); "
            f"semua kenaikan diskalakan ke {whatif['scale'] * 100:.0f}%."
        )
    if whatif["perf_missing"]:
        st.info("Skor performa tidak tersedia/tidak valid; merit berbasis performa dijalankan sebagai merit merata.")
    if whatif["need"] == 0:
        st.info("Tidak ada karyawan yang memenuhi kriteria strategi ini (mis. tidak ada anomali underpaid).")

    if whatif["group_table"] is not None:
        gt = whatif["group_table"]
        st.markdown(f"**Median gaji per {FIELD_LABELS.get(group_col, group_col)}: sebelum vs sesudah**")
        st.dataframe(gt.round(1), hide_index=True)
        st.bar_chart(gt.set_index(group_col)[["median_sebelum", "median_sesudah"]])

    if st.button("💾 Simpan hasil simulasi ini ke riwayat"):
        add_history(
            "Simulasi What-If",
            "(tanpa AI)",
            f"{whatif['strategy']}: +{fmt_money(whatif['delta'], cur)} ({whatif['delta_pct']:.2f}%), "
            f"anomali {whatif['anom_before']}→{whatif['anom_after']}.",
            {
                "strategi": whatif["strategy"],
                "pool_%": whatif["cap_pct"],
                "batas_individu_%": whatif["max_individual"],
                "merit_%": whatif["merit_pct"],
                "ambang_mz": round(whatif["mz_thresh"], 2),
                "ambang_tren": round(whatif["resid_thresh"], 2),
                "payroll_awal": round(whatif["total_before"]),
                "payroll_proyeksi": round(whatif["total_after"]),
                "penerima": whatif["n_recipients"],
                "anomali_sebelum": whatif["anom_before"],
                "anomali_sesudah": whatif["anom_after"],
            },
        )
        st.success("Tersimpan ke riwayat sesi (lihat Tab 4).")


def tab_audit(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    notes: list[str],
    group_col: Optional[str],
    cfg: dict,
    whatif: dict,
) -> None:
    """Tab Modul 1: laporan audit payroll."""
    cur = cfg["currency"]
    st.subheader("Modul 1 - Payroll Audit & Anomaly Detection")

    n_anom = int(audit_df["is_anomaly"].sum())
    gaps = compute_group_gaps(df, group_col) if group_col else None
    max_gap = float(gaps["selisih_vs_median_perusahaan_%"].abs().max()) if gaps is not None and len(gaps) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah karyawan", f"{len(df):,}")
    c2.metric("Total payroll / bulan", fmt_money(df["salary"].sum(), cur))
    c3.metric("Median gaji", fmt_money(df["salary"].median(), cur))
    c4.metric("Anomali terdeteksi", f"{n_anom}", delta=f"{n_anom / max(len(df), 1) * 100:.1f}% dari data", delta_color="off")

    with st.expander("Sanity check kualitas data", expanded=bool(notes)):
        if notes:
            for n in notes:
                st.warning(n)
        else:
            st.success("Tidak ditemukan masalah kualitas data dasar.")

    st.markdown("#### Daftar anomali")
    anomalies = audit_df[audit_df["is_anomaly"]].copy()
    if anomalies.empty:
        st.info("Tidak ada anomali pada ambang saat ini. Coba turunkan ambang di sidebar.")
    else:
        show_cols = [c for c in ("employee_id", "position", "department", "experience", "salary", "anomaly_reasons") if c in anomalies.columns]
        st.dataframe(anomalies[show_cols].sort_values("salary", ascending=False), hide_index=True)

    if gaps is not None:
        st.markdown(f"#### Ketimpangan antar-{group_col}")
        st.dataframe(gaps.round(1), hide_index=True)
        st.bar_chart(gaps.set_index(group_col)["median"])
        st.caption("Grafik: median gaji per grup.")

    gender_gap = compute_gender_gap(df)
    if gender_gap is not None:
        st.markdown("#### Selisih gaji antar-gender (median)")
        st.dataframe(gender_gap.round(1), hide_index=True)
        st.caption("Perbedaan median bersifat deskriptif dan belum mengontrol pengalaman, pendidikan, atau posisi.")

    if "experience" in df.columns:
        st.markdown("#### Gaji vs pengalaman")
        st.scatter_chart(df, x="experience", y="salary")

    st.divider()
    render_whatif(df, audit_df, group_col, cfg, whatif)

    st.divider()
    st.markdown("#### Ringkasan oleh Agen Auditor (opsional)")
    if st.button("Minta ringkasan AI"):
        try:
            client = get_client(cfg["api_key"])
            top_reasons = anomalies["anomaly_reasons"].head(15).to_list()
            gap_text = gaps.round(1).to_string(index=False) if gaps is not None else "(tidak ada kolom grup)"
            prompt = (
                f"Total karyawan: {len(df)}; anomali: {n_anom}; median gaji: {df['salary'].median():,.0f} {cur}.\n"
                f"Statistik grup:\n{gap_text}\n\nContoh temuan anomali:\n- " + "\n- ".join(top_reasons or ["(tidak ada)"])
            )
            with st.spinner("Agen Auditor menulis ringkasan..."):
                answer = llm_chat(
                    client,
                    cfg["model"],
                    "Anda auditor payroll. Tulis ringkasan temuan dan 3 tindakan prioritas dalam Bahasa Indonesia, maksimal 180 kata, "
                    "hanya berdasarkan data yang diberikan.",
                    prompt,
                    temperature=cfg["temperature"],
                    max_tokens=500,
                )
            st.session_state["audit_ai"] = {"text": answer, "time": now_str(), "model": cfg["model"]}
            add_history(
                "Audit Payroll (ringkasan AI)",
                cfg["model"],
                f"{n_anom} anomali dari {len(df)} karyawan. " + answer[:120].replace("\n", " ") + "...",
                {"karyawan": len(df), "anomali": n_anom, "ambang_mz": round(cfg["mz_thresh"], 2),
                 "ambang_tren": round(cfg["resid_thresh"], 2), "ringkasan_ai": answer},
            )
            st.markdown(answer)
        except LLMError as exc:
            st.error(str(exc))
    elif st.session_state["audit_ai"]:
        saved = st.session_state["audit_ai"]
        st.caption(f"Ringkasan tersimpan | {saved['time']} | model {saved['model']}")
        st.markdown(saved["text"])


def tab_negotiation(df: pd.DataFrame, audit_df: pd.DataFrame, idx: int, group_col: Optional[str], cfg: dict) -> None:
    """Tab Modul 2: simulasi negosiasi multi-agen."""
    cur = cfg["currency"]
    st.subheader("Modul 2 - Hierarchical Decision-Making & Negotiation")
    if cfg["scenario_active"]:
        st.info(
            f"Skenario What-If aktif: plafon pool {cfg['cap_pct']:.1f}% dan batas individu {cfg['max_individual']:.1f}% "
            "dipakai pada negosiasi ini."
        )

    row = df.loc[idx]
    emp_id = str(row["employee_id"])
    flag_reason = audit_df.loc[idx, "anomaly_reasons"]
    facts = build_employee_facts(row, df, group_col, cur, flag_reason)

    b = budget_state(df, cfg["cap_pct"])
    ceiling_pct = min(cfg["max_individual"], (b["remaining"] / row["salary"] * 100) if row["salary"] > 0 else 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pool anggaran / bulan", fmt_money(b["pool"], cur))
    c2.metric("Sudah terpakai", fmt_money(b["committed"], cur))
    c3.metric("Sisa anggaran", fmt_money(b["remaining"], cur))
    c4.metric("Plafon untuk karyawan ini", f"{ceiling_pct:.2f}%")

    with st.expander("Fakta kasus yang dilihat para agen", expanded=False):
        st.text(facts)

    if ceiling_pct <= 0:
        st.error("Sisa anggaran habis. Tambah plafon di sidebar atau reset anggaran terpakai.")
        return

    run = st.button("▶️ Jalankan simulasi negosiasi", type="primary")
    if run:
        try:
            client = get_client(cfg["api_key"])
            st.session_state["negotiations"].pop(emp_id, None)
            st.markdown("#### Transkrip negosiasi (real-time)")
            result = run_negotiation(
                client, cfg["model"], cfg["temperature"], facts, ceiling_pct, cfg["rounds"], cur, float(row["salary"])
            )
            result["time"] = now_str()
            result["model"] = cfg["model"]
            result["emp_id"] = emp_id
            st.session_state["negotiations"][emp_id] = result
            add_history(
                "Negosiasi Multi-Agen",
                cfg["model"],
                f"{emp_id}: {'konsensus' if result['status'] == 'consensus' else 'tanpa konsensus'}, "
                f"kenaikan {result['final_pct']:.2f}% (plafon {ceiling_pct:.2f}%).",
                {
                    "karyawan": emp_id,
                    "status": result["status"],
                    "kenaikan_%": round(result["final_pct"], 2),
                    "plafon_%": round(ceiling_pct, 2),
                    "gaji_sebelum": round(result["salary_before"]),
                    "gaji_sesudah": round(result["salary_after"]),
                    "alasan": result["rationale"],
                    "transkrip": [{"ronde": t["round"], "agen": t["name"], "teks": t["text"]} for t in result["transcript"]],
                },
            )
        except LLMError as exc:
            st.error(str(exc))
    elif emp_id in st.session_state["negotiations"]:
        st.markdown("#### Transkrip negosiasi (tersimpan)")
        render_transcript(st.session_state["negotiations"][emp_id]["transcript"])

    result = st.session_state["negotiations"].get(emp_id)
    if result:
        st.markdown("#### Keputusan akhir")
        r1, r2, r3 = st.columns(3)
        r1.metric("Kenaikan disepakati", f"{result['final_pct']:.2f}%")
        r2.metric("Gaji baru / bulan", fmt_money(result["salary_after"], cur), delta=fmt_money(result["monthly_increase"], cur))
        r3.metric("Status", "Konsensus" if result["status"] == "consensus" else "Tanpa konsensus")
        if result["rationale"]:
            st.info(result["rationale"])
        if result["applied"]:
            st.success("Hasil ini sudah dicatat ke anggaran terpakai.")
        elif st.button("✅ Terapkan ke anggaran"):
            st.session_state["committed_raise"] += result["monthly_increase"]
            result["applied"] = True
            st.rerun()


def tab_career(df: pd.DataFrame, audit_df: pd.DataFrame, idx: int, group_col: Optional[str], cfg: dict) -> None:
    """Tab Modul 3: penasihat karier."""
    cur = cfg["currency"]
    st.subheader("Modul 3 - Career Growth & Recommendation Advisor")

    row = df.loc[idx]
    emp_id = str(row["employee_id"])
    q = salary_quartile_info(df, row, group_col)
    facts = build_employee_facts(row, df, group_col, cur, audit_df.loc[idx, "anomaly_reasons"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Kuartil gaji", q["label"])
    c2.metric("Target level berikutnya", fmt_money(q["target"], cur))
    c3.metric("Selisih menuju target", fmt_money(q["gap"], cur))
    st.caption(f"Pembanding: {q['group_name']} (n={q['pool_size']}).")

    negotiation = st.session_state["negotiations"].get(emp_id)
    if negotiation:
        st.info(f"Hasil negosiasi Modul 2 ikut dipertimbangkan: kenaikan {negotiation['final_pct']:.2f}%.")

    if st.button("🎯 Minta rekomendasi karier", type="primary"):
        try:
            client = get_client(cfg["api_key"])
            with st.spinner("Agen Penasihat Karier menyusun rekomendasi..."):
                answer = llm_chat(
                    client,
                    cfg["model"],
                    CAREER_SYSTEM,
                    build_career_prompt(facts, q, cur, negotiation),
                    temperature=cfg["temperature"],
                    max_tokens=900,
                )
            points = extract_key_points(answer)
            st.session_state["careers"][emp_id] = {
                "text": answer,
                "points": points,
                "time": now_str(),
                "model": cfg["model"],
                "quartile": q["label"],
                "gap": q["gap"],
            }
            add_history(
                "Penasihat Karier",
                cfg["model"],
                f"{emp_id} ({q['label']}): " + "; ".join(points[:2]),
                {"karyawan": emp_id, "kuartil": q["label"], "selisih_target": round(q["gap"]), "poin_utama": points,
                 "rekomendasi": answer},
            )
            st.markdown(answer)
        except LLMError as exc:
            st.error(str(exc))
    elif emp_id in st.session_state["careers"]:
        saved = st.session_state["careers"][emp_id]
        st.caption(f"Rekomendasi tersimpan | {saved['time']} | model {saved['model']}")
        st.markdown(saved["text"])


# =============================================================================
# 9. TAB 4: RINGKASAN EKSEKUTIF & EKSPOR
# =============================================================================

def csv_bytes(frame: pd.DataFrame) -> bytes:
    """CSV UTF-8 dengan BOM agar terbuka rapi di Excel."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def negotiations_frame() -> pd.DataFrame:
    """Hasil semua negosiasi sesi ini sebagai tabel."""
    rows = []
    for emp_id, r in st.session_state["negotiations"].items():
        rows.append(
            {
                "employee_id": emp_id,
                "status": r["status"],
                "plafon_%": round(r["ceiling_pct"], 2),
                "kenaikan_disepakati_%": round(r["final_pct"], 2),
                "gaji_sebelum": round(r["salary_before"]),
                "gaji_sesudah": round(r["salary_after"]),
                "kenaikan_per_bulan": round(r["monthly_increase"]),
                "diterapkan_ke_anggaran": r["applied"],
                "alasan": r["rationale"],
                "waktu": r.get("time", ""),
                "model": r.get("model", ""),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    notes: list[str],
    group_col: Optional[str],
    cfg: dict,
    whatif: dict,
) -> str:
    """Menyusun laporan ringkasan eksekutif (Markdown) dari state saat ini."""
    cur = cfg["currency"]
    n = len(df)
    anomalies = audit_df[audit_df["is_anomaly"]]
    L: list[str] = [
        "# Laporan Ringkasan Eksekutif - HR & Payroll Simulation",
        f"_Dibuat: {now_str()} | Model AI: {cfg['model']}_",
        "",
    ]
    if st.session_state["exec_ai"]:
        L += ["## Narasi Eksekutif (AI)", st.session_state["exec_ai"], ""]

    # --- 1. Audit
    L += ["## 1. Audit Payroll"]
    L += [
        f"- Karyawan dianalisis: {n:,}",
        f"- Total payroll/bulan: {fmt_money(df['salary'].sum(), cur)} | median: {fmt_money(df['salary'].median(), cur)}",
        f"- Anomali terdeteksi: {len(anomalies)} ({len(anomalies) / max(n, 1) * 100:.1f}%) "
        f"pada ambang z-grup {cfg['mz_thresh']:.2f} dan tren {cfg['resid_thresh']:.2f}",
    ]
    if group_col:
        gaps = compute_group_gaps(df, group_col)
        if len(gaps):
            top, low = gaps.iloc[0], gaps.iloc[-1]
            L.append(
                f"- Median tertinggi per {group_col}: {top[group_col]} ({fmt_money(top['median'], cur)}); "
                f"terendah: {low[group_col]} ({fmt_money(low['median'], cur)})"
            )
    gender_gap = compute_gender_gap(df)
    if gender_gap is not None and len(gender_gap) > 1:
        L.append("- Median gaji per gender: " + ", ".join(f"{r['gender']} {fmt_money(r['median'], cur)}" for _, r in gender_gap.iterrows())
                 + " (deskriptif, belum mengontrol faktor lain)")
    for nt in notes:
        L.append(f"- Catatan data: {nt}")
    if len(anomalies):
        L += ["", "Anomali teratas (berdasarkan gaji):"]
        for _, r in anomalies.sort_values("salary", ascending=False).head(5).iterrows():
            L.append(f"  - {r['employee_id']}: {fmt_money(r['salary'], cur)} - {r['anomaly_reasons']}")
    if st.session_state["audit_ai"]:
        L += ["", "**Ringkasan Agen Auditor (AI):**", st.session_state["audit_ai"]["text"]]

    # --- 2. What-if
    L += ["", "## 2. Simulasi Skenario Kebijakan & Anggaran"]
    L += [
        f"- Skenario aktif: {'Ya' if cfg['scenario_active'] else 'Tidak (pengaturan baku sidebar)'}",
        f"- Strategi: {whatif['strategy']}" + (f" (merit {whatif['merit_pct']:.1f}%)" if whatif["strategy"] != STRAT_CORRECT else ""),
        f"- Pool {whatif['cap_pct']:.1f}% dari payroll | batas individu {whatif['max_individual']:.1f}%",
        f"- Payroll awal: {fmt_money(whatif['total_before'], cur)} -> proyeksi: {fmt_money(whatif['total_after'], cur)} "
        f"(+{fmt_money(whatif['delta'], cur)}, {whatif['delta_pct']:.2f}%)",
        f"- Pool terpakai: {whatif['pool_used_pct']:.1f}% | penerima kenaikan: {whatif['n_recipients']:,} "
        f"| rata-rata kenaikan penerima: {whatif['avg_raise_pct']:.2f}%",
        f"- Anomali sebelum -> sesudah penyesuaian: {whatif['anom_before']} -> {whatif['anom_after']}",
    ]
    if whatif["scale"] < 1:
        L.append(f"- Catatan: kebutuhan melebihi pool, kenaikan diskalakan ke {whatif['scale'] * 100:.0f}%.")

    # --- 3. Negosiasi
    L += ["", "## 3. Negosiasi Multi-Agen"]
    negs = st.session_state["negotiations"]
    if negs:
        b = budget_state(df, cfg["cap_pct"])
        L.append(f"- Jumlah negosiasi sesi ini: {len(negs)} | anggaran terpakai: {fmt_money(b['committed'], cur)} dari pool {fmt_money(b['pool'], cur)}")
        for emp_id, r in negs.items():
            stat = "konsensus" if r["status"] == "consensus" else "tanpa konsensus"
            L.append(
                f"- {emp_id}: {stat}, kenaikan {r['final_pct']:.2f}% "
                f"({fmt_money(r['salary_before'], cur)} -> {fmt_money(r['salary_after'], cur)})"
                f"{' [diterapkan]' if r['applied'] else ''}. {r['rationale']}"
            )
    else:
        L.append("_Belum ada negosiasi yang dijalankan._")

    # --- 4. Karier
    L += ["", "## 4. Penasihat Karier"]
    careers = st.session_state["careers"]
    if careers:
        for emp_id, c in careers.items():
            L.append(f"- **{emp_id}** ({c['quartile']}, selisih ke target {fmt_money(c['gap'], cur)}):")
            for p in c["points"]:
                L.append(f"  - {p}")
    else:
        L.append("_Belum ada rekomendasi karier yang dibuat._")

    L += ["", "---", "_Catatan: keluaran AI bersifat simulatif dan bukan nasihat hukum/keuangan. "
          "Keputusan kompensasi nyata perlu ditinjau manusia._"]
    return "\n".join(L)


def tab_export(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    notes: list[str],
    group_col: Optional[str],
    cfg: dict,
    whatif: dict,
) -> None:
    """Tab 4: ringkasan eksekutif terpadu, unduhan, dan riwayat sesi."""
    cur = cfg["currency"]
    st.subheader("Ringkasan Eksekutif & Ekspor")

    negs = st.session_state["negotiations"]
    careers = st.session_state["careers"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anomali payroll", f"{int(audit_df['is_anomaly'].sum())}")
    k2.metric("Proyeksi tambahan payroll", fmt_money(whatif["delta"], cur), delta=f"{whatif['delta_pct']:.2f}%", delta_color="off")
    k3.metric("Negosiasi dijalankan", f"{len(negs)}")
    k4.metric("Rekomendasi karier", f"{len(careers)}")

    if st.button("🤖 Buat narasi eksekutif dengan AI (opsional)"):
        try:
            client = get_client(cfg["api_key"])
            st.session_state["exec_ai"] = ""  # agar narasi lama tidak ikut masuk prompt
            base_report = build_report(df, audit_df, notes, group_col, cfg, whatif)
            with st.spinner("Menyusun narasi eksekutif..."):
                narrative = llm_chat(
                    client,
                    cfg["model"],
                    "Anda konsultan HR senior. Tulis ringkasan eksekutif untuk pimpinan dalam Bahasa Indonesia, maksimal 200 kata: "
                    "temuan kunci, dampak anggaran, keputusan negosiasi (jika ada), dan langkah lanjut. Gunakan hanya data yang diberikan.",
                    base_report[:8000],
                    temperature=cfg["temperature"],
                    max_tokens=600,
                )
            st.session_state["exec_ai"] = narrative
            add_history("Ringkasan Eksekutif (AI)", cfg["model"], narrative[:140].replace("\n", " ") + "...", {"narasi": narrative})
        except LLMError as exc:
            st.error(str(exc))

    report = build_report(df, audit_df, notes, group_col, cfg, whatif)
    st.markdown(report)

    st.markdown("### ⬇️ Unduh")
    stamp = datetime.now(WIB).strftime("%Y%m%d_%H%M")
    d1, d2 = st.columns(2)
    d1.download_button("📄 Laporan ringkasan (.md)", report.encode("utf-8"), f"laporan_hr_payroll_{stamp}.md", "text/markdown")
    d2.download_button("📄 Laporan ringkasan (.txt)", report.encode("utf-8"), f"laporan_hr_payroll_{stamp}.txt", "text/plain")

    d3, d4, d5 = st.columns(3)
    anomalies = audit_df[audit_df["is_anomaly"]]
    cols = [c for c in ("employee_id", "position", "department", "experience", "salary", "anomaly_reasons") if c in anomalies.columns]
    d3.download_button("📊 Data anomali (.csv)", csv_bytes(anomalies[cols]), f"anomali_payroll_{stamp}.csv", "text/csv")
    d4.download_button("📊 Hasil simulasi what-if (.csv)", csv_bytes(whatif["detail"]), f"simulasi_whatif_{stamp}.csv", "text/csv")
    if negs:
        d5.download_button("📊 Hasil negosiasi (.csv)", csv_bytes(negotiations_frame()), f"hasil_negosiasi_{stamp}.csv", "text/csv")
    else:
        d5.caption("CSV negosiasi tersedia setelah Modul 2 dijalankan.")

    st.divider()
    render_history()


# =============================================================================
# 10. TITIK MASUK
# =============================================================================

def employee_selector(df: pd.DataFrame, audit_df: pd.DataFrame, cur: str) -> int:
    """Selectbox karyawan fokus yang dipakai bersama oleh Modul 2 dan 3."""
    only_flagged = st.checkbox("Hanya tampilkan karyawan yang ditandai anomali", value=False)
    pool = audit_df[audit_df["is_anomaly"]] if only_flagged and audit_df["is_anomaly"].any() else audit_df

    def label(i: int) -> str:
        r = df.loc[i]
        parts = [str(r["employee_id"])]
        for c in ("position", "department"):
            if c in df.columns and pd.notna(r[c]):
                parts.append(str(r[c]))
        parts.append(fmt_money(r["salary"], cur))
        return " | ".join(parts)

    return st.selectbox("Karyawan fokus (dipakai Modul 2 & 3)", list(pool.index), format_func=label)


def main() -> None:
    """Titik masuk aplikasi."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    init_session_state()

    st.title("🧭 " + APP_TITLE)
    st.caption("Audit payroll -> negosiasi multi-agen -> penasihat karier, dalam satu alur kerja. Didukung Groq API.")

    cfg = sidebar_config()
    raw = load_raw_data(cfg)
    if raw is None:
        st.info("Pilih sumber data di sidebar: unggah CSV dari Kaggle, letakkan di `data/employee_salary_dataset.csv`, atau pakai data contoh.")
        return

    mapping = column_mapping_ui(raw)
    if not mapping.get("salary"):
        st.error("Kolom gaji belum dipetakan. Pilih kolom gaji di sidebar > Pemetaan kolom.")
        render_data_readiness(raw, mapping)
        st.dataframe(raw.head(20))
        return

    df, notes = prepare_dataframe(raw, mapping)
    if df.empty:
        st.error("Tidak ada baris valid setelah pembersihan data (periksa isi kolom gaji).")
        render_data_readiness(raw, mapping)
        return

    # Pilihan kolom pengelompokan (posisi/departemen/dll.) untuk audit dan kuartil.
    group_options = [c for c in ("position", "department", "education", "city") if c in df.columns]
    group_col: Optional[str] = None
    if group_options:
        group_col = st.sidebar.selectbox(
            "Kelompokkan berdasarkan", group_options, format_func=lambda c: FIELD_LABELS[c]
        )

    if cfg["source"] == "Data contoh (sintetis)":
        st.warning("Anda memakai data SINTETIS untuk demo, bukan dataset Kaggle asli.")

    audit_df = run_audit(df, group_col, cfg["mz_thresh"], cfg["resid_thresh"])
    whatif = simulate_scenario(df, audit_df, group_col, cfg)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 Data", "1️⃣ Audit Payroll", "2️⃣ Negosiasi Multi-Agen", "3️⃣ Penasihat Karier", "4️⃣ Ringkasan & Ekspor"]
    )

    with tab0:
        tab_data(raw, mapping, df, notes)

    with tab1:
        tab_audit(df, audit_df, notes, group_col, cfg, whatif)

    with tab2:
        idx2 = employee_selector(df, audit_df, cfg["currency"])
        tab_negotiation(df, audit_df, idx2, group_col, cfg)

    with tab3:
        st.caption("Karyawan yang dipilih di tab 2 tidak otomatis terbawa; pilih ulang di bawah bila perlu.")
        idx3 = st.selectbox(
            "Karyawan fokus",
            list(df.index),
            format_func=lambda i: f"{df.loc[i, 'employee_id']} | {fmt_money(df.loc[i, 'salary'], cfg['currency'])}",
            key="career_employee",
        )
        tab_career(df, audit_df, idx3, group_col, cfg)

    with tab4:
        tab_export(df, audit_df, notes, group_col, cfg, whatif)

    st.divider()
    st.caption(
        "Catatan: keluaran LLM bersifat simulatif dan bukan nasihat hukum/keuangan. "
        "Keputusan kompensasi nyata perlu ditinjau manusia."
    )


if __name__ == "__main__":
    main()
