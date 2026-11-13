"""
Integrated Multi-Agent HR & Payroll Simulation System
=====================================================
Platform simulasi dan tata kelola SDM berbasis multi-agen.

Alur end-to-end (data yang sama mengalir antar modul):
    Data gaji  ->  Modul 1: Audit & deteksi anomali
               ->  Modul 2: Negosiasi hierarkis multi-agen (HR vs Keuangan vs Karyawan)
               ->  Modul 3: Penasihat karier (kuartil gaji + jalur kompetensi)

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

# Dugaan nama kolom untuk pemetaan otomatis (huruf kecil, spasi -> underscore).
# Nama kolom persis dataset Kaggle dapat berbeda; pengguna dapat mengoreksi
# pemetaan lewat sidebar.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "salary": ["monthly_salary", "salary", "gaji", "annual_salary", "ctc", "salary_inr"],
    "department": ["department", "dept", "departemen"],
    "position": ["job_title", "jobtitle", "position", "role", "designation", "job_role", "title"],
    "experience": ["experience_years", "experience", "years_of_experience", "years_experience", "tenure", "masa_kerja"],
    "education": ["education_level", "education", "pendidikan"],
    "age": ["age", "umur", "usia"],
    "gender": ["gender", "sex", "jenis_kelamin"],
    "city": ["city", "location", "kota"],
    "employee_id": ["employee_id", "emp_id", "empid", "id"],
    "performance": ["performance_score", "performance", "performance_rating", "rating"],
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

NONE_OPTION = "(tidak ada)"

MODEL_OPTIONS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "(ketik manual)"]


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
    """Membaca CSV dari path atau file-upload, mencoba beberapa encoding."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = source.getvalue()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Tidak dapat membaca file CSV (encoding tidak dikenali).")


def detect_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Menebak kolom mana yang berperan sebagai gaji, departemen, dst."""
    normalized = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    mapping: dict[str, Optional[str]] = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        mapping[field] = next((normalized[c] for c in candidates if c in normalized), None)
    return mapping


def prepare_dataframe(raw: pd.DataFrame, mapping: dict[str, Optional[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Membentuk DataFrame standar (kolom: salary, department, ...) dan daftar catatan kualitas data."""
    notes: list[str] = []
    out = pd.DataFrame()
    for field, col in mapping.items():
        if col and col in raw.columns:
            out[field] = raw[col]

    # Gaji wajib numerik.
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    missing_salary = int(out["salary"].isna().sum())
    if missing_salary:
        notes.append(f"{missing_salary} baris dibuang karena gaji kosong / bukan angka.")
    out = out.dropna(subset=["salary"])

    for numeric_field in ("experience", "age", "performance"):
        if numeric_field in out.columns:
            out[numeric_field] = pd.to_numeric(out[numeric_field], errors="coerce")

    out = out.reset_index(drop=True)
    if "employee_id" not in out.columns:
        out["employee_id"] = [f"E{i + 1:04d}" for i in range(len(out))]
        notes.append("Kolom ID karyawan tidak ada; ID dibuat otomatis.")
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
    """Satu panggilan chat completion ke Groq dengan penanganan error yang rinci."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
    "**Roadmap 0-3 / 3-6 / 6-12 Bulan**, **Indikator Keberhasilan**, **Catatan Realistis**. "
    "Maksimal sekitar 350 kata."
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


# =============================================================================
# 7. ANTARMUKA STREAMLIT
# =============================================================================

def init_session_state() -> None:
    """Menyiapkan state yang bertahan antar interaksi Streamlit."""
    st.session_state.setdefault("committed_raise", 0.0)  # total kenaikan (per bulan) yang sudah disetujui
    st.session_state.setdefault("negotiations", {})  # emp_id -> hasil negosiasi


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
    model_choice = sb.selectbox("Model", MODEL_OPTIONS, index=0)
    model = sb.text_input("Nama model", value="llama-3.3-70b-versatile") if model_choice == "(ketik manual)" else model_choice
    temperature = sb.slider("Temperature", 0.0, 1.0, 0.4, 0.05)

    if sb.button("🔌 Uji koneksi API"):
        try:
            client = get_client(api_key)
            reply = llm_chat(client, model, "Balas singkat.", "Jawab dengan kata: OK", temperature=0.0, max_tokens=10)
            sb.success(f"Koneksi berhasil. Balasan model: {reply}")
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

    return {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "source": source,
        "uploaded": uploaded,
        "currency": currency,
        "cap_pct": cap_pct,
        "max_individual": max_individual,
        "rounds": rounds,
        "mz_thresh": mz_thresh,
        "resid_thresh": resid_thresh,
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
    mapping: dict[str, Optional[str]] = {}
    with st.sidebar.expander("🧭 Pemetaan kolom", expanded=detected["salary"] is None):
        for field, label in FIELD_LABELS.items():
            default = detected[field] if detected[field] in options else NONE_OPTION
            choice = st.selectbox(label, options, index=options.index(default), key=f"map_{field}")
            mapping[field] = None if choice == NONE_OPTION else choice
    return mapping


def tab_audit(df: pd.DataFrame, audit_df: pd.DataFrame, notes: list[str], group_col: Optional[str], cfg: dict) -> None:
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
            st.markdown(answer)
        except LLMError as exc:
            st.error(str(exc))


def tab_negotiation(df: pd.DataFrame, audit_df: pd.DataFrame, idx: int, group_col: Optional[str], cfg: dict) -> None:
    """Tab Modul 2: simulasi negosiasi multi-agen."""
    cur = cfg["currency"]
    st.subheader("Modul 2 - Hierarchical Decision-Making & Negotiation")

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
            st.session_state["negotiations"][emp_id] = result
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
            st.markdown(answer)
        except LLMError as exc:
            st.error(str(exc))


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
        st.dataframe(raw.head(20))
        return

    df, notes = prepare_dataframe(raw, mapping)
    if df.empty:
        st.error("Tidak ada baris valid setelah pembersihan data.")
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

    tab0, tab1, tab2, tab3 = st.tabs(["📄 Data", "1️⃣ Audit Payroll", "2️⃣ Negosiasi Multi-Agen", "3️⃣ Penasihat Karier"])

    with tab0:
        st.subheader("Pratinjau data")
        st.caption(f"{len(df):,} baris valid dari {len(raw):,} baris mentah.")
        st.dataframe(df.head(200), hide_index=True)

    with tab1:
        tab_audit(df, audit_df, notes, group_col, cfg)

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

    st.divider()
    st.caption(
        "Catatan: keluaran LLM bersifat simulatif dan bukan nasihat hukum/keuangan. "
        "Keputusan kompensasi nyata perlu ditinjau manusia."
    )


if __name__ == "__main__":
    main()
