"""
DCVPA-C Analytics Dashboard
===========================
Streamlit dashboard for the Digitalized Comprehensive Visual Perception
Assessment - Children. Reads registrations + scores from Firestore.

Run:
    pip install -r requirements.txt
    streamlit run dashboard.py

Auth (data sync, admin-only):
    Place a Firebase Admin SDK service account JSON next to this script as
    `serviceAccountKey.json`, OR set the env var GOOGLE_APPLICATION_CREDENTIALS
    pointing to it, OR upload it via the sidebar at runtime.

Auth (dashboard login):
    Every viewer signs in with an email + password checked against Firebase
    Authentication. Requires a Firebase Web API key set as either the
    Streamlit secret `FIREBASE_WEB_API_KEY` or the env var of the same name.

    Access level per email:
      - Emails listed in the `ADMIN_EMAILS` secret/env var (comma-separated)
        get the full, unrestricted dashboard (all participants + sync UI).
      - Every other email lands on a restricted "My data" view scoped to the
        registrations whose `adminEmail` field matches their login email.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import firebase_admin
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from firebase_admin import credentials, firestore

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

DOMAINS: list[str] = [
    "Visual Attention",
    "Visual Memory",
    "Visual Discrimination",
    "Visual Form Constancy",
    "Visual Figure Ground",
    "Visual Closure",
    "Visual Spatial Relationships",
    "Visual Topography",
    "Global Motion Perception",
    "Local Motion Perception",
    "Motion Speed",
]

DOMAIN_GROUPS: dict[str, list[int]] = {
    "Visual Foundation Skills": [0, 1, 2],
    "Object-Based Visual Skills": [3, 4, 5],
    "Space & Motion Visual Skills": [6, 7, 8, 9, 10],
}

LEVELS_PER_DOMAIN = 10

# Local data cache (JSON files persisted to disk, loaded on every page render)
DATA_DIR = Path(__file__).parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)
REGISTRATIONS_FILE = DATA_DIR / "registrations.json"
SCORES_FILE = DATA_DIR / "scores.json"
META_FILE = DATA_DIR / "sync_metadata.json"

FIRESTORE_IDENTITY_TOOLKIT_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
)

st.set_page_config(
    page_title="DCVPA-C Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Firebase initialisation                                                     #
# --------------------------------------------------------------------------- #


def _init_firebase_from_dict(cred_dict: dict[str, Any]) -> firestore.Client:
    """Initialise (or reuse) a Firebase Admin app and return a Firestore client."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    return firestore.client()


@st.cache_resource(show_spinner=False)
def get_db(cred_source: str, cred_payload: str) -> firestore.Client | None:
    """
    Cache the Firestore client across reruns.

    cred_source: 'file' or 'upload'
    cred_payload: file path (for 'file') or JSON string (for 'upload')
    """
    try:
        if cred_source == "file":
            with open(cred_payload, "r") as f:
                cred_dict = json.load(f)
        else:
            cred_dict = json.loads(cred_payload)
        return _init_firebase_from_dict(cred_dict)
    except Exception as e:
        st.sidebar.error(f"Firebase init failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Login (Firebase Authentication)                                            #
# --------------------------------------------------------------------------- #


def _get_secret(name: str) -> str | None:
    """Look up a config value from Streamlit secrets, falling back to env vars."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def _admin_emails() -> set[str]:
    raw = _get_secret("ADMIN_EMAILS") or ""
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def firebase_sign_in_with_password(email: str, password: str, api_key: str) -> tuple[bool, str]:
    """Verify email/password against Firebase Authentication. Returns (ok, message)."""
    try:
        resp = requests.post(
            FIRESTORE_IDENTITY_TOOLKIT_URL,
            params={"key": api_key},
            json={"email": email, "password": password, "returnSecureToken": True},
            timeout=10,
        )
    except requests.RequestException as e:
        return False, f"Network error contacting Firebase: {e}"

    if resp.status_code == 200:
        return True, "ok"

    try:
        msg = resp.json().get("error", {}).get("message", "Login failed")
    except Exception:
        msg = "Login failed"
    return False, msg


def resolve_access(email: str, reg_df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Determine what an authenticated email is allowed to see.

    Returns (role, registration_ids):
      - ("admin", [])          — full unrestricted dashboard
      - ("user", [<VPD IDs>])  — restricted to registrations whose
                                  `adminEmail` field matches this email
    """
    email = email.lower()
    if email in _admin_emails():
        return "admin", []

    if reg_df.empty or "adminEmail" not in reg_df.columns or "registrationId" not in reg_df.columns:
        return "user", []

    mask = reg_df["adminEmail"].astype(str).str.strip().str.lower() == email
    return "user", reg_df.loc[mask, "registrationId"].dropna().tolist()


def render_login() -> None:
    st.title("🔒 DCVPA-C Dashboard — Sign in")

    api_key = _get_secret("FIREBASE_WEB_API_KEY")
    if not api_key:
        st.error(
            "Login isn't configured yet: set `FIREBASE_WEB_API_KEY` "
            "(Firebase project's Web API key) in Streamlit secrets or as an "
            "environment variable."
        )
        return

    with st.form("login_form"):
        email = st.text_input("Email").strip().lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        if not email or not password:
            st.warning("Enter both email and password.")
            return
        ok, message = firebase_sign_in_with_password(email, password, api_key)
        if ok:
            st.session_state["auth_email"] = email
            st.rerun()
        else:
            st.error(f"Sign-in failed: {message}")


# --------------------------------------------------------------------------- #
# Data fetching                                                               #
# --------------------------------------------------------------------------- #


def _json_default(obj: Any) -> Any:
    """JSON serializer for Firestore types (timestamps, etc.)."""
    if hasattr(obj, "isoformat"):  # datetime, date
        return obj.isoformat()
    return str(obj)


def _fetch_all_registrations(db: firestore.Client) -> list[dict[str, Any]]:
    """Pull all registration docs as a plain list of dicts."""
    docs = db.collection("registrations").stream()
    rows: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.to_dict() or {}
        data["_id"] = doc.id
        rows.append(data)
    return rows


def _fetch_scores_for_vpd(db: firestore.Client, vpd_id: str) -> dict[str, dict]:
    """{date: {domain: {Level N: {score, timestamp}}}} for one VPD."""
    base = db.collection("scores").document(vpd_id)
    out: dict[str, dict] = {}
    for sub in base.collections():
        out[sub.id] = {}
        for d in sub.stream():
            out[sub.id][d.id] = d.to_dict() or {}
    return out


def sync_from_firestore(db: firestore.Client) -> dict[str, Any]:
    """
    Pull *everything* from Firestore and write it to disk as JSON.

    Returns a metadata dict: {synced_at, registrations_count, scores_vpd_count, ...}
    """
    progress = st.progress(0.0, text="Syncing from Firestore…")

    # --- Registrations -------------------------------------------------- #
    progress.progress(0.05, text="Fetching registrations…")
    registrations = _fetch_all_registrations(db)

    REGISTRATIONS_FILE.write_text(
        json.dumps(registrations, indent=2, default=_json_default),
        encoding="utf-8",
    )

    # --- Scores --------------------------------------------------------- #
    vpd_ids = [r.get("registrationId") or r.get("_id") for r in registrations]
    vpd_ids = [v for v in vpd_ids if v]

    scores: dict[str, dict] = {}
    n = max(len(vpd_ids), 1)
    for i, vpd_id in enumerate(vpd_ids):
        try:
            scores[vpd_id] = _fetch_scores_for_vpd(db, vpd_id)
        except Exception as e:
            st.warning(f"Could not read scores for {vpd_id}: {e}")
            scores[vpd_id] = {}
        progress.progress(
            0.1 + 0.85 * (i + 1) / n,
            text=f"Fetching scores: {i + 1}/{n} ({vpd_id})",
        )

    SCORES_FILE.write_text(
        json.dumps(scores, indent=2, default=_json_default),
        encoding="utf-8",
    )

    # --- Metadata ------------------------------------------------------- #
    progress.progress(0.97, text="Finalising…")
    sessions_count = sum(
        len(date_map) > 0
        for vpd_map in scores.values()
        for date_map in vpd_map.values()
    )
    score_docs_count = sum(
        len(domain_map)
        for vpd_map in scores.values()
        for domain_map in vpd_map.values()
    )

    meta = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "registrations_count": len(registrations),
        "scores_vpd_count": len(scores),
        "score_sessions_count": sessions_count,
        "score_docs_count": score_docs_count,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    progress.progress(1.0, text="Done")
    progress.empty()
    return meta


# --------------------------------------------------------------------------- #
# Local data loading (from JSON cache on disk)                                #
# --------------------------------------------------------------------------- #


def get_sync_metadata() -> dict[str, Any] | None:
    """Return last-sync metadata, or None if nothing's synced yet."""
    if not META_FILE.exists():
        return None
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner="Loading registrations from local cache…")
def load_registrations_local(file_mtime: float) -> pd.DataFrame:
    """
    Load registrations from the local JSON. The mtime arg is a cache-busting
    key — when the file changes, the cache invalidates.
    """
    if not REGISTRATIONS_FILE.exists():
        return pd.DataFrame()

    rows = json.loads(REGISTRATIONS_FILE.read_text(encoding="utf-8"))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "createdAt" in df.columns:
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

    if "consent" in df.columns:
        df["consent_assessment"] = df["consent"].apply(
            lambda c: c.get("assessment") if isinstance(c, dict) else None
        )
    return df


@st.cache_data(show_spinner="Loading scores from local cache…")
def load_scores_local(file_mtime: float) -> dict[str, dict]:
    """Load the nested scores dict from the local JSON."""
    if not SCORES_FILE.exists():
        return {}
    return json.loads(SCORES_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Building score DataFrame…")
def scores_dict_to_long_df(file_mtime: float) -> pd.DataFrame:
    """
    Convert the nested local scores dict into a long-format DataFrame:
    vpd_id, date, domain, level, score, correct, response_ms.
    """
    if not SCORES_FILE.exists():
        return pd.DataFrame(columns=[
            "vpd_id", "date", "domain", "level", "score", "correct", "response_ms"
        ])

    scores = json.loads(SCORES_FILE.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for vpd_id, date_map in scores.items():
        if not isinstance(date_map, dict):
            continue
        for date_key, domain_map in date_map.items():
            if not isinstance(domain_map, dict):
                continue
            for domain_name, level_map in domain_map.items():
                if not isinstance(level_map, dict):
                    continue
                for level_key, level_payload in level_map.items():
                    if not isinstance(level_payload, dict):
                        continue
                    try:
                        level_num = int(str(level_key).split()[-1])
                    except (ValueError, IndexError):
                        continue
                    score_val = level_payload.get("score")
                    rt_val = level_payload.get("timestamp")
                    rows.append({
                        "vpd_id": vpd_id,
                        "date": date_key,
                        "domain": domain_name,
                        "level": level_num,
                        "score": score_val,
                        "correct": bool(score_val) if score_val is not None else None,
                        "response_ms": rt_val,
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "vpd_id", "date", "domain", "level", "score", "correct", "response_ms"
        ])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_file_mtime(path: Path) -> float:
    """Cache-busting key — returns 0 if the file doesn't exist."""
    return path.stat().st_mtime if path.exists() else 0.0


# --------------------------------------------------------------------------- #
# Computation helpers                                                          #
# --------------------------------------------------------------------------- #


def parse_age_to_years(age_str: Any) -> float | None:
    """Parse 'X years, Y months' (the form's stored format) to a float year value."""
    if not isinstance(age_str, str):
        return None
    try:
        years_part, _, months_part = age_str.partition(",")
        years = int(years_part.strip().split()[0])
        months = 0
        if months_part:
            months = int(months_part.strip().split()[0])
        return years + months / 12.0
    except (ValueError, IndexError):
        return None


def parse_numeric(val: Any) -> float | None:
    """Best-effort numeric parse. Returns None if not parseable."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def bmi_category(bmi: float | None) -> str:
    if bmi is None:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def safe_nested_get(series: pd.Series, key: str) -> pd.Series:
    """For a Series of dicts, pull out a sub-key as a Series."""
    return series.apply(lambda d: d.get(key) if isinstance(d, dict) else None)


def compute_domain_summary(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Per (vpd_id, date, domain): correct count, total time, accuracy."""
    if scores_long.empty:
        return pd.DataFrame()

    grouped = (
        scores_long.groupby(["vpd_id", "date", "domain"], dropna=False)
        .agg(
            correct=("correct", lambda s: int(pd.Series(s).fillna(False).sum())),
            attempted=("level", "count"),
            total_time_ms=("response_ms", lambda s: float(pd.Series(s).fillna(0).sum())),
        )
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / LEVELS_PER_DOMAIN
    return grouped


def compute_grouped_totals(domain_summary: pd.DataFrame) -> pd.DataFrame:
    """Roll up domain summary into the three skill groups + grand total per session."""
    if domain_summary.empty:
        return pd.DataFrame()

    domain_to_group = {}
    for group, idxs in DOMAIN_GROUPS.items():
        for i in idxs:
            domain_to_group[DOMAINS[i]] = group

    df = domain_summary.copy()
    df["group"] = df["domain"].map(domain_to_group)

    group_totals = (
        df.groupby(["vpd_id", "date", "group"], dropna=False)
        .agg(
            correct=("correct", "sum"),
            total_items=("domain", lambda s: len(s) * LEVELS_PER_DOMAIN),
            total_time_ms=("total_time_ms", "sum"),
        )
        .reset_index()
    )
    group_totals["accuracy"] = group_totals["correct"] / group_totals["total_items"]

    grand = (
        df.groupby(["vpd_id", "date"], dropna=False)
        .agg(
            correct=("correct", "sum"),
            total_items=("domain", lambda s: len(s) * LEVELS_PER_DOMAIN),
            total_time_ms=("total_time_ms", "sum"),
        )
        .reset_index()
    )
    grand["group"] = "Visual Perception Total"
    grand["accuracy"] = grand["correct"] / grand["total_items"]

    return pd.concat([group_totals, grand], ignore_index=True)


# --------------------------------------------------------------------------- #
# Sidebar: credentials + navigation                                            #
# --------------------------------------------------------------------------- #


def sidebar_credentials() -> firestore.Client | None:
    # Auto-collapse the connection panel if local cache already exists,
    # since the user only needs it for syncing.
    has_cache = META_FILE.exists()
    with st.sidebar.expander("🔐 Firebase connection", expanded=not has_cache):
        default_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json"
        )
        has_local = os.path.exists(default_path)

        mode = st.radio(
            "Credential source",
            ["Local file", "Upload JSON"],
            index=0 if has_local else 1,
            help="Admin SDK service account key — NOT the web apiKey.",
        )

        if mode == "Local file":
            path = st.text_input("Service account JSON path", value=default_path)
            if not os.path.exists(path):
                st.warning(f"File not found at: {path}")
                return None
            client = get_db("file", path)
            if client is not None:
                st.success("Connected ✓")
            return client

        uploaded = st.file_uploader("Upload service account JSON", type=["json"])
        if uploaded is None:
            st.info("Upload your service account JSON to connect.")
            return None
        payload = uploaded.read().decode("utf-8")
        client = get_db("upload", payload)
        if client is not None:
            st.success("Connected ✓")
        return client


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #


def page_overview(reg_df: pd.DataFrame, scores_long: pd.DataFrame):
    st.title("📊 Analytics Overview")

    if reg_df.empty:
        st.info("No registrations found yet.")
        return

    # ---------------------------------------------------------------- #
    # Derive helper columns                                            #
    # ---------------------------------------------------------------- #
    df = reg_df.copy()
    df["gender_clean"] = df.get("gender", pd.Series(index=df.index)).fillna("unspecified").str.lower()
    df["age_years"] = df.get("age", pd.Series(index=df.index)).apply(parse_age_to_years)
    df["bmi_num"] = df.get("bmi", pd.Series(index=df.index)).apply(parse_numeric)
    df["bmi_cat"] = df["bmi_num"].apply(bmi_category)
    df["height_num"] = df.get("height", pd.Series(index=df.index)).apply(parse_numeric)
    df["weight_num"] = df.get("weight", pd.Series(index=df.index)).apply(parse_numeric)
    df["screen_hrs"] = df.get("screenTimeHome", pd.Series(index=df.index))

    male_count = (df["gender_clean"] == "male").sum()
    female_count = (df["gender_clean"] == "female").sum()
    other_count = ((df["gender_clean"] != "male") & (df["gender_clean"] != "female")).sum()
    total = len(df)

    # ---------------------------------------------------------------- #
    # Top KPI row                                                      #
    # ---------------------------------------------------------------- #
    st.markdown("### Key metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total participants", f"{total}")
    k2.metric("Male", f"{male_count}", delta=f"{male_count/total:.0%}" if total else None)
    k3.metric("Female", f"{female_count}", delta=f"{female_count/total:.0%}" if total else None)
    k4.metric("Other / unspecified", f"{other_count}")
    avg_age = df["age_years"].mean()
    k5.metric("Avg age (yrs)", f"{avg_age:.1f}" if pd.notna(avg_age) else "—")
    k6.metric("Unique schools", df["schoolName"].nunique() if "schoolName" in df else 0)

    st.markdown("---")

    # ---------------------------------------------------------------- #
    # Section 1: Demographics                                          #
    # ---------------------------------------------------------------- #
    st.markdown("### 👥 Demographics")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Gender split**")
        gender_counts = df["gender_clean"].replace({"unspecified": "Unspecified"}).str.title().value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.pie(
            gender_counts, names="Gender", values="Count", hole=0.5,
            color="Gender",
            color_discrete_map={"Male": "#3B82F6", "Female": "#EC4899", "Other": "#A78BFA", "Unspecified": "#9CA3AF"},
        )
        fig.update_traces(textposition="outside", textinfo="label+percent+value")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Age distribution (years)**")
        age_data = df["age_years"].dropna()
        if len(age_data):
            fig = px.histogram(age_data, nbins=20, labels={"value": "Age (years)"})
            fig.update_traces(marker_color="#6366F1")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                              showlegend=False, bargap=0.05, xaxis_title="Age (years)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No age data parseable yet.")

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("**Age × Gender**")
        ag = df.dropna(subset=["age_years"])
        if len(ag):
            fig = px.box(ag, x="gender_clean", y="age_years", color="gender_clean", points="all",
                         labels={"gender_clean": "Gender", "age_years": "Age (years)"})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Age data unavailable.")

    with c4:
        st.markdown("**Residential setting**")
        if "residentialType" in df.columns:
            rc = df["residentialType"].fillna("Unknown").value_counts().reset_index()
            rc.columns = ["Type", "Count"]
            fig = px.pie(rc, names="Type", values="Count", hole=0.5,
                         color_discrete_sequence=["#10B981", "#F59E0B", "#9CA3AF"])
            fig.update_traces(textinfo="label+percent+value")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------- #
    # Section 2: Body & health                                         #
    # ---------------------------------------------------------------- #
    st.markdown("### 🩺 Body & Health")
    h1, h2 = st.columns([1, 1])

    with h1:
        st.markdown("**BMI distribution**")
        bmi_data = df["bmi_num"].dropna()
        if len(bmi_data):
            fig = px.histogram(bmi_data, nbins=20, labels={"value": "BMI"})
            fig.update_traces(marker_color="#14B8A6")
            # Reference lines for BMI categories — short labels, alternating
            # vertical position to keep them from colliding with each other
            # or with the bar tops.
            cutoffs = [
                (18.5, "18.5", 0.95),
                (25,   "25",   0.85),
                (30,   "30",   0.95),
            ]
            for x, label, ypos in cutoffs:
                fig.add_vline(
                    x=x, line_dash="dash", line_color="grey",
                    annotation_text=label,
                    annotation_position="top",
                    annotation_yref="paper",
                    annotation_y=ypos,
                    annotation_font=dict(size=10, color="grey"),
                )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=False,
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Dashed lines mark WHO BMI cutoffs: 18.5 (underweight→normal), 25 (→overweight), 30 (→obese).")
        else:
            st.caption("No BMI data yet.")

    with h2:
        st.markdown("**BMI category**")
        cat = df["bmi_cat"].value_counts().reset_index()
        cat.columns = ["Category", "Count"]
        order = ["Underweight", "Normal", "Overweight", "Obese", "Unknown"]
        cat["Category"] = pd.Categorical(cat["Category"], categories=order, ordered=True)
        cat = cat.sort_values("Category")
        fig = px.bar(
            cat, x="Category", y="Count", color="Category", text="Count",
            color_discrete_map={
                "Underweight": "#FBBF24", "Normal": "#10B981",
                "Overweight": "#F59E0B", "Obese": "#EF4444", "Unknown": "#9CA3AF",
            },
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    h3, h4 = st.columns([1, 1])
    with h3:
        st.markdown("**Vision status**")
        if "visionStatus" in df.columns:
            vs = df["visionStatus"].fillna("Not recorded").replace("", "Not recorded").value_counts().reset_index()
            vs.columns = ["Status", "Count"]
            fig = px.pie(vs, names="Status", values="Count", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textinfo="label+percent+value")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with h4:
        st.markdown("**Hearing status**")
        if "hearingStatus" in df.columns:
            hs = df["hearingStatus"].fillna("Not recorded").replace("", "Not recorded").value_counts().reset_index()
            hs.columns = ["Status", "Count"]
            fig = px.pie(hs, names="Status", values="Count", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textinfo="label+percent+value")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Visual aids breakdown
    if "visionAids" in df.columns:
        st.markdown("**Visual aids in use**")
        glasses = safe_nested_get(df["visionAids"], "glasses").fillna(False).astype(bool).sum()
        lenses = safe_nested_get(df["visionAids"], "lenses").fillna(False).astype(bool).sum()
        others = safe_nested_get(df["visionAids"], "others").fillna(False).astype(bool).sum()
        none = total - max(glasses + lenses + others, 0)
        none = max(none, 0)
        aids_df = pd.DataFrame({
            "Aid": ["Glasses", "Contact lenses", "Other", "None / not specified"],
            "Count": [int(glasses), int(lenses), int(others), int(none)],
        })
        fig = px.bar(aids_df, x="Aid", y="Count", text="Count", color="Aid",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------- #
    # Section 3: Education                                             #
    # ---------------------------------------------------------------- #
    st.markdown("### 🏫 Education")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("**Education type**")
        if "educationType" in df.columns:
            et = df["educationType"].fillna("Not specified").replace("", "Not specified").value_counts().reset_index()
            et.columns = ["Type", "Count"]
            fig = px.bar(et, x="Type", y="Count", text="Count", color="Type",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with e2:
        st.markdown("**School type**")
        if "schoolType" in df.columns:
            sc = df["schoolType"].fillna("Not specified").replace("", "Not specified").value_counts().reset_index()
            sc.columns = ["Type", "Count"]
            fig = px.pie(sc, names="Type", values="Count", hole=0.5,
                         color_discrete_sequence=["#3B82F6", "#F97316", "#9CA3AF"])
            fig.update_traces(textinfo="label+percent+value")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with e3:
        st.markdown("**School board**")
        if "schoolBoard" in df.columns:
            sb = df["schoolBoard"].fillna("Not specified").replace("", "Not specified").value_counts().reset_index()
            sb.columns = ["Board", "Count"]
            fig = px.bar(sb, x="Board", y="Count", text="Count", color="Board",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------- #
    # Section 4: Languages & environment                               #
    # ---------------------------------------------------------------- #
    st.markdown("### 🌐 Languages & Environment")
    l1, l2 = st.columns([1, 1])

    with l1:
        st.markdown("**Mother tongue (top 10)**")
        if "motherTongue" in df.columns:
            mt = df["motherTongue"].dropna().replace("", pd.NA).dropna().value_counts().head(10).reset_index()
            mt.columns = ["Language", "Count"]
            fig = px.bar(mt, x="Count", y="Language", orientation="h", text="Count",
                         color="Count", color_continuous_scale="Tealgrn")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with l2:
        st.markdown("**Screen time at home (hours/day)**")
        if "screenTimeHome" in df.columns:
            sth = df["screenTimeHome"].fillna("Not specified").replace("", "Not specified").value_counts().reset_index()
            sth.columns = ["Hours", "Count"]
            # Try a sensible ordering: numbers first, then text
            def sort_key(v):
                try:
                    return (0, float(v))
                except (ValueError, TypeError):
                    return (1, str(v))
            sth = sth.sort_values("Hours", key=lambda s: s.map(sort_key))
            fig = px.bar(sth, x="Hours", y="Count", text="Count",
                         color="Count", color_continuous_scale="Sunsetdark")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------- #
    # Section 5: Activity over time                                    #
    # ---------------------------------------------------------------- #
    st.markdown("### 📅 Registration activity")
    if "createdAt" in df.columns and df["createdAt"].notna().any():
        ts = (
            df.dropna(subset=["createdAt"])
            .assign(day=lambda d: d["createdAt"].dt.date)
            .groupby("day")
            .size()
            .reset_index(name="count")
        )
        ts["cumulative"] = ts["count"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=ts["day"], y=ts["count"], name="Daily", marker_color="#6366F1"))
        fig.add_trace(go.Scatter(
            x=ts["day"], y=ts["cumulative"], name="Cumulative", mode="lines+markers",
            line=dict(color="#EF4444", width=2), yaxis="y2",
        ))
        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Daily registrations"),
            yaxis2=dict(title="Cumulative", overlaying="y", side="right", showgrid=False),
            xaxis=dict(title="Date"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No timestamp data yet.")


def page_registrations(reg_df: pd.DataFrame):
    st.title("📋 Registrations")

    if reg_df.empty:
        st.info("No registrations to show.")
        return

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            search = st.text_input("Search (name / VPD ID / school)").strip().lower()
        with c2:
            genders = ["(all)"] + sorted(reg_df["gender"].dropna().unique().tolist()) if "gender" in reg_df else ["(all)"]
            gender_filter = st.selectbox("Gender", genders)
        with c3:
            res_types = ["(all)"] + sorted(reg_df["residentialType"].dropna().unique().tolist()) if "residentialType" in reg_df else ["(all)"]
            res_filter = st.selectbox("Residence", res_types)

    filtered = reg_df.copy()
    if search:
        cols_to_search = [c for c in ["fullName", "_id", "registrationId", "schoolName"] if c in filtered.columns]
        mask = pd.Series(False, index=filtered.index)
        for col in cols_to_search:
            mask |= filtered[col].astype(str).str.lower().str.contains(search, na=False)
        filtered = filtered[mask]
    if gender_filter != "(all)" and "gender" in filtered:
        filtered = filtered[filtered["gender"] == gender_filter]
    if res_filter != "(all)" and "residentialType" in filtered:
        filtered = filtered[filtered["residentialType"] == res_filter]

    display_cols = [
        c for c in [
            "registrationId", "fullName", "gender", "age", "dob",
            "schoolName", "classSection", "motherTongue", "residentialType",
            "bmi", "createdAt",
        ] if c in filtered.columns
    ]

    st.caption(f"Showing {len(filtered)} of {len(reg_df)} registrations")
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv,
        file_name=f"registrations_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


def render_score_card(vpd_id: str, scores_for_date: dict[str, dict]):
    """Render the visual perception score card for one (participant, date)."""
    rows = []
    for d_idx, domain in enumerate(DOMAINS):
        domain_data = scores_for_date.get(domain, {}) or {}
        correct = 0
        total_time = 0.0
        for lvl in range(1, LEVELS_PER_DOMAIN + 1):
            lvl_payload = domain_data.get(f"Level {lvl}")
            if isinstance(lvl_payload, dict):
                if lvl_payload.get("score"):
                    correct += 1
                total_time += float(lvl_payload.get("timestamp") or 0)
        rows.append({
            "Domain": domain,
            "Correct": f"{correct}/10",
            "_correct": correct,
            "Accuracy": correct / LEVELS_PER_DOMAIN,
            "Total Time (ms)": int(total_time),
        })

    df = pd.DataFrame(rows)

    # Per-group + grand totals (Appendix 2 logic)
    group_rows = []
    for group, idxs in DOMAIN_GROUPS.items():
        sub = df.iloc[idxs]
        c = sub["_correct"].sum()
        t = len(idxs) * LEVELS_PER_DOMAIN
        time_ms = sub["Total Time (ms)"].sum()
        group_rows.append({
            "Domain": group,
            "Correct responses": int(c),
            "Total": int(t),
            "Accuracy": c / t,
            "Total Time (ms)": int(time_ms),
        })
    grand = {
        "Domain": "Visual Perception Total",
        "Correct responses": int(sum(r["Correct responses"] for r in group_rows)),
        "Total": int(sum(r["Total"] for r in group_rows)),
        "Total Time (ms)": int(sum(r["Total Time (ms)"] for r in group_rows)),
    }
    grand["Accuracy"] = grand["Correct responses"] / grand["Total"] if grand["Total"] else 0
    group_rows.append(grand)
    group_df = pd.DataFrame(group_rows)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Graphical representation of accuracy scores")
        fig = px.bar(
            df, x="Domain", y="Accuracy",
            text=df["Accuracy"].map(lambda v: f"{v:.0%}"),
            color="Accuracy", color_continuous_scale="RdYlGn", range_color=[0, 1],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Domain scores")
        st.dataframe(
            group_df.assign(Accuracy=group_df["Accuracy"].map(lambda v: f"{v:.1%}")),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Item based scores")
    st.dataframe(
        df.drop(columns=["_correct"]).assign(Accuracy=df["Accuracy"].map(lambda v: f"{v:.0%}")),
        use_container_width=True, hide_index=True,
    )


def page_participant(reg_df: pd.DataFrame, scores_dict: dict[str, dict]):
    st.title("👤 Participant deep-dive")

    if reg_df.empty:
        st.info("No registrations available. Sync from Firestore first.")
        return

    label_col = "fullName" if "fullName" in reg_df.columns else "_id"
    options = reg_df.assign(
        label=lambda d: d["registrationId"].astype(str) + " — " + d.get(label_col, pd.Series([""] * len(d))).astype(str)
    ).sort_values("registrationId")

    selected_label = st.selectbox("Pick a participant", options["label"].tolist())
    selected_row = options[options["label"] == selected_label].iloc[0]
    vpd_id = selected_row["registrationId"]

    # Header card
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VPD ID", vpd_id)
    c2.metric("Age", selected_row.get("age", "—"))
    c3.metric("Gender", selected_row.get("gender", "—"))
    c4.metric("BMI", selected_row.get("bmi", "—"))

    with st.expander("Full registration record"):
        st.json({k: v for k, v in selected_row.items() if k != "label"}, expanded=False)

    st.markdown("---")

    # Pull all sessions for this VPD from local cache
    sessions = scores_dict.get(vpd_id, {})
    if not sessions:
        st.warning("No score sessions found for this participant in the local cache. "
                   "If they've completed assessments, hit '🔄 Sync from Firestore' to refresh.")
        return

    dates = sorted(sessions.keys(), reverse=True)
    chosen_date = st.selectbox("Assessment date", dates)

    render_score_card(vpd_id, sessions[chosen_date])


def page_my_data(reg_df: pd.DataFrame, scores_dict: dict[str, dict]):
    """Restricted view for a non-admin login: only their linked participant(s)."""
    st.title("👤 My Data")

    label_col = "fullName" if "fullName" in reg_df.columns else "_id"
    options = reg_df.assign(
        label=lambda d: d["registrationId"].astype(str) + " — " + d.get(label_col, pd.Series([""] * len(d))).astype(str)
    ).sort_values("registrationId")

    if len(options) > 1:
        selected_label = st.selectbox("Participant", options["label"].tolist())
    else:
        selected_label = options["label"].iloc[0]
        st.caption(f"Participant: {selected_label}")
    selected_row = options[options["label"] == selected_label].iloc[0]
    vpd_id = selected_row["registrationId"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VPD ID", vpd_id)
    c2.metric("Age", selected_row.get("age", "—"))
    c3.metric("Gender", selected_row.get("gender", "—"))
    c4.metric("BMI", selected_row.get("bmi", "—"))

    st.markdown("---")

    sessions = scores_dict.get(vpd_id, {})
    if not sessions:
        st.info("No assessment sessions recorded yet for this participant.")
        return

    dates = sorted(sessions.keys(), reverse=True)
    chosen_date = st.selectbox("Assessment date", dates)
    render_score_card(vpd_id, sessions[chosen_date])


def page_analytics(scores_long: pd.DataFrame):
    st.title("📈 Cohort analytics")

    if scores_long.empty:
        st.info("No score data yet. Once participants complete assessments, charts will appear here.")
        return

    domain_summary = compute_domain_summary(scores_long)
    group_summary = compute_grouped_totals(domain_summary)

    st.subheader("Mean accuracy by domain (across all sessions)")
    by_domain = (
        domain_summary.groupby("domain")
        .agg(mean_accuracy=("accuracy", "mean"),
             mean_time_ms=("total_time_ms", "mean"),
             n_sessions=("vpd_id", "count"))
        .reindex(DOMAINS)
        .reset_index()
    )

    fig = px.bar(
        by_domain, x="domain", y="mean_accuracy",
        text=by_domain["mean_accuracy"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—"),
        color="mean_accuracy", color_continuous_scale="RdYlGn", range_color=[0, 1],
        hover_data={"n_sessions": True, "mean_time_ms": ":.0f"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, yaxis_tickformat=".0%", xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Mean response time by domain (ms)")
        fig = px.bar(by_domain, x="domain", y="mean_time_ms")
        fig.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Skill group accuracy (per session)")
        fig = px.box(group_summary, x="group", y="accuracy", points="all")
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Per-level accuracy heatmap (cohort)")
    heat = (
        scores_long
        .assign(correct_int=lambda d: d["correct"].fillna(False).astype(int))
        .groupby(["domain", "level"])["correct_int"]
        .mean()
        .reset_index()
        .pivot(index="domain", columns="level", values="correct_int")
        .reindex(DOMAINS)
    )
    fig = go.Figure(data=go.Heatmap(
        z=heat.values, x=heat.columns, y=heat.index,
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar=dict(title="Accuracy", tickformat=".0%"),
        hovertemplate="Domain: %{y}<br>Level: %{x}<br>Accuracy: %{z:.0%}<extra></extra>",
    ))
    fig.update_layout(height=480, xaxis_title="Level", yaxis_title="Domain")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw long-format data")
    st.dataframe(scores_long, use_container_width=True, hide_index=True)
    csv = scores_long.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download all scores (CSV)",
        data=csv,
        file_name=f"all_scores_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _format_relative_time(iso_str: str) -> str:
    """'2 minutes ago', '3 hours ago', etc."""
    try:
        ts = datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return "unknown"
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = now - ts
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def admin_dashboard():
    """Full, unrestricted dashboard — only reached by emails in ADMIN_EMAILS."""
    # --- Connection (only needed for sync) ----------------------------- #
    db = sidebar_credentials()

    # --- Sync UI: always visible ---------------------------------------- #
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Local data cache")

    meta = get_sync_metadata()
    if meta:
        synced_at_iso = meta.get("synced_at", "")
        st.sidebar.caption(
            f"**Last synced:** {_format_relative_time(synced_at_iso)}  \n"
            f"📋 {meta.get('registrations_count', 0)} registrations  \n"
            f"🎯 {meta.get('scores_vpd_count', 0)} participants with scores  \n"
            f"📄 {meta.get('score_docs_count', 0)} score documents"
        )
    else:
        st.sidebar.warning("No local data yet. Click below to fetch from Firestore.")

    sync_disabled = db is None
    sync_label = "🔄 Sync from Firestore" if meta else "⬇️ Initial Sync from Firestore"

    if st.sidebar.button(sync_label, disabled=sync_disabled, use_container_width=True):
        with st.spinner("Syncing…"):
            new_meta = sync_from_firestore(db)
        st.cache_data.clear()
        st.sidebar.success(
            f"Synced {new_meta['registrations_count']} registrations "
            f"and {new_meta['score_docs_count']} score docs."
        )
        st.rerun()

    if sync_disabled and not meta:
        st.title("🧠 DCVPA-C Analytics Dashboard")
        st.info(
            "Connect to Firestore using the sidebar, then click "
            "**Initial Sync from Firestore** to download data locally. "
            "After that, the dashboard reads from the local JSON cache "
            "and only re-syncs when you ask it to."
        )
        return

    # --- Load from local cache ----------------------------------------- #
    reg_mtime = get_file_mtime(REGISTRATIONS_FILE)
    scores_mtime = get_file_mtime(SCORES_FILE)

    reg_df = load_registrations_local(reg_mtime)
    scores_dict = load_scores_local(scores_mtime)
    scores_long = scores_dict_to_long_df(scores_mtime)

    if reg_df.empty:
        st.title("🧠 DCVPA-C Analytics Dashboard")
        st.warning(
            "Local cache is empty. Use **🔄 Sync from Firestore** in the sidebar to populate it."
        )
        return

    # --- Routing -------------------------------------------------------- #
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Registrations", "Participant deep-dive", "Cohort analytics"],
    )

    # Show data source banner so the user knows where charts come from
    st.sidebar.caption("📂 Charts render from local JSON. Sync to refresh.")

    if page == "Overview":
        page_overview(reg_df, scores_long)
    elif page == "Registrations":
        page_registrations(reg_df)
    elif page == "Participant deep-dive":
        page_participant(reg_df, scores_dict)
    else:
        page_analytics(scores_long)


def restricted_dashboard(allowed_ids: list[str]):
    """Scoped view for a logged-in participant/parent email: only their own data."""
    reg_mtime = get_file_mtime(REGISTRATIONS_FILE)
    scores_mtime = get_file_mtime(SCORES_FILE)

    reg_df = load_registrations_local(reg_mtime)
    scores_dict = load_scores_local(scores_mtime)

    if reg_df.empty or "registrationId" not in reg_df.columns:
        st.title("🧠 DCVPA-C Dashboard")
        st.info("No data available yet. Please check back later.")
        return

    my_reg_df = reg_df[reg_df["registrationId"].isin(allowed_ids)]
    if my_reg_df.empty:
        st.title("🧠 DCVPA-C Dashboard")
        st.warning("No records found matching your account. Contact an administrator.")
        return

    page_my_data(my_reg_df, scores_dict)


def main():
    if "auth_email" not in st.session_state:
        render_login()
        return

    email = st.session_state["auth_email"]
    reg_df = load_registrations_local(get_file_mtime(REGISTRATIONS_FILE))
    role, allowed_ids = resolve_access(email, reg_df)

    st.sidebar.title("DCVPA-C Dashboard")
    st.sidebar.caption(f"Signed in as **{email}**" + (" · admin" if role == "admin" else ""))
    if st.sidebar.button("Sign out"):
        del st.session_state["auth_email"]
        st.rerun()
    st.sidebar.markdown("---")

    if role == "admin":
        admin_dashboard()
    else:
        restricted_dashboard(allowed_ids)


if __name__ == "__main__":
    main()