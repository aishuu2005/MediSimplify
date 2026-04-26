"""
main.py  —  MediSimplify
========================
Entry point for the MediSimplify Streamlit web application.

Run with:  streamlit run main.py

Fixes applied vs the original:
  1. verify_user now returns (bool, dict|str); login handler unpacks the tuple.
  2. create_user returns (bool, str); signup handler already unpacked correctly.
  3. `df` is always initialised before the `if not df.empty:` guard at line ~696
     in the original (it was referenced inside a branch where it might be undefined).
  4. All `use_column_width=True` replaced with `use_container_width=True`.
  5. All st.number_input `value=` arguments use `or 0.0` / `or 0` guards so
     that a None from session_state never causes a TypeError.
  6. Triage logic for HbA1c explicitly flags ≥ 6.5 as DIABETIC (red) and
     5.7–6.4 as PRE_DIABETIC (orange); the triage card colour and the Quick
     Analyser both follow these rules via the medical_engine helpers.
  7. Abnormal triage cards now set the icon AND colour correctly for every
     status (DIABETIC, PRE_DIABETIC, HIGH_CHOLESTEROL, HIGH, LOW).
"""

import datetime
import io
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Project modules ──────────────────────────────────────────────────────────
from database import (
    init_db, create_user, verify_user,
    create_profile, get_profiles, delete_profile,
    add_health_record, get_records_for_profile,
    delete_health_record, get_unique_test_names,
)
from medical_engine import (
    analyse_result, get_advice, get_status_colour,
    get_test_info, translate_jargon, get_all_known_tests,
)
from ocr_engine import process_lab_image, extract_numbers

# ── FPDF (graceful degradation) ───────────────────────────────────────────────
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 – PAGE CONFIG & GLOBAL STYLES
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediSimplify",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #a8d8ea !important; }

    [data-testid="metric-container"] {
        background: #f0f4ff; border: 1px solid #dde3ff;
        border-radius: 12px; padding: 16px;
    }

    /* ── Status badges ───────────────────────────────────── */
    .badge-normal      { background:#d4edda; color:#155724; padding:4px 12px;
                         border-radius:20px; font-weight:600; font-size:0.85em; }
    .badge-high        { background:#ffe4cc; color:#7d3c00; padding:4px 12px;
                         border-radius:20px; font-weight:600; font-size:0.85em; }
    .badge-low         { background:#fce4ec; color:#880e4f; padding:4px 12px;
                         border-radius:20px; font-weight:600; font-size:0.85em; }
    .badge-prediabetic { background:#fff3cd; color:#856404; padding:4px 12px;
                         border-radius:20px; font-weight:600; font-size:0.85em; }
    /* DIABETIC badge – always red */
    .badge-diabetic    { background:#f8d7da; color:#721c24; padding:4px 12px;
                         border-radius:20px; font-weight:700; font-size:0.85em; }
    .badge-unknown     { background:#e0e0e0; color:#424242; padding:4px 12px;
                         border-radius:20px; font-weight:600; font-size:0.85em; }

    .page-title    { font-size:2rem; font-weight:700; color:#1a237e; margin-bottom:.5rem; }
    .page-subtitle { color:#546e7a; margin-bottom:1.5rem; }

    .advice-box        { background:#e8f5e9; border-left:5px solid #27ae60;
                         padding:1rem; border-radius:8px; margin:.5rem 0; }
    .advice-box-warn   { background:#fff8e1; border-left:5px solid #f39c12;
                         padding:1rem; border-radius:8px; margin:.5rem 0; }
    .advice-box-danger { background:#fdecea; border-left:5px solid #e74c3c;
                         padding:1rem; border-radius:8px; margin:.5rem 0; }

    hr { border:none; border-top:1px solid #e0e0e0; margin:1.5rem 0; }

    .stButton > button {
        border-radius:8px; font-weight:600; transition:all 0.2s;
    }
    .stButton > button:hover {
        transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,.15);
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 – SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    defaults = {
        "logged_in":           False,
        "user_id":             None,
        "username":            "",
        "active_profile_id":   None,
        "active_profile_name": "",
        "current_page":        "add_records",
        "auth_mode":           "login",
        "ocr_results":         [],
        "ocr_raw_text":        "",
        "ocr_prefill":         {},
        # OCR bridge / prefill — always numeric defaults to avoid TypeError
        "prefill_hba1c":       0.0,
        "prefill_glucose":     0.0,
        "prefill_cholesterol": 0.0,
        "selected_lang":       "english",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 – AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════

def render_auth_screen() -> None:
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0 1rem 0;">
            <div style="font-size:4rem;">🏥</div>
            <h1 style="color:#1a237e; font-size:2.5rem; margin:0; font-weight:800;">
                MediSimplify
            </h1>
            <p style="color:#546e7a; font-size:1.1rem; margin-top:.25rem;">
                Your Family's Medical Records, Simplified.
            </p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑  Login", "📝  Create Account"])

        # ── LOGIN ─────────────────────────────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password",
                                         placeholder="Enter your password")
                submit   = st.form_submit_button("Login", use_container_width=True,
                                                  type="primary")

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    # verify_user returns (bool, dict|str)
                    ok, result = verify_user(username, password)
                    if ok:
                        st.session_state["logged_in"]  = True
                        st.session_state["user_id"]    = result["id"]
                        st.session_state["username"]   = result["username"]
                        st.success(f"Welcome back, {result['username']}! 👋")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")

        # ── SIGNUP ────────────────────────────────────────────────────────────
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup_form", clear_on_submit=False):
                new_username = st.text_input("Choose a Username",
                                             placeholder="e.g. john_doe")
                new_password = st.text_input("Choose a Password", type="password",
                                             placeholder="At least 6 characters")
                confirm_pw   = st.text_input("Confirm Password", type="password")
                signup_btn   = st.form_submit_button("Create Account",
                                                      use_container_width=True,
                                                      type="primary")

            if signup_btn:
                new_username = new_username.strip()
                new_password = new_password.strip()

                if not new_username or not new_password:
                    st.error("All fields are required.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_password != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    # create_user returns (bool, str)
                    ok, msg = create_user(new_username, new_password)
                    if ok:
                        st.success("✅ Account created! Please log in.")
                    else:
                        st.error(f"❌ {msg}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 – SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2.5rem;">🏥</div>
            <h2 style="margin:0; font-size:1.4rem; font-weight:700;">MediSimplify</h2>
            <p style="font-size:.8rem; opacity:.7; margin-top:.2rem;">Family Health Hub</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"👤 **{st.session_state['username']}**")
        st.markdown("---")

        # ── Profile selector ──────────────────────────────────────────────────
        st.markdown("### 👨‍👩‍👧 Family Profiles")
        profiles = get_profiles(st.session_state["user_id"])

        if profiles:
            # database.get_profiles guarantees the "patient_name" alias exists
            profile_names = [p["patient_name"] for p in profiles]
            profile_ids   = [p["id"]           for p in profiles]

            current_idx = 0
            if st.session_state["active_profile_id"] in profile_ids:
                current_idx = profile_ids.index(st.session_state["active_profile_id"])

            selected_name = st.selectbox(
                "Active Profile",
                options=profile_names,
                index=current_idx,
                key="profile_selector",
            )

            selected_idx = profile_names.index(selected_name)
            st.session_state["active_profile_id"]   = profile_ids[selected_idx]
            st.session_state["active_profile_name"] = selected_name

            active_p = profiles[selected_idx]
            st.caption(
                f"🎂 Age: {active_p.get('age', 'N/A')}  |  "
                f"⚧ {active_p.get('gender', 'N/A')}"
            )
        else:
            st.info("No profiles yet. Create one below! ⬇️")
            st.session_state["active_profile_id"]   = None
            st.session_state["active_profile_name"] = ""

        # ── Add profile ────────────────────────────────────────────────────────
        with st.expander("➕ Add New Profile"):
            with st.form("add_profile_form", clear_on_submit=False):
                p_name   = st.text_input("Patient Name*", placeholder="e.g. Mother")
                p_age    = st.number_input("Age", min_value=0, max_value=120, value=30)
                p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                p_submit = st.form_submit_button("Add Profile", use_container_width=True)

            if p_submit:
                if not p_name.strip():
                    st.error("Patient name is required.")
                else:
                    new_id = create_profile(
                        st.session_state["user_id"], p_name, p_age, p_gender
                    )
                    if new_id:
                        st.success(f"✅ Profile '{p_name}' added!")
                        st.session_state["active_profile_id"]   = new_id
                        st.session_state["active_profile_name"] = p_name
                        st.rerun()
                    else:
                        st.error("Failed to create profile.")

        st.markdown("---")

        # ── Navigation ─────────────────────────────────────────────────────────
        st.markdown("### 🧭 Navigation")
        nav_items = [
            ("add_records", "📋 Add Records"),
            ("dashboard",   "📊 Analytics Dashboard"),
            ("triage",      "🔬 Medical Triage"),
            ("dictionary",  "📖 Medical Dictionary"),
            ("pdf_export",  "📄 Export PDF Report"),
        ]
        for page_key, page_label in nav_items:
            is_active = st.session_state["current_page"] == page_key
            if st.button(page_label, key=f"nav_{page_key}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["current_page"] = page_key
                st.rerun()

        st.markdown("---")

        # ── Language ───────────────────────────────────────────────────────────
        st.markdown("### 🌐 Language")
        languages = {
            "English":            "english",
            "हिंदी (Hindi)":      "hindi",
            "தமிழ் (Tamil)":      "tamil",
            "తెలుగు (Telugu)":    "telugu",
            "ಕನ್ನಡ (Kannada)":    "kannada",
            "മലയാളം (Malayalam)": "malayalam",
        }
        selected_lang_display = st.selectbox(
            "Medical Advice Language",
            options=list(languages.keys()),
            index=list(languages.values()).index(st.session_state["selected_lang"]),
            key="lang_selector",
        )
        st.session_state["selected_lang"] = languages[selected_lang_display]

        st.markdown("---")

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 – ADD RECORDS PAGE
# ════════════════════════════════════════════════════════════════════════════

def render_add_records_page() -> None:
    st.markdown('<p class="page-title">📋 Add Lab Records</p>', unsafe_allow_html=True)

    profile_id = st.session_state.get("active_profile_id")

    # Fetch records and build df BEFORE any conditional return so that
    # `df` is always defined in this function's scope.
    records: list[dict] = get_records_for_profile(profile_id) if profile_id else []
    df = pd.DataFrame(records) if records else pd.DataFrame()

    if not profile_id:
        st.warning("⚠️ Please create and select a patient profile first using the sidebar.")
        return

    st.markdown(
        f'<p class="page-subtitle">Adding records for: '
        f'<strong>{st.session_state["active_profile_name"]}</strong></p>',
        unsafe_allow_html=True,
    )

    tab_manual, tab_ocr = st.tabs(["✍️ Manual Entry", "📷 OCR Upload"])

    # ── TAB 1: MANUAL ENTRY ──────────────────────────────────────────────────
    with tab_manual:
        st.markdown("#### Enter Lab Test Details")

        # OCR auto-fill strip
        if st.session_state.get("ocr_raw_text") and st.session_state.get("ocr_results"):
            st.markdown("##### ✨ OCR Auto-Fill")
            col_auto, col_btn = st.columns([3, 1])
            with col_auto:
                st.caption("Values extracted from the uploaded OCR image:")
                extracted_preview = extract_numbers(st.session_state["ocr_raw_text"])
                st.write(f"**HbA1c:** {extracted_preview.get('hba1c', 'Not found')}%")
                st.write(f"**Glucose:** {extracted_preview.get('glucose', 'Not found')} mg/dL")
                st.write(f"**Cholesterol:** {extracted_preview.get('cholesterol', 'Not found')} mg/dL")
            with col_btn:
                if st.button("✨ Auto-Fill from OCR", type="secondary",
                             use_container_width=True):
                    ex = extract_numbers(st.session_state["ocr_raw_text"])
                    # Guard: always store a numeric default (never None)
                    st.session_state["prefill_hba1c"]       = float(ex.get("hba1c") or 0.0)
                    st.session_state["prefill_glucose"]     = float(ex.get("glucose") or 0.0)
                    st.session_state["prefill_cholesterol"] = float(ex.get("cholesterol") or 0.0)
                    st.success("Values ready to use!")
                    st.rerun()

        # Quick entry for the three most common diabetes/lipid tests
        st.markdown("##### Quick Entry for Common Tests")
        col1, col2, col3 = st.columns(3)

        with col1:
            # FIX: value= always a float — use `or 0.0` guard
            hba1c_value = st.number_input(
                "HbA1c (%)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state.get("prefill_hba1c") or 0.0),
                step=0.1, format="%.1f",
                key="hba1c_input",
            )
            if st.button("Save HbA1c", key="save_hba1c"):
                if hba1c_value > 0:
                    ok = add_health_record(
                        profile_id=profile_id,
                        test_name="HBA1C",
                        test_value=hba1c_value,
                        unit="%",
                        date_recorded=str(datetime.date.today()),
                    )
                    if ok:
                        st.success(f"✅ HbA1c {hba1c_value}% saved!")
                        st.session_state["prefill_hba1c"] = 0.0
                        st.rerun()
                    else:
                        st.error("❌ Failed to save.")
                else:
                    st.error("Please enter a valid HbA1c value.")

        with col2:
            # FIX: value= always float (number_input works with either int or float)
            glucose_value = st.number_input(
                "Fasting Glucose (mg/dL)",
                min_value=0.0, max_value=500.0,
                value=float(st.session_state.get("prefill_glucose") or 0.0),
                step=1.0,
                key="glucose_input",
            )
            if st.button("Save Glucose", key="save_glucose"):
                if glucose_value > 0:
                    ok = add_health_record(
                        profile_id=profile_id,
                        test_name="FASTING BLOOD SUGAR",
                        test_value=glucose_value,
                        unit="mg/dL",
                        date_recorded=str(datetime.date.today()),
                    )
                    if ok:
                        st.success(f"✅ Glucose {glucose_value} mg/dL saved!")
                        st.session_state["prefill_glucose"] = 0.0
                        st.rerun()
                    else:
                        st.error("❌ Failed to save.")
                else:
                    st.error("Please enter a valid glucose value.")

        with col3:
            cholesterol_value = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=0.0, max_value=500.0,
                value=float(st.session_state.get("prefill_cholesterol") or 0.0),
                step=1.0,
                key="cholesterol_input",
            )
            if st.button("Save Cholesterol", key="save_cholesterol"):
                if cholesterol_value > 0:
                    ok = add_health_record(
                        profile_id=profile_id,
                        test_name="TOTAL CHOLESTEROL",
                        test_value=cholesterol_value,
                        unit="mg/dL",
                        date_recorded=str(datetime.date.today()),
                    )
                    if ok:
                        st.success(f"✅ Cholesterol {cholesterol_value} mg/dL saved!")
                        st.session_state["prefill_cholesterol"] = 0.0
                        st.rerun()
                    else:
                        st.error("❌ Failed to save.")
                else:
                    st.error("Please enter a valid cholesterol value.")

        st.markdown("---")
        st.markdown("##### Or Enter Any Other Test Manually")

        known_tests = ["-- Type custom --"] + get_all_known_tests()
        ocr_prefill = st.session_state.get("ocr_prefill", {})
        ocr_candidates = list(ocr_prefill.keys()) if isinstance(ocr_prefill, dict) else []
        default_prefill: dict = {}

        if ocr_candidates:
            selected_prefill = st.selectbox(
                "Detected OCR candidates",
                options=ocr_candidates,
                index=0,
                help="Choose which OCR-detected value to pre-fill.",
                key="ocr_prefill_choice",
            )
            default_prefill = ocr_prefill.get(selected_prefill, {})
            st.caption("Edit values if needed before saving.")

        with st.form("manual_entry_form", clear_on_submit=False):
            col1, col2 = st.columns(2)

            with col1:
                prefill_name   = default_prefill.get("test_name", "")
                selected_index = known_tests.index(prefill_name) if prefill_name in known_tests else 0
                test_pick = st.selectbox(
                    "Select Test (from dictionary)",
                    options=known_tests,
                    index=selected_index,
                )
                custom_test = st.text_input(
                    "Or type custom Test Name",
                    value="" if selected_index != 0 else prefill_name,
                    placeholder="e.g. Vitamin C",
                )

            with col2:
                # FIX: default is always float, never None
                test_value = st.number_input(
                    "Test Value*",
                    min_value=0.0,
                    value=float(default_prefill.get("test_value") or 0.0),
                    step=0.1,
                    format="%.2f",
                )
                test_unit = st.text_input(
                    "Unit (optional)",
                    value=default_prefill.get("unit", ""),
                    placeholder="e.g. mg/dL, g/dL, %",
                )

            date_recorded = st.date_input(
                "Date of Test*",
                value=datetime.date.today(),
                max_value=datetime.date.today(),
            )
            submit_manual = st.form_submit_button(
                "💾 Save Record", use_container_width=True, type="primary"
            )

        if submit_manual:
            if test_pick != "-- Type custom --":
                final_test_name = test_pick.title()
            elif custom_test.strip():
                final_test_name = custom_test.strip().title()
            else:
                st.error("Please select or type a test name.")
                return

            ok = add_health_record(
                profile_id=profile_id,
                test_name=final_test_name,
                test_value=test_value,
                unit=test_unit.strip(),
                date_recorded=str(date_recorded),
            )
            if ok:
                st.success(f"✅ Saved: **{final_test_name}** = {test_value} {test_unit}")
                st.rerun()
            else:
                st.error("❌ Failed to save record. Please try again.")

        # ── Saved records summary ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 📋 Saved Records")

        # FIX: df is always defined above; this guard is now safe
        if not df.empty:
            display_df = df[["test_name", "test_value", "unit", "date_recorded"]].copy()
            display_df.columns = ["Test Name", "Value", "Unit", "Date"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("🗑️ Delete a Record"):
                record_options = {
                    f"{r['test_name']} | {r['test_value']} {r['unit']} | {r['date_recorded']}": r["id"]
                    for r in records
                }
                to_delete = st.selectbox("Select record to delete", list(record_options.keys()))
                if st.button("Delete Selected Record", type="secondary"):
                    delete_health_record(record_options[to_delete])
                    st.success("Record deleted.")
                    st.rerun()
        else:
            st.info("No saved records yet. Use the form or OCR scan above to add your first record.")

    # ── TAB 2: OCR UPLOAD ────────────────────────────────────────────────────
    with tab_ocr:
        st.markdown("#### Upload a Lab Report Image")
        st.info(
            "📸 Upload a clear photo or scan of your lab report. "
            "The app will extract test names and values automatically. "
            "Review and confirm before saving."
        )

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="ocr_uploader",
        )

        if uploaded_file is not None:
            col_img, col_results = st.columns([1, 1])

            with col_img:
                # FIX: use_container_width replaces deprecated use_column_width
                st.image(uploaded_file, caption="Uploaded Report",
                         use_container_width=True)

            with col_results:
                if st.button("🔍 Extract Data via OCR", type="primary"):
                    with st.spinner("Running OCR… Please wait."):
                        image_bytes = uploaded_file.getvalue()
                        raw_text, parsed = process_lab_image(image_bytes)

                        # ── Fallback: nothing could be extracted ─────────────
                        if not parsed:
                            st.warning(
                                "⚠️ Could not parse values, please enter manually. "
                                "Try a clearer, well-lit scan or photo of the report."
                            )
                            st.session_state["ocr_raw_text"]        = raw_text
                            st.session_state["ocr_results"]         = []
                            st.session_state["ocr_prefill"]         = {}
                            st.session_state["prefill_hba1c"]       = 0.0
                            st.session_state["prefill_glucose"]     = 0.0
                            st.session_state["prefill_cholesterol"] = 0.0
                        else:
                            extracted = extract_numbers(raw_text)

                            # ocr_prefill is a dict keyed by test_name for the
                            # "Detected OCR candidates" selectbox in the Manual tab.
                            ocr_prefill_map: dict = {
                                r["test_name"]: r for r in parsed
                            }

                            st.session_state["ocr_raw_text"]        = raw_text
                            st.session_state["ocr_results"]         = parsed
                            st.session_state["ocr_prefill"]         = ocr_prefill_map
                            # Bridge → quick-entry inputs (always float, never None)
                            st.session_state["prefill_hba1c"]       = float(extracted.get("hba1c") or 0.0)
                            st.session_state["prefill_glucose"]     = float(extracted.get("glucose") or 0.0)
                            st.session_state["prefill_cholesterol"] = float(extracted.get("cholesterol") or 0.0)
                            st.success(f"✅ Extracted {len(parsed)} test result(s). Review below.")
                        st.rerun()

        # Review extracted rows
        if st.session_state.get("ocr_results"):
            st.markdown("---")
            st.markdown("#### 🔎 Extracted Results – Review & Save")
            st.caption("Uncheck any rows you do NOT want to save.")

            parsed     = st.session_state["ocr_results"]
            save_flags = []

            for i, rec in enumerate(parsed):
                col_chk, col_name, col_val, col_unit = st.columns([0.5, 2, 1, 1])
                with col_chk:
                    flag = st.checkbox("", value=True, key=f"ocr_chk_{i}")
                with col_name:
                    rec["test_name"] = st.text_input(
                        "Test Name", value=rec["test_name"],
                        key=f"ocr_name_{i}", label_visibility="collapsed",
                    )
                with col_val:
                    # FIX: value= always float
                    rec["test_value"] = st.number_input(
                        "Value",
                        value=float(rec.get("test_value") or 0.0),
                        step=0.1,
                        key=f"ocr_val_{i}",
                        label_visibility="collapsed",
                    )
                with col_unit:
                    rec["unit"] = st.text_input(
                        "Unit", value=rec.get("unit", ""),
                        key=f"ocr_unit_{i}", label_visibility="collapsed",
                    )
                save_flags.append(flag)

            ocr_date = st.date_input(
                "Date of Test for all OCR records",
                value=datetime.date.today(),
                max_value=datetime.date.today(),
                key="ocr_date_picker",
            )

            if st.button("💾 Save All Selected Records", type="primary"):
                saved_count = 0
                for i, rec in enumerate(parsed):
                    if save_flags[i]:
                        ok = add_health_record(
                            profile_id=profile_id,
                            test_name=rec["test_name"],
                            test_value=rec["test_value"],
                            unit=rec.get("unit", ""),
                            date_recorded=str(ocr_date),
                        )
                        if ok:
                            saved_count += 1
                st.success(f"✅ {saved_count} record(s) saved successfully!")
                st.session_state["ocr_results"]  = []
                st.session_state["ocr_raw_text"] = ""
                st.session_state["ocr_prefill"]  = {}
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 – ANALYTICS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def render_dashboard_page() -> None:
    st.markdown('<p class="page-title">📊 Analytics Dashboard</p>', unsafe_allow_html=True)

    profile_id   = st.session_state["active_profile_id"]
    profile_name = st.session_state["active_profile_name"]

    if not profile_id:
        st.warning("Please select a patient profile from the sidebar.")
        return

    st.markdown(
        f'<p class="page-subtitle">Health Trends for: <strong>{profile_name}</strong></p>',
        unsafe_allow_html=True,
    )

    records = get_records_for_profile(profile_id)
    if not records:
        st.info("No records found. Go to **Add Records** to start tracking.")
        return

    df = pd.DataFrame(records)
    df["date_recorded"] = pd.to_datetime(df["date_recorded"])
    df = df.sort_values("date_recorded")

    # ── Latest value cards ────────────────────────────────────────────────────
    st.markdown("### 📌 Latest Values")
    latest = df.sort_values("date_recorded").groupby("test_name").last().reset_index()

    badge_map = {
        "NORMAL":           "badge-normal",
        "HIGH":             "badge-high",
        "LOW":              "badge-low",
        "PRE_DIABETIC":     "badge-prediabetic",
        "DIABETIC":         "badge-diabetic",    # always red
        "HIGH_CHOLESTEROL": "badge-high",
        "UNKNOWN":          "badge-unknown",
    }

    cols = st.columns(4)
    for i, (_, row) in enumerate(latest.iterrows()):
        status      = analyse_result(row["test_name"], row["test_value"])
        badge_class = badge_map.get(status, "badge-unknown")
        with cols[i % 4]:
            st.metric(
                label=row["test_name"],
                value=f"{row['test_value']} {row.get('unit', '')}",
            )
            st.markdown(f'<span class="{badge_class}">{status}</span>',
                        unsafe_allow_html=True)
            st.caption(f"📅 {row['date_recorded'].strftime('%d %b %Y')}")

    st.markdown("---")

    # ── Trend chart ────────────────────────────────────────────────────────────
    st.markdown("### 📈 Test Trend Over Time")
    test_names = get_unique_test_names(profile_id)

    if not test_names:
        st.info("Not enough data for a trend chart.")
        return

    selected_test = st.selectbox("Select a Test to Plot", options=test_names,
                                 key="trend_test_selector")

    test_df   = df[df["test_name"] == selected_test].sort_values("date_recorded")
    test_info = get_test_info(selected_test)
    unit_label = test_df["unit"].iloc[-1] if (not test_df.empty and "unit" in test_df.columns) else ""

    fig = px.line(
        test_df,
        x="date_recorded", y="test_value",
        markers=True,
        title=f"{selected_test} Trend ({unit_label})",
        labels={"date_recorded": "Date", "test_value": f"Value ({unit_label})"},
        color_discrete_sequence=["#1a237e"],
    )

    if test_info["range"] != "N/A":
        try:
            if "HBA1C" not in selected_test.upper() and "FASTING BLOOD SUGAR" not in selected_test.upper():
                low_str, high_str = test_info["range"].split("–")
                low_val  = float(low_str.strip().replace("< ", ""))
                high_val = float(high_str.split()[0].strip())
                fig.add_hrect(
                    y0=low_val, y1=high_val,
                    fillcolor="green", opacity=0.08, line_width=0,
                    annotation_text="Normal Range", annotation_position="top left",
                )
        except Exception:
            pass

    fig.update_traces(marker=dict(size=10, color="#1a237e"), line=dict(width=3))
    fig.update_layout(
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        font=dict(family="Inter"), title_font=dict(size=18, color="#1a237e"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View All Records Table"):
        display = df[["test_name", "test_value", "unit", "date_recorded"]].copy()
        display.columns = ["Test Name", "Value", "Unit", "Date"]
        display["Date"] = display["Date"].dt.strftime("%d %b %Y")
        st.dataframe(display, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 – MEDICAL TRIAGE
# ════════════════════════════════════════════════════════════════════════════

# Map every possible status to (icon, CSS box class, hex colour)
_TRIAGE_STYLE: dict[str, tuple[str, str, str]] = {
    "DIABETIC":         ("🔴", "advice-box-danger", "#c0392b"),
    "HIGH":             ("🔴", "advice-box-danger", "#c0392b"),
    "LOW":              ("🔴", "advice-box-danger", "#c0392b"),
    "HIGH_CHOLESTEROL": ("🟠", "advice-box-warn",   "#e67e22"),
    "PRE_DIABETIC":     ("🟡", "advice-box-warn",   "#d35400"),
}
_TRIAGE_STYLE_DEFAULT = ("🔵", "advice-box",        "#2980b9")


def render_triage_page() -> None:
    st.markdown('<p class="page-title">🔬 Medical Triage & Analysis</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">AI-assisted triage with plain-English advice.</p>',
        unsafe_allow_html=True,
    )
    st.warning(
        "⚠️ **Disclaimer:** For educational purposes only. "
        "Always consult a qualified medical professional.",
        icon="⚠️",
    )

    profile_id   = st.session_state["active_profile_id"]
    profile_name = st.session_state["active_profile_name"]

    if not profile_id:
        st.warning("Please select a patient profile from the sidebar.")
        return

    records = get_records_for_profile(profile_id)
    if not records:
        st.info("No records to analyse. Add some records first.")
        return

    df = pd.DataFrame(records)
    df["date_recorded"] = pd.to_datetime(df["date_recorded"])
    latest = df.sort_values("date_recorded").groupby("test_name").last().reset_index()

    st.markdown(f"### Analysis for: **{profile_name}**")

    abnormal_records = []
    normal_records   = []

    for _, row in latest.iterrows():
        status    = analyse_result(row["test_name"], row["test_value"])
        test_info = get_test_info(row["test_name"])
        advice    = get_advice(status, st.session_state["selected_lang"])

        result = {
            "status":     status,
            "advice":     advice,
            "label":      test_info["label"],
            "range":      test_info["range"],
            "test_name":  row["test_name"],
            "test_value": row["test_value"],
            "unit":       row.get("unit", ""),
        }

        if status in ("HIGH", "LOW", "PRE_DIABETIC", "DIABETIC", "HIGH_CHOLESTEROL"):
            abnormal_records.append(result)
        else:
            normal_records.append(result)

    # ── Abnormal ────────────────────────────────────────────────────────────
    if abnormal_records:
        st.markdown("#### 🚨 Attention Required")
        for res in abnormal_records:
            status = res["status"]
            icon, box_class, text_colour = _TRIAGE_STYLE.get(status, _TRIAGE_STYLE_DEFAULT)

            # HbA1c 9.2 → analyse_result returns "DIABETIC" → red card below
            st.markdown(f"""
            <div class="{box_class}">
                <strong>{icon} {res['test_name']}</strong>
                &nbsp;&nbsp;→&nbsp;&nbsp;
                <code>{res['test_value']} {res['unit']}</code>
                &nbsp;&nbsp;
                <span style="font-weight:700; color:{text_colour};">{status}</span>
                <br><br>{res['advice']}
                <br><small style="opacity:.7">Normal range: {res['range']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 All analysed values are within normal ranges!")

    # ── Normal ──────────────────────────────────────────────────────────────
    if normal_records:
        with st.expander(f"✅ Normal Results ({len(normal_records)})"):
            for res in normal_records:
                st.markdown(f"""
                <div class="advice-box">
                    <strong>✅ {res['test_name']}</strong>
                    &nbsp;→&nbsp;
                    <code>{res['test_value']} {res['unit']}</code>
                    &nbsp;&nbsp;NORMAL &nbsp;|&nbsp;
                    Range: {res['range']}
                </div>
                """, unsafe_allow_html=True)

    # ── Quick Analyser ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Quick Test Analyser")
    st.caption(
        "Type any HbA1c ≥ 6.5 → DIABETIC (red) | 5.7–6.4 → PRE_DIABETIC (orange) | "
        "< 5.7 → NORMAL (green)"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        qa_test = st.selectbox("Test Name", get_all_known_tests(), key="qa_test")
    with col2:
        # FIX: value= always 0.0 (float), never None
        qa_value = st.number_input("Value", min_value=0.0, value=0.0, step=0.1,
                                   key="qa_value")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        qa_btn = st.button("Analyse", type="primary")

    if qa_btn:
        status    = analyse_result(qa_test, qa_value)
        colour    = get_status_colour(status)
        test_info = get_test_info(qa_test)
        advice    = get_advice(status, st.session_state["selected_lang"])
        st.markdown(f"""
        <div style="background:{colour}22; border-left:5px solid {colour};
                    padding:1rem; border-radius:8px; margin-top:1rem;">
            <strong style="color:{colour}; font-size:1.1rem;">{status}</strong>
            &nbsp;|&nbsp; {test_info['label']} = {qa_value}
            <br><br>{advice}
            <br><small style="opacity:.7">Reference range: {test_info['range']}</small>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 – MEDICAL DICTIONARY
# ════════════════════════════════════════════════════════════════════════════

def render_dictionary_page() -> None:
    st.markdown('<p class="page-title">📖 Medical Dictionary</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Understand complex medical terminology in plain English.</p>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input(
        "🔍 Search for a medical term",
        placeholder="e.g. hemoglobin, TSH, leukopenia, creatinine…",
        key="dict_search",
    )

    if search_term:
        explanation = translate_jargon(search_term)
        if "not in" in explanation.lower() or "sorry" in explanation.lower():
            st.warning(f"🤔 {explanation}")
        else:
            st.markdown(f"""
            <div style="background:#e8f5e9; border-left:5px solid #27ae60;
                        padding:1.5rem; border-radius:8px; font-size:1.05rem;">
                <strong style="color:#1b5e20;">📗 {search_term.title()}</strong>
                <br><br>{explanation}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Full Glossary")

    from medical_engine import JARGON_GLOSSARY
    glossary_items = sorted(JARGON_GLOSSARY.items())
    col1, col2 = st.columns(2)
    mid = len(glossary_items) // 2
    for col, items in [(col1, glossary_items[:mid]), (col2, glossary_items[mid:])]:
        with col:
            for term, defn in items:
                with st.expander(f"**{term.title()}**"):
                    st.write(defn)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 – PDF EXPORT
# ════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(profile_name: str, age, gender, records: list) -> bytes:
    """Generate a professional PDF summary; returns raw bytes."""
    from fpdf import FPDF

    def safe_text(text) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_fill_color(26, 35, 126)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, safe_text("MediSimplify"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(10)
    pdf.cell(0, 8, safe_text("Family Health Report – Confidential"), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(140, 28)
    pdf.cell(60, 6,
             safe_text(f"Generated: {datetime.date.today().strftime('%d %B %Y')}"),
             align="R")

    # Patient details
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 50)
    pdf.set_fill_color(240, 244, 255)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe_text("Patient Information"), ln=True, fill=True, border=1)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(10)
    pdf.cell(60, 8, safe_text(f"Patient: {profile_name}"), border=1)
    pdf.cell(60, 8, safe_text(f"Age: {age}"),              border=1)
    pdf.cell(70, 8, safe_text(f"Gender: {gender}"),        ln=True, border=1)
    pdf.ln(5)

    # Table header
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 244, 255)
    pdf.cell(0, 8, safe_text("Lab Test Results"), ln=True, fill=True, border=1)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(26, 35, 126)
    pdf.set_text_color(255, 255, 255)
    for header, width in [("Test Name", 65), ("Value", 25), ("Unit", 20),
                           ("Date", 30), ("Status", 20), ("Ref. Range", 30)]:
        pdf.cell(width, 8, safe_text(header), border=1, fill=True,
                 align="C" if header != "Test Name" else "L")
    pdf.ln()

    # Table rows  —  DIABETIC → light red, PRE_DIABETIC → light yellow, etc.
    pdf.set_text_color(0, 0, 0)
    for rec in records:
        status    = analyse_result(rec["test_name"], rec["test_value"])
        test_info = get_test_info(rec["test_name"])

        fill_colours = {
            "DIABETIC":         (255, 200, 200),   # red   — explicit for HbA1c 9.2
            "HIGH":             (255, 230, 200),   # orange
            "HIGH_CHOLESTEROL": (255, 230, 200),
            "LOW":              (255, 220, 220),
            "PRE_DIABETIC":     (255, 243, 205),   # yellow
            "NORMAL":           (220, 255, 220),   # green
        }
        r, g, b = fill_colours.get(status, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(65, 7, safe_text(rec["test_name"][:30]),           border=1, fill=True)
        pdf.cell(25, 7, safe_text(str(rec["test_value"])),          border=1, fill=True, align="C")
        pdf.cell(20, 7, safe_text((rec.get("unit") or "")[:10]),    border=1, fill=True, align="C")
        pdf.cell(30, 7, safe_text(str(rec["date_recorded"])),       border=1, fill=True, align="C")
        pdf.cell(20, 7, safe_text(status),                          border=1, fill=True, align="C")
        pdf.cell(30, 7, safe_text(test_info["range"][:15]),         border=1, fill=True, align="C")
        pdf.ln()

    pdf.ln(8)

    # Advice summary (abnormal only)
    abnormal = [
        r for r in records
        if analyse_result(r["test_name"], r["test_value"]) in
           ("HIGH", "LOW", "PRE_DIABETIC", "DIABETIC", "HIGH_CHOLESTEROL")
    ]
    if abnormal:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(255, 243, 205)
        pdf.cell(0, 8, safe_text("Clinical Observations"), ln=True, fill=True, border=1)

        for rec in abnormal:
            status = analyse_result(rec["test_name"], rec["test_value"])
            advice = get_advice(status, st.session_state["selected_lang"])
            advice_short = advice[:200] + "…" if len(advice) > 200 else advice

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(26, 35, 126)
            pdf.cell(0, 7, safe_text(f"• {rec['test_name']} ({status})"), ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, safe_text(advice_short))
            pdf.ln(2)

    # Footer
    pdf.set_text_color(130, 130, 130)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_y(-20)
    pdf.multi_cell(
        0, 5,
        safe_text(
            "DISCLAIMER: This report is generated by MediSimplify for informational "
            "purposes only and does not constitute medical advice. Please consult a "
            "qualified healthcare professional for diagnosis and treatment."
        ),
        align="C",
    )

    return pdf.output(dest="S").encode("latin-1")


def render_pdf_export_page() -> None:
    st.markdown('<p class="page-title">📄 Export PDF Report</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Generate a professional, doctor-ready health summary.</p>',
        unsafe_allow_html=True,
    )

    if not FPDF_AVAILABLE:
        st.error("❌ fpdf2 is not installed. Run `pip install fpdf2` to enable this feature.")
        return

    profile_id   = st.session_state["active_profile_id"]
    profile_name = st.session_state["active_profile_name"]

    if not profile_id:
        st.warning("Please select a patient profile from the sidebar.")
        return

    records = get_records_for_profile(profile_id)
    if not records:
        st.info("No records available. Add health records first.")
        return

    profiles       = get_profiles(st.session_state["user_id"])
    active_profile = next((p for p in profiles if p["id"] == profile_id), {})
    age    = active_profile.get("age", "N/A")
    gender = active_profile.get("gender", "N/A")

    st.markdown(f"### Report Preview for: **{profile_name}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient", profile_name)
    c2.metric("Age", age)
    c3.metric("Gender", gender)
    c4.metric("Total Records", len(records))
    st.markdown("---")

    st.markdown("#### Filter Records for Export")
    date_options  = ["All Dates"] + sorted(set(r["date_recorded"] for r in records), reverse=True)
    selected_date = st.selectbox("Select Date (or All)", date_options, key="pdf_date_filter")

    filtered = records if selected_date == "All Dates" else [
        r for r in records if r["date_recorded"] == selected_date
    ]
    st.caption(f"📋 {len(filtered)} record(s) will be included.")

    if st.button("🖨️ Generate & Download PDF Report", type="primary"):
        with st.spinner("Generating PDF…"):
            try:
                pdf_bytes = generate_pdf_report(profile_name, age, gender, filtered)
                filename  = (
                    f"MediSimplify_{profile_name.replace(' ', '_')}_"
                    f"{datetime.date.today().strftime('%Y%m%d')}.pdf"
                )
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("✅ PDF generated! Click above to download.")
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {e}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 – MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    init_db()
    _init_session_state()

    if not st.session_state["logged_in"]:
        render_auth_screen()
    else:
        render_sidebar()
        page = st.session_state["current_page"]

        if   page == "add_records": render_add_records_page()
        elif page == "dashboard":   render_dashboard_page()
        elif page == "triage":      render_triage_page()
        elif page == "dictionary":  render_dictionary_page()
        elif page == "pdf_export":  render_pdf_export_page()
        else:                       render_add_records_page()


if __name__ == "__main__":
    main()