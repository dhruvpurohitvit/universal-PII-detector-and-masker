"""
app.py — Enterprise PII Detector · Streamlit UI  v3.1
======================================================
"""
from __future__ import annotations

import io, os, pathlib, tempfile, time, warnings
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM",     "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",     "error")
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise PII Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0d1117 0%,#0f1923 60%,#0d1117 100%);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1f33,#0d1117);border-right:1px solid #1e2d3d;}
.card{background:rgba(22,27,34,.8);border:1px solid rgba(56,68,77,.6);border-radius:14px;
      padding:1.2rem 1.5rem;margin-bottom:.8rem;backdrop-filter:blur(12px);}
.badge-pii {display:inline-block;background:#3d1515;color:#ff6b6b;border:1px solid #ff6b6b44;
            border-radius:20px;padding:2px 10px;font-size:.76rem;font-weight:600;}
.badge-excl{display:inline-block;background:#3d2e10;color:#ffa657;border:1px solid #ffa65744;
            border-radius:20px;padding:2px 10px;font-size:.76rem;font-weight:600;}
.badge-safe{display:inline-block;background:#0f2d1a;color:#3fb950;border:1px solid #3fb95044;
            border-radius:20px;padding:2px 10px;font-size:.76rem;font-weight:600;}
[data-testid="stMetricValue"]{color:#58a6ff;font-weight:700;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#58a6ff,#bc8cff);border-radius:99px;}
.log-box{background:#0d1117;border:1px solid #1e2d3d;border-radius:10px;padding:.7rem 1rem;
         max-height:180px;overflow-y:auto;font-family:'JetBrains Mono',monospace;
         font-size:.76rem;color:#8b949e;line-height:1.8;}
/* Make tabs larger and more distinct */
button[data-baseweb="tab"] {font-size:1.1rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k, _v in [
    ("results",    None),
    ("df_input",   None),
    ("masked_df",  None),
    ("mask_pw",    ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .5rem;">
      <div style="font-size:2.6rem;">🛡️</div>
      <div style="font-weight:700;font-size:1.05rem;color:#c9d1d9;">PII Detector</div>
      <div style="font-size:.7rem;color:#58a6ff;letter-spacing:.1em;">ENTERPRISE · v16.0.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**⚙️ Detection Settings**")
    max_samples = st.slider("GLiNER samples / column", 5, 50, 20, 5,
        help="Higher = more accurate, slower. 20 is the recommended default.")
    st.divider()
    st.markdown("""<div style="font-size:.72rem;color:#484f58;text-align:center;">
    GLiNER · Presidio · spaCy · Regex<br>AES-256-GCM · PBKDF2-SHA256</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:1rem 0 .5rem;">
<h1 style="margin:0;font-size:2.5rem;font-weight:800;
   background:linear-gradient(100deg,#58a6ff,#bc8cff 60%,#ff7b72);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
  🛡️ Enterprise PII Detector
</h1>
<p style="color:#8b949e;font-size:.95rem;max-width:660px;margin:.5rem auto 0;line-height:1.6;">
Upload any CSV · detect PII with three AI engines · mask with AES-256-GCM · unmask anytime.
</p></div>
""", unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Loading AI engines — first run ~30s…")
def _load_engines():
    from pii_detector.core.engines import build_presidio_engine, load_gliner
    return build_presidio_engine(), load_gliner()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP TABS
# ══════════════════════════════════════════════════════════════════════════════
main_tab_scan, main_tab_unmask = st.tabs(["🔍 Detect & Mask", "🔓 Unmask Data"])

# ─── TAB 1: DETECT & MASK ─────────────────────────────────────────────────────
with main_tab_scan:
    
    uploaded = st.file_uploader("📂 Upload a CSV file to scan for PII", type=["csv"], key="scan_upload")

    if uploaded is None:
        c1, c2, c3, c4 = st.columns(4)
        _cards = [
            ("🔍", "Presidio NLP",  "spaCy large NER — names, emails, phones, IDs"),
            ("🔎", "40+ Regex",     "Aadhaar · PAN · SSN · IBAN · CC · MAC · crypto"),
            ("🤖", "GLiNER AI",     "Zero-shot — catches PII even with random column names"),
            ("🔒", "AES-256-GCM",   "One password to mask · same password to unmask"),
        ]
        for _col, (_icon, _title, _desc) in zip([c1, c2, c3, c4], _cards):
            with _col:
                st.markdown(f"""<div class="card" style="text-align:center;">
                  <div style="font-size:1.7rem;">{_icon}</div>
                  <div style="font-weight:600;color:#c9d1d9;margin:.3rem 0 .2rem;">{_title}</div>
                  <div style="font-size:.8rem;color:#6e7681;line-height:1.5;">{_desc}</div>
                </div>""", unsafe_allow_html=True)
    else:
        df_input = pd.read_csv(uploaded, dtype=str).fillna("")
        st.session_state.df_input = df_input
        st.success(f"✅ Loaded **{len(df_input):,} rows × {len(df_input.columns)} columns** from `{uploaded.name}`")
        with st.expander("🔎 Preview — first 5 rows", expanded=False):
            st.dataframe(df_input.head(), use_container_width=True)

        if st.button("🚀 Start PII Scan", type="primary", use_container_width=True):
            analyzer, gliner_model = _load_engines()
            
            st.session_state.results   = None
            st.session_state.masked_df = None
            st.session_state.mask_pw   = ""

            from main import process_column
            results: List[dict] = []
            n = len(df_input.columns)
            prog    = st.progress(0, text="Starting…")
            log_ph  = st.empty()
            log_lines: List[str] = []

            t0 = time.perf_counter()
            for i, col in enumerate(df_input.columns):
                prog.progress(i/n, text=f"Scanning column **{i+1}/{n}**: `{col}`")
                res = process_column(col, df_input, analyzer, gliner_model, max_samples=max_samples)
                results.append(res)
                action = res.get("Policy_Action","NONE")
                entity = res.get("Final_Entity_Type") or res.get("Primary_Entity") or "Safe"
                icon   = "🔴" if action=="PROTECT" else ("🟡" if action=="DETECTED_EXCLUDED" else "🟢")
                score  = res.get("Evidence_Score",0.0) or 0.0
                log_lines.append(f"{icon} <b>{col}</b> → {entity} ({score:.0%})")
                log_ph.markdown(
                    f"<div class='log-box'>{'<br>'.join(log_lines[-10:])}</div>",
                    unsafe_allow_html=True)

            elapsed = time.perf_counter() - t0
            prog.progress(1.0, text=f"✅ Done in {elapsed:.1f}s")
            log_ph.empty()
            st.session_state.results = results
            st.rerun()

        # Show results if we have them
        if st.session_state.results is not None:
            results = st.session_state.results
            pii_list  = [r for r in results if r.get("Policy_Action") == "PROTECT"]
            excl_list = [r for r in results if r.get("Policy_Action") == "DETECTED_EXCLUDED"]
            safe_list = [r for r in results if r.get("Policy_Action","NONE") == "NONE"]
            pii_cols  = [r["Column_Name"] for r in pii_list]

            st.divider()
            res_tab_sum, res_tab_det, res_tab_time, res_tab_mask = st.tabs([
                "📊 Summary", "🗂️ Detailed", "⏱️ Timing", "🔒 Mask Data"
            ])

            with res_tab_sum:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("📋 Columns",  len(results))
                m2.metric("🔴 PII",     len(pii_list))
                m3.metric("🟡 Excluded",len(excl_list))
                m4.metric("🟢 Safe",    len(safe_list))
                
                st.markdown("#### Column Status")
                GRID = 5
                for chunk in [results[i:i+GRID] for i in range(0,len(results),GRID)]:
                    cols_g = st.columns(len(chunk))
                    for gc, r in zip(cols_g, chunk):
                        action = r.get("Policy_Action","NONE")
                        entity = r.get("Final_Entity_Type") or r.get("Primary_Entity") or "—"
                        score  = r.get("Evidence_Score",0.0) or 0.0
                        cname  = r.get("Column_Name","")
                        if action=="PROTECT":
                            badge,icon = "badge-pii","🔴"
                        elif action=="DETECTED_EXCLUDED":
                            badge,icon = "badge-excl","🟡"
                        else:
                            badge,icon,entity = "badge-safe","🟢","Safe"
                        with gc:
                            st.markdown(f"""<div class="card" style="text-align:center;padding:.9rem;">
                              <div style="font-size:.7rem;color:#6e7681;word-break:break-all;margin-bottom:.3rem;">
                                {cname}</div>
                              <div style="font-size:1.4rem;">{icon}</div>
                              <div class="{badge}" style="margin:.3rem 0;">{entity}</div>
                              <div style="font-size:.7rem;color:#484f58;">{score:.0%}</div>
                            </div>""", unsafe_allow_html=True)
                
                sum_rows = [{"Column": r.get("Column_Name",""),
                             "Status": "PII" if r.get("Policy_Action")=="PROTECT" else ("EXCLUDED" if r.get("Policy_Action")=="DETECTED_EXCLUDED" else "SAFE"),
                             "Entity": r.get("Final_Entity_Type") or "—",
                             "Confidence": f"{(r.get('Evidence_Score') or 0):.0%}"}
                            for r in results]
                sb = io.StringIO(); pd.DataFrame(sum_rows).to_csv(sb, index=False)
                st.download_button("📋 Download Summary CSV", sb.getvalue(), "summary_report.csv", "text/csv")

            with res_tab_det:
                det_rows = []
                for r in results:
                    action = r.get("Policy_Action","NONE")
                    entity = r.get("Final_Entity_Type") or r.get("Primary_Entity") or "—"
                    score  = r.get("Evidence_Score",0.0) or 0.0
                    det_rows.append({
                        "Column":       r.get("Column_Name",""),
                        "Status":       "🔴 PII" if action=="PROTECT" else ("🟡 Excl" if action=="DETECTED_EXCLUDED" else "🟢 Safe"),
                        "Entity":       entity,
                        "Confidence":   f"{score:.1%}",
                        "Presidio":     "✓" if r.get("Presidio_Support") else "—",
                        "Regex":        "✓" if r.get("Regex_Support") else "—",
                        "GLiNER":       f"{r.get('GLiNER_Value_Confirmation',0.0):.0%}" if r.get("GLiNER_Value_Confirmation") else "—",
                        "Validator":    r.get("Validator_Type","—") or "—",
                        "Reason":       (r.get("Decision_Reason","") or "")[:90],
                    })
                st.dataframe(pd.DataFrame(det_rows), use_container_width=True, height=400)
                db = io.StringIO(); pd.DataFrame(results).to_csv(db, index=False)
                st.download_button("📊 Download Full Report", db.getvalue(), "detailed_report.csv", "text/csv")

            with res_tab_time:
                from pii_detector.pipeline.run_report import save_timing_report
                with tempfile.NamedTemporaryFile(suffix=".txt",delete=False,mode="w",encoding="utf-8") as tf:
                    tf_path = tf.name
                timing_txt = save_timing_report(results, tf_path)
                pathlib.Path(tf_path).unlink(missing_ok=True)
                st.code(timing_txt, language=None)
                st.download_button("⏱️ Download Timing", timing_txt, "timing_report.txt", "text/plain")

            with res_tab_mask:
                if not pii_list:
                    st.success("✅ No PII detected — nothing to mask!", icon="🛡️")
                else:
                    st.markdown(f"**{len(pii_list)} PII column(s) will be encrypted:** `{'` · `'.join(pii_cols)}`")
                    st.info("💡 **Security Note:** We use AES-256-GCM. Your password is the *only* way to decrypt this data. Keep it safe.")
                    
                    c_pw1, c_pw2 = st.columns(2)
                    pw1 = c_pw1.text_input("🔑 Set masking password", type="password", key="pw1")
                    pw2 = c_pw2.text_input("🔑 Confirm password", type="password", key="pw2")

                    pw_ok = pw1 and pw1 == pw2
                    if pw1 and pw2 and not pw_ok:
                        st.error("❌ Passwords do not match.")

                    if pw_ok:
                        st.markdown("#### 👁️ Preview (First 5 rows)")
                        from pii_detector.masking.aes_masker import preview_masked
                        df_prev = st.session_state.df_input.head(5).copy()
                        for col in pii_cols:
                            if col in df_prev.columns:
                                df_prev[col] = preview_masked(df_prev[col])
                        st.dataframe(df_prev, use_container_width=True)

                        if st.button("🔒 Encrypt & Mask Data", type="primary", use_container_width=True):
                            from pii_detector.masking.aes_masker import mask_dataframe
                            with st.spinner(f"Encrypting with AES-256-GCM…"):
                                masked = mask_dataframe(st.session_state.df_input, pii_cols, pw1)
                                st.session_state.masked_df = masked
                                st.session_state.mask_pw   = pw1
                            st.success(f"✅ Successfully encrypted {len(pii_cols)} column(s).")
                        
                        if st.session_state.masked_df is not None and st.session_state.mask_pw == pw1:
                            st.divider()
                            st.markdown("#### 📦 Your Masked Dataset")
                            st.dataframe(st.session_state.masked_df, use_container_width=True, height=250)
                            
                            col_dl, col_test = st.columns(2)
                            buf = io.StringIO(); st.session_state.masked_df.to_csv(buf, index=False)
                            col_dl.download_button("📥 Download `masked_data.csv`", buf.getvalue(), "masked_data.csv", "text/csv", type="primary", use_container_width=True)
                            
                            if col_test.button("🔄 Test Decrypt (Verify)", use_container_width=True):
                                from pii_detector.masking.aes_masker import unmask_dataframe
                                try:
                                    with st.spinner("Decrypting to verify…"):
                                        test_dec = unmask_dataframe(st.session_state.masked_df, pw1)
                                    st.success("✅ Verification successful! Data decrypts perfectly.")
                                    st.dataframe(test_dec.head(5), use_container_width=True)
                                except Exception as e:
                                    st.error(f"❌ Verification failed: {e}")


# ─── TAB 2: UNMASK DATA ───────────────────────────────────────────────────────
with main_tab_unmask:
    st.markdown("""
    <div class="card">
      <b style="color:#c9d1d9">🔓 Unmask an Encrypted CSV</b><br>
      <div style="color:#8b949e;font-size:.84rem;margin-top:.5rem;line-height:1.8;">
        Upload a <code>masked_data.csv</code> file generated by this tool, enter your password, and restore the original data.
      </div>
    </div>
    """, unsafe_allow_html=True)

    um_file = st.file_uploader("📂 Upload masked CSV", type=["csv"], key="um_file")

    if um_file is not None:
        df_to_decrypt = pd.read_csv(um_file, dtype=str).fillna("")
        st.success(f"✅ Loaded `{um_file.name}` ({len(df_to_decrypt):,} rows)")
        
        from pii_detector.masking.aes_masker import is_masked_dataframe
        masked_cols = is_masked_dataframe(df_to_decrypt)
        
        if not masked_cols:
            st.warning("⚠️ No AES-encrypted columns found in this file.", icon="⚠️")
        else:
            st.markdown(f"**Encrypted columns detected:** `{'` · `'.join(masked_cols)}`")
            st.dataframe(df_to_decrypt.head(5), use_container_width=True)
            
            # Use pre-filled password if they just masked in the same session, else empty
            default_pw = st.session_state.get("mask_pw", "")
            um_pw = st.text_input("🔑 Enter decryption password", type="password", key="um_pw", value=default_pw)
            
            if um_pw and st.button("🔓 Decrypt Data", type="primary"):
                from pii_detector.masking.aes_masker import unmask_dataframe
                try:
                    with st.spinner("Decrypting AES-256-GCM data…"):
                        df_decrypted = unmask_dataframe(df_to_decrypt, um_pw)
                    st.success("✅ Decryption successful!")
                    st.markdown("#### 📦 Your Original Dataset")
                    st.dataframe(df_decrypted, use_container_width=True, height=300)
                    
                    dec_buf = io.StringIO(); df_decrypted.to_csv(dec_buf, index=False)
                    st.download_button("📥 Download `decrypted_data.csv`", dec_buf.getvalue(), "decrypted_data.csv", "text/csv", type="primary")
                except ValueError as e:
                    st.error(f"❌ Wrong password or corrupted data: {e}", icon="🔐")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
