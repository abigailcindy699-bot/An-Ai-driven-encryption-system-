"""
AI-Driven Dynamic Encryption System
Main Streamlit Application

Implements the system described in the academic project:
- AI Decision Engine (Random Forest) selects encryption strength
- Dynamic AES encryption (128/192/256-bit) based on data sensitivity & threat level
- Encryption log with compliance tracking (GDPR/HIPAA)
- Model training interface with performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json

# Internal modules
from data.generate_dataset import generate_training_data
from models.ai_engine import (
    train_model, load_model, predict_encryption,
    get_feature_importance, ENCRYPTION_LABELS, DATA_TYPE_MAP, THREAT_LEVEL_MAP
)
from utils.encryption import encrypt_data, decrypt_data, compute_security_score
from utils.logger import append_log, load_logs, clear_logs, get_summary_stats

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Driven Dynamic Encryption System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2rem; margin: 0; }
    .main-header p { color: #a0aec0; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .label { font-size: 0.8rem; opacity: 0.85; }
    .metric-card .value { font-size: 1.8rem; font-weight: bold; }

    .aes128 { background: linear-gradient(135deg, #11998e, #38ef7d) !important; }
    .aes192 { background: linear-gradient(135deg, #f7971e, #ffd200) !important; color: #1a1a1a !important; }
    .aes256 { background: linear-gradient(135deg, #e53e3e, #c53030) !important; }

    .result-box {
        background: #1a1a2e;
        border: 1px solid #4a5568;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e2e8f0;
        font-family: monospace;
        word-break: break-all;
    }

    .compliance-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .badge-pass { background: #276749; color: #9ae6b4; }
    .badge-fail { background: #742a2a; color: #fed7d7; }

    .sidebar-section {
        background: #2d3748;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🤖 Train AI Model", "🔒 Encrypt Data",
         "🔓 Decrypt Data", "📊 Encryption Logs", "📈 System Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    # Model status
    model = load_model()
    if model:
        st.success("✅ AI Model: Trained & Ready")
    else:
        st.warning("⚠️ AI Model: Not Trained Yet")
    st.markdown("---")
    st.caption("AI-Driven Dynamic Encryption System v1.0\nBuilt with Random Forest + AES")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🔐 AI-Driven Dynamic Encryption System</h1>
        <p>Adaptive data protection powered by Machine Learning and AES Cryptography</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">AI Engine</div>
            <div class="value">Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Encryption Algorithms</div>
            <div class="value">AES 128/192/256</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Compliance</div>
            <div class="value">GDPR & HIPAA</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **How It Works (Algorithm)**
        
        1. **Input** — User provides data and context (type, sensitivity, threat level)
        2. **Feature Analysis** — System evaluates sensitivity and threat conditions
        3. **AI Decision** — Random Forest model predicts optimal encryption strength
        4. **Dynamic Encryption** — AES applied with selected key length
        5. **Logging** — Activity logged with compliance status
        6. **Output** — Encrypted data returned securely
        """)
    with col_b:
        st.markdown("""
        **Encryption Selection Rules**
        
        | Risk Level | Encryption | Key Size |
        |---|---|---|
        | 🟢 Low risk | AES-128 | 128-bit |
        | 🟡 Medium risk | AES-192 | 192-bit |
        | 🔴 High risk | AES-256 | 256-bit |
        
        **Regulated Sectors Supported**
        - 🏥 Healthcare (HIPAA)
        - 🏦 Finance (GDPR)
        - 🏛️ Government data
        """)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    st.info("""
    **Step 1:** Go to **🤖 Train AI Model** — generate a dataset and train the Random Forest model.
    
    **Step 2:** Go to **🔒 Encrypt Data** — enter your data, select context, and let the AI choose encryption strength.
    
    **Step 3:** Go to **🔓 Decrypt Data** — paste the encrypted output to retrieve original data.
    
    **Step 4:** Go to **📊 Encryption Logs** — review all encryption operations and compliance status.
    """)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: TRAIN AI MODEL
# ═══════════════════════════════════════════════════════════════════════
elif page == "🤖 Train AI Model":
    st.markdown("## 🤖 Train the AI Decision Engine")
    st.write("Generate synthetic training data and train the Random Forest classifier that selects encryption strength.")

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Training Samples", min_value=500, max_value=5000, value=1000, step=500)
    with col2:
        seed = st.number_input("Random Seed (for reproducibility)", value=42, step=1)

    if st.button("⚡ Generate Dataset & Train Model", type="primary", use_container_width=True):
        with st.spinner("Generating dataset..."):
            df = generate_training_data(n_samples=n_samples, seed=int(seed))

        st.success(f"✅ Dataset generated: {len(df)} samples")

        with st.expander("📋 Dataset Preview (first 10 rows)"):
            display_df = df.copy()
            display_df['data_type'] = display_df['data_type'].map({0: 'General', 1: 'Financial', 2: 'Medical'})
            display_df['threat_level'] = display_df['threat_level'].map({0: 'Normal', 1: 'Elevated', 2: 'Critical'})
            display_df['encryption_strength'] = display_df['encryption_strength'].map({0: 'AES-128', 1: 'AES-192', 2: 'AES-256'})
            st.dataframe(display_df.head(10), use_container_width=True)

        # Class distribution
        col_a, col_b = st.columns(2)
        with col_a:
            dist = df['encryption_strength'].value_counts().sort_index()
            st.markdown("**Class Distribution**")
            for k, v in dist.items():
                label = ENCRYPTION_LABELS[k]
                st.write(f"• {label}: {v} samples ({v/len(df)*100:.1f}%)")

        with st.spinner("Training Random Forest model..."):
            model, accuracy, report, X_test, y_test, y_pred = train_model(df)

        st.success(f"✅ Model trained! Accuracy: **{accuracy*100:.2f}%**")

        col1, col2, col3 = st.columns(3)
        for i, (label, col) in enumerate(zip(["AES-128", "AES-192", "AES-256"], [col1, col2, col3])):
            with col:
                prec = report[label]['precision']
                rec = report[label]['recall']
                f1 = report[label]['f1-score']
                st.metric(f"{label} F1-Score", f"{f1:.3f}")
                st.caption(f"Precision: {prec:.3f} | Recall: {rec:.3f}")

        # Feature importance
        st.markdown("### 📊 Feature Importance")
        importance = get_feature_importance()
        if importance:
            imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            imp_df = imp_df.sort_values('Importance', ascending=True)
            st.bar_chart(imp_df.set_index('Feature'))


# ═══════════════════════════════════════════════════════════════════════
# PAGE: ENCRYPT DATA
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔒 Encrypt Data":
    st.markdown("## 🔒 Encrypt Data")
    st.write("Enter your data and contextual information. The AI engine will determine the appropriate encryption strength.")

    model = load_model()
    if not model:
        st.error("⚠️ AI Model is not trained yet. Please go to **🤖 Train AI Model** first.")
        st.stop()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 📝 Data Input")
        plaintext = st.text_area("Enter data to encrypt", height=150,
                                  placeholder="Enter medical records, financial data, or any sensitive text here...")
        passphrase = st.text_input("Encryption passphrase", type="password",
                                    placeholder="Enter a secret passphrase (needed for decryption)")

    with col2:
        st.markdown("### 🎯 Threat Context")
        data_type = st.selectbox("Data Type", ["General", "Financial", "Medical"])
        sensitivity_score = st.slider("Sensitivity Score", 0.0, 10.0, 5.0, 0.5,
                                       help="0 = not sensitive, 10 = extremely sensitive")
        threat_level = st.selectbox("Threat Level", ["Normal", "Elevated", "Critical"])
        access_frequency = st.number_input("Access Frequency (requests/hour)", 1, 1000, 10)
        failed_attempts = st.number_input("Failed Access Attempts (last hour)", 0, 100, 0)
        time_of_day = st.slider("Current Hour (0-23)", 0, 23, datetime.datetime.now().hour)

    st.markdown("---")
    if st.button("🔒 Analyze & Encrypt", type="primary", use_container_width=True):
        if not plaintext.strip():
            st.error("Please enter data to encrypt.")
        elif not passphrase.strip():
            st.error("Please enter a passphrase.")
        else:
            with st.spinner("AI analyzing context..."):
                prediction = predict_encryption(
                    data_type=data_type,
                    sensitivity_score=sensitivity_score,
                    threat_level=threat_level,
                    access_frequency=access_frequency,
                    failed_attempts=failed_attempts,
                    time_of_day=time_of_day
                )

            enc_label = prediction['encryption_label']
            key_bytes = prediction['key_length_bytes']
            confidence = prediction['confidence']

            # Color for encryption type
            color_map = {"AES-128": "aes128", "AES-192": "aes192", "AES-256": "aes256"}
            css_class = color_map.get(enc_label, "")

            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div class="label">🤖 AI Decision — Encryption Strength Selected</div>
                <div class="value">{enc_label} ({key_bytes*8}-bit)</div>
                <div class="label">Model Confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("**AI Confidence Breakdown:**")
            prob_cols = st.columns(3)
            for i, (alg, prob) in enumerate(prediction['probabilities'].items()):
                with prob_cols[i]:
                    st.metric(alg, f"{prob*100:.1f}%")

            # Encrypt
            with st.spinner("Encrypting data..."):
                result = encrypt_data(plaintext, passphrase, key_bytes, enc_label)

            # Compliance
            compliance = compute_security_score(enc_label, threat_level, sensitivity_score)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("### 🔐 Encryption Result")
                st.markdown(f"""
                <div class="result-box">
                    <b>Encryption Type:</b> {result['encryption_type']}<br>
                    <b>Key Length:</b> {result['key_length_bits']}-bit<br>
                    <b>Mode:</b> {result['mode']}<br>
                    <b>Timestamp:</b> {result['timestamp']}<br><br>
                    <b>IV (Base64):</b><br><code>{result['iv']}</code><br><br>
                    <b>Ciphertext (Base64):</b><br><code>{result['ciphertext'][:80]}...</code>
                </div>
                """, unsafe_allow_html=True)

            with col_r2:
                st.markdown("### 📋 Compliance Report")
                score = compliance['security_score']
                st.metric("Security Score", f"{score}/100")
                gdpr_badge = "badge-pass" if compliance['gdpr_compliant'] else "badge-fail"
                gdpr_text = "✅ GDPR Compliant" if compliance['gdpr_compliant'] else "❌ GDPR Non-Compliant"
                hipaa_badge = "badge-pass" if compliance['hipaa_compliant'] else "badge-fail"
                hipaa_text = "✅ HIPAA Compliant" if compliance['hipaa_compliant'] else "❌ HIPAA Non-Compliant"

                st.markdown(f"""
                <span class="compliance-badge {gdpr_badge}">{gdpr_text}</span><br>
                <span class="compliance-badge {hipaa_badge}">{hipaa_text}</span><br>
                """, unsafe_allow_html=True)
                st.caption(f"Compliance Level: **{compliance['compliance_level']}**")

            # Store full result in session for copy
            st.session_state['last_encryption'] = result

            # Save to log
            append_log({
                "data_type": data_type,
                "sensitivity_score": sensitivity_score,
                "threat_level": threat_level,
                "failed_attempts": failed_attempts,
                "encryption_type": enc_label,
                "key_length_bits": key_bytes * 8,
                "security_score": compliance['security_score'],
                "gdpr_compliant": compliance['gdpr_compliant'],
                "hipaa_compliant": compliance['hipaa_compliant'],
                "status": "success",
                "timestamp": result['timestamp']
            })

            st.success("✅ Encryption complete! Activity logged.")

            # Full ciphertext for copying
            with st.expander("📋 Full Encrypted Output (for decryption)"):
                full_output = {
                    "iv": result['iv'],
                    "ciphertext": result['ciphertext'],
                    "encryption_type": result['encryption_type'],
                    "key_length_bits": result['key_length_bits']
                }
                st.code(json.dumps(full_output, indent=2), language="json")
                st.caption("Copy this JSON and paste it into the Decrypt page.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: DECRYPT DATA
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔓 Decrypt Data":
    st.markdown("## 🔓 Decrypt Data")
    st.write("Paste the encrypted JSON output from the Encrypt page along with your passphrase to recover the original data.")

    col1, col2 = st.columns(2)
    with col1:
        encrypted_json = st.text_area("Paste Encrypted JSON Output", height=200,
                                       placeholder='{"iv": "...", "ciphertext": "...", "encryption_type": "AES-256", "key_length_bits": 256}')
    with col2:
        passphrase = st.text_input("Decryption Passphrase", type="password",
                                    placeholder="Enter the passphrase used during encryption")
        st.info("The passphrase must match exactly what was used during encryption.")

    if st.button("🔓 Decrypt", type="primary", use_container_width=True):
        if not encrypted_json.strip():
            st.error("Please paste the encrypted JSON output.")
        elif not passphrase.strip():
            st.error("Please enter the passphrase.")
        else:
            try:
                data = json.loads(encrypted_json)
                iv_b64 = data['iv']
                ciphertext_b64 = data['ciphertext']
                key_bits = data.get('key_length_bits', 256)
                key_bytes = key_bits // 8
                enc_type = data.get('encryption_type', 'AES-256')

                with st.spinner("Decrypting..."):
                    result = decrypt_data(ciphertext_b64, iv_b64, passphrase, key_bytes)

                if result['status'] == 'success':
                    st.success("✅ Decryption successful!")
                    st.markdown(f"""
                    <div class="result-box">
                        <b>Decryption Type Used:</b> {enc_type}<br>
                        <b>Timestamp:</b> {result['timestamp']}<br><br>
                        <b>Original Data:</b><br>
                        {result['plaintext']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Decryption failed: {result['message']}")
                    st.caption("Check your passphrase or encrypted data.")

            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format. Please paste the full JSON output from the Encrypt page.")
            except KeyError as e:
                st.error(f"❌ Missing field in JSON: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PAGE: ENCRYPTION LOGS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Encryption Logs":
    st.markdown("## 📊 Encryption Log & History")
    st.write("Complete audit trail of all encryption operations performed by the system.")

    df = load_logs()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            clear_logs()
            st.success("Logs cleared.")
            st.rerun()

    if df.empty:
        st.info("No encryption operations logged yet. Go to the Encrypt Data page to get started.")
    else:
        stats = get_summary_stats(df)

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Encryptions", stats.get('total_encryptions', 0))
        with c2:
            st.metric("AES-128", stats.get('aes_128_count', 0))
        with c3:
            st.metric("AES-192", stats.get('aes_192_count', 0))
        with c4:
            st.metric("AES-256", stats.get('aes_256_count', 0))
        with c5:
            st.metric("Avg Security Score", f"{stats.get('avg_security_score', 0)}/100")

        st.markdown("---")
        st.markdown("### 📋 Full Log")

        # Display log table
        display_df = df.copy()
        display_df['gdpr_compliant'] = display_df['gdpr_compliant'].map({True: '✅', False: '❌', 1: '✅', 0: '❌'})
        display_df['hipaa_compliant'] = display_df['hipaa_compliant'].map({True: '✅', False: '❌', 1: '✅', 0: '❌'})
        st.dataframe(
            display_df[['log_id', 'timestamp', 'data_type', 'threat_level',
                         'encryption_type', 'key_length_bits', 'security_score',
                         'gdpr_compliant', 'hipaa_compliant']].sort_values('log_id', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Download
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Log as CSV",
            data=csv_data,
            file_name="encryption_log.csv",
            mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📈 System Analytics":
    st.markdown("## 📈 System Analytics")

    df = load_logs()

    if df.empty:
        st.info("No data yet. Perform some encryption operations first.")
    else:
        stats = get_summary_stats(df)

        st.markdown("### Compliance Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GDPR Compliance Rate", f"{stats.get('gdpr_compliant_pct', 0)}%")
        with col2:
            st.metric("HIPAA Compliance Rate", f"{stats.get('hipaa_compliant_pct', 0)}%")

        st.markdown("### Encryption Type Distribution")
        enc_counts = df['encryption_type'].value_counts()
        st.bar_chart(enc_counts)

        st.markdown("### Threat Level Distribution")
        threat_counts = df['threat_level'].value_counts()
        st.bar_chart(threat_counts)

        st.markdown("### Security Score Over Time")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            score_over_time = df.set_index('timestamp')['security_score']
            st.line_chart(score_over_time)

        st.markdown("### Data Type Breakdown")
        dtype_counts = df['data_type'].value_counts()
        st.bar_chart(dtype_counts)

    # AI model feature importance
    st.markdown("---")
    st.markdown("### 🤖 AI Model — Feature Importance")
    importance = get_feature_importance()
    if importance:
        imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        st.bar_chart(imp_df.set_index('Feature'))
    else:
        st.info("Train the AI model first to see feature importance.")
