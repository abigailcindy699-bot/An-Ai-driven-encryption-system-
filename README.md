# 🔐 AI-Driven Dynamic Encryption System

**Academic Project Implementation**  
AI-driven adaptive encryption using Random Forest + AES-128/192/256

---

## 📋 Requirements

- Python 3.9 or higher
- pip (Python package manager)

---

## ⚙️ Installation & Setup

### Step 1 — Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### Step 2 — Launch the application

```bash
python -m streamlit run app.py
```

Your browser will automatically open at **http://localhost:8501**

---

## 🧭 How to Use the System (Step-by-Step)

### 1. Train the AI Model (do this first!)
- Click **🤖 Train AI Model** in the left sidebar
- Adjust the number of training samples (1000 is good to start)
- Click **"Generate Dataset & Train Model"**
- The system will train a Random Forest classifier and show accuracy, F1-scores, and feature importance

### 2. Encrypt Data
- Click **🔒 Encrypt Data**
- Type or paste the data you want to encrypt in the text box
- Enter a **passphrase** (you'll need this to decrypt later — keep it safe!)
- Set the data context:
  - **Data Type**: General, Financial, or Medical
  - **Sensitivity Score**: 0–10 (how sensitive is this data?)
  - **Threat Level**: Normal, Elevated, or Critical
  - **Access Frequency**: How many requests per hour
  - **Failed Attempts**: How many failed access attempts recently
- Click **"Analyze & Encrypt"**
- The AI decides whether to use AES-128, AES-192, or AES-256
- Copy the **Full Encrypted Output** JSON for decryption later

### 3. Decrypt Data
- Click **🔓 Decrypt Data**
- Paste the JSON output from the encryption step
- Enter the **same passphrase** used during encryption
- Click **"Decrypt"** to recover the original data

### 4. View Logs
- Click **📊 Encryption Logs** to see all encryption operations
- Download the log as CSV for reporting
- View GDPR & HIPAA compliance status for each operation

### 5. Analytics
- Click **📈 System Analytics** for charts on:
  - Encryption type distribution
  - Threat level trends
  - Security score over time
  - AI feature importance

---

## 🏗️ Project Structure

```
ai_encryption_system/
│
├── app.py                    ← Main Streamlit application (run this)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── data/
│   └── generate_dataset.py   ← Synthetic dataset generator
│
├── models/
│   ├── ai_engine.py          ← Random Forest model (train + predict)
│   └── rf_model.pkl          ← Saved model (created after training)
│
├── utils/
│   ├── encryption.py         ← AES encryption/decryption logic
│   └── logger.py             ← Encryption log manager
│
└── logs/
    └── encryption_log.csv    ← Auto-generated activity log
```

---

## 🤖 How the AI Works

The **Random Forest classifier** is trained on 6 features:

| Feature | Description |
|---|---|
| Data Type | General (0), Financial (1), Medical (2) |
| Sensitivity Score | 0–10 scale |
| Threat Level | Normal (0), Elevated (1), Critical (2) |
| Access Frequency | Requests per hour |
| Failed Attempts | Failed logins in last hour |
| Time of Day | Hour (0–23) |

**Output classes:**
- `AES-128` — Low risk (128-bit key)
- `AES-192` — Medium risk (192-bit key)
- `AES-256` — High risk (256-bit key)

---

## 🔐 Encryption Details

- **Algorithm:** AES (Advanced Encryption Standard)
- **Mode:** CBC (Cipher Block Chaining)
- **IV:** Random 16-byte IV generated per encryption
- **Key Derivation:** SHA-256 hash of passphrase, trimmed to 16/24/32 bytes
- **Padding:** PKCS7

---

## 📜 Compliance Scoring

| Score | GDPR | HIPAA |
|---|---|---|
| ≥ 75 | ✅ | ✅ |
| 70–74 | ✅ | ❌ |
| < 70 | ❌ | ❌ |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| AI Engine | Scikit-learn (Random Forest) |
| Encryption | PyCryptodome (AES-CBC) |
| Data | Pandas, NumPy |
| Language | Python 3.9+ |
