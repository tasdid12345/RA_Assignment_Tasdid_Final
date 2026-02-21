# Task-1a: Reproduction and Extension of “I Can Hear Your Alexa”


---
## Folder Structure

Task1a_Alexa_Reproduction/
│
├── → requirements.txt
│   (Python dependency list required to run reproduction scripts.)
│
├── → README.md
│    (Main documentation explaining reproduction steps, dataset usage,
│     model pipeline, and experimental setup.)
│
├── src/
│       → SB_HYBRID_TFIDF_SVC.py (Core source code directory of best model.)
│
├── results/
│       → classification_report.txt
│       → confusion_matrix.csv
│       → confusion_matrix.png
│       → metrics.txt
│
├── data_sample/
│    (Sample subset of processed Alexa traffic traces from authors repo)
│
└── Task1A_Technical_Summary.docx
      (Technical summary document)

## 1. Overview

This project reproduces and extends the classical machine-learning baseline from the paper:

"I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers"

The goal is to infer spoken voice commands from Amazon Echo devices using only encrypted network metadata. No packet payloads are accessed.

The system demonstrates that packet size, direction, and timing alone are sufficient to reveal sensitive user activities.

This implementation uses a lightweight but powerful pipeline based on TF-IDF and Linear Support Vector Machines.

---

## 2. Dataset

Source repository:

https://github.com/SmartHomePrivacyProject/VCFingerprinting

Files used:

data/trace_csv/*.csv  
amazon_echo_query_list_100.xlsx  

Dataset characteristics:

- 1000 encrypted traffic traces  
- 100 distinct voice commands  
- 10 captures per command  
- Each CSV contains packet timestamp, size, and direction  

The script automatically downloads the dataset using GitHub’s REST Contents API.

---

## 3. Original Data Collection (From Paper)

Traffic was collected from:

### Device

- Amazon Echo smart speaker


### Protocols Observed

- TLS over TCP  
- HTTPS  

Captured attributes:

- Packet timestamp  
- Packet size  
- Packet direction  

No payload content was recorded.

This ensures the attack operates entirely on encrypted traffic metadata.

---

## 4. Dataset Access Method (This Implementation)

The dataset is programmatically downloaded using:

GitHub REST Contents API:

https://api.github.com/repos/SmartHomePrivacyProject/VCFingerprinting/contents/data/trace_csv

Python library:

requests

Each CSV file is fetched via HTTPS and stored locally.

---

## 5. Feature Engineering

Each traffic trace is transformed into a sequence of discrete tokens.

### 5.1 Direction Tokens

IN / OUT  

---

### 5.2 Raw Packet Size Tokens

Examples:

OUT_1500  
IN_118  

---

### 5.3 Binned Packet Size Tokens

Packet sizes are discretized into bins:

OUT_B0 … OUT_B10  

---

### 5.4 Inter-Arrival Time Tokens (IAT)

Time gaps between packets are quantized:

I0 – I8  

---

### Final Representation Example

OUT_1500 OUT_B7 I_START IN_118 IN_B1 I3 OUT_600 OUT_B4 I2 ...

This converts packet traces into NLP-style token sequences.

---

## 6. Model Architecture

Classical ML pipeline:

Packet Tokens  
↓  
TF-IDF Vectorizer (1–4 grams)  
↓  
LinearSVC  
↓  
Probability Calibration  

Components:

- TfidfVectorizer  
- LinearSVC  
- CalibratedClassifierCV  

Why Linear SVM:

- Excellent performance on sparse data  
- Lightweight and fast  
- Interpretable  
- Matches original paper baseline  
- Suitable for academic reproducibility  

---

## 7. Evaluation Metrics

- Top-1 Accuracy  
- Top-3 Accuracy  
- Top-5 Accuracy  
- Full Classification Report  
- Confusion Matrix  

---

## 8. Output Files

After execution, the results folder contains:

classification_report.txt  
metrics.txt  
confusion_matrix.csv  
confusion_matrix.png  

---

## 9. How to Run

### Step 1 — Create Virtual Environment

python -m venv .venv  

Activate (Windows):

.venv\Scripts\activate  

---

### Step 2 — Install Dependencies

pip install -r requirements.txt  

---

### Step 3 — Run Model

python SB_TFIDF_SVC_DUALTOKENS_GRID.py  

---

## 10. requirements.txt

numpy  
pandas  
scikit-learn  
matplotlib  
requests  
openpyxl  

---

## 11. Privacy & Ethics

- No packet payloads are inspected  
- Only encrypted metadata is used  
- Demonstrates real privacy leakage risks in smart devices  
- Intended strictly for academic research  

---

## 12. Possible Extensions

- 1D CNN / LSTM / TCN models  
- SHAP / LIME explainability  
- Burst smoothing defenses  
- Traffic padding  
- User behavior profiling  
- Website fingerprinting  
- Mobile application fingerprinting  
- Real-time deployment  

---

## 13. Research Significance

This project demonstrates that:

Encrypted traffic alone can leak voice commands and user behavior.

Even simple classical ML models achieve strong performance, highlighting serious privacy risks in IoT ecosystems.

---

## 14. License

Academic research use only.

---

End of README