# Task-1b: Encrypted Website Traffic Fingerprinting Dataset


---

## Objective

This project demonstrates the capability to collect encrypted network traffic data and prepare a machine-learning–ready fingerprinting dataset. The goal is to show a complete experimental pipeline including traffic capture, preprocessing, labeling, and baseline validation for website fingerprinting.

Only packet-level metadata is used. No payload content is inspected.

---

## Folder Structure

Task1b_WebFingerprint_dataCollection/
│
├── pcap/              Raw encrypted traffic captures (.pcapng)
├── csv_raw/           Packet metadata exported from PCAP
├── csv_clean/         Cleaned ML-ready traces
├── pcap2csv_code/     Conversion & preprocessing scripts
└── labels.csv         Mapping of cleaned traces to labels

---

## Dataset Collection Parameters

Encrypted traffic was collected from the following five websites:

- Amazon  
- BBC  
- GitHub  
- Wikipedia  
- YouTube  

Each website was accessed **10 times**, and every browsing session was recorded for approximately **13 seconds**, resulting in a balanced dataset of short encrypted traffic traces.

This controlled setup ensures:
- equal number of traces per website  
- consistent capture duration  
- reduced background interference  
- reproducible browsing behavior  

The final dataset consists of 50 encrypted traces (5 websites × 10 repetitions), suitable for supervised website fingerprinting experiments.

---


## Tools and Technologies

### Traffic Collection
- Google Chrome  
- Selenium (Python) for automated browsing  
- Wireshark / tshark for encrypted packet capture  

### Data Processing
- Python 3.x  
- pandas  
- NumPy  
- tshark  

Extracted packet features:
- Relative timestamp  
- Packet size  
- Direction (+1 outgoing, −1 incoming)  

---

## Data Collection Methodology

1. Wireshark/tshark captures encrypted TLS traffic.  
2. PCAP files are converted to CSV (csv_raw).  
3. CSV files are cleaned and normalized (csv_clean).  
4. Cleaned traces are mapped using labels.csv.  

Each trace represents approximately 13 seconds of browsing activity.

---


## Example End-to-End Pipeline

pcap/amazon_trace_1.pcapng  
→ csv_raw/amazon_trace_1.csv  
→ csv_clean/amazon_trace_1_clean.csv  
→ labels.csv mapping  

This demonstrates the complete acquisition and preprocessing workflow.

---

## Dataset Components

- pcap/: Raw encrypted traffic captures  
- csv_raw/: Direct packet exports  
- csv_clean/: Final ML-ready traces  
- pcap2csv_code/: Python preprocessing scripts  
- labels.csv: Maps cleaned traces to website labels  

Example labels.csv entry:

amazon,amazon_trace_1_clean.csv

---

## Ethical Considerations

- Only encrypted traffic metadata collected  
- No payload inspection  
- Data generated on personal devices for research purposes  

---

## Purpose

Prepared to demonstrate:

- Real traffic collection  
- Feature extraction  
- Dataset construction  
- Label management  

---

