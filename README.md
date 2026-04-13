# 🚀 X-Ray Cargo Inspection AI  
A multimodal AI system that uses **YOLOv8** for X‑ray object detection and an **LLM** for intelligent report generation by analyzing both shipping documents and image detections.

---

## 🎥 Demo Video  
Click below to watch the system in action:

👉 **[Demo Video](YOUR_VIDEO_LINK_HERE)**  
*(Replace this link with your GitHub or YouTube video link)*

---

## 📌 Overview  
This project automates the inspection of cargo and parcels by combining **computer vision** and **large language models**.  
It analyzes:

- X‑ray images  
- Shipping documents  
- Detected objects  
- Declared contents  

Then generates a structured, intelligent inspection report.

This system is designed for **logistics**, **customs**, **security screening**, and **automated threat detection**.

---

## 🎯 Problem Statement  
Manual cargo inspection is slow and error‑prone:

- X‑ray images are difficult to interpret  
- Shipping documents must be cross‑checked manually  
- Human inspectors may miss anomalies  
- Threat detection requires expertise  

This project solves these issues using **AI‑powered multimodal analysis**.

---

## 🧠 Solution Architecture  

### **1. YOLOv8 Object Detection**
- Detects items inside X‑ray images  
- Identifies suspicious or prohibited objects  
- Outputs bounding boxes + class labels  

### **2. LLM Document Understanding**
- Reads shipping documents (PDF, text, invoice, manifest)  
- Extracts declared items and descriptions  

### **3. Multimodal Reasoning**
- Compares detected objects vs declared items  
- Flags mismatches or anomalies
