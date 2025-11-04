# AI Hackathon: Elephant Identification

## 🏆 Competition Results
- 🥈 **Second Place**
- 🏅 **Excellent System Design Award**

## 👥 Team — ฟอร์มช้าง (FormChang)
**Members:** Phatthanasak Kraiduang · Kanlayanawat School \n Nattakit Chantara-aree · SKR \n Thanapong Wanna · SKR

## 🔎 What is this?
- **Goal:** Recognize **individual elephants** from photos (re-ID) for conservation & monitoring.
- **Why it matters:** Non-invasive tracking, reducing manual labeling effort.
- **Key idea:** Visual embeddings + metric learning + calibrated decision thresholds.

## ✨ Features
- 🔍 **Top-1 ID prediction** 
- 🧭 **Unknown detection**
- 🧠 **ONNX Runtime** inference (CPU/GPU) + Streamlit demo UI
- 📦 **One-file deployment** (drop-in `model.onnx` + `config.json`)
- 
---

## 🧱 Tech Stack
- **Model:** ViT → **ONNX**
- **Runtime:** Python 3.10+, **onnxruntime**
- **UI:** **Streamlit**
- **Training:** PyTorch
- **Utilities:** OpenCV, TorchVision, NumPy, Pandas
