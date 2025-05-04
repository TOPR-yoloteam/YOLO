<p align="center">
  <img src="https://raw.githubusercontent.com/TOPR-yoloteam/YOLO/main-1/src/utils/THU_Logo.png" alt="THU Logo" width="100"/>
</p>


# 🛠️ Hands-On-Tutorial: Image Detection with YOLO AI on a Raspberry Pi
### THU Summer School 2025 · Cybersecurity Today

Welcome to the official tutorial repository for the **"Image Detection with YOLO AI on a Raspberry Pi"** workshop, developed as part of the [THU Summer School 2025](https://www.thu.de/de/Downloads/THU_Summerschool_2025.pdf).  
This hands-on project is tailored for international participants of the *Cybersecurity Today* summer school program, held from **July 28 – August 2, 2025**, at Technische Hochschule Ulm.

---

## 🎯 What You’ll Learn

In this practical session, you’ll dive into computer vision on edge devices, using YOLOv5 for detecting license plates and YOLOv8n/YOLOv11n human faces. You'll gain hands-on experience in:

- Setting up and configuring a Raspberry Pi for real-time object detection  
- Detecting and reading license plates via OCR  
- Implementing real-time person and face detection  
- Comparing YOLO and MediaPipe for efficiency and accuracy  
- Understanding real-world use cases of computer vision in cybersecurity contexts

---

## 🌍 Real-World Applications

### 🚗 License Plate Recognition (LPR) – *Toll Systems & Smart Parking*
Automated license plate recognition is widely used in toll collection systems and urban parking zones. Cameras at toll booths or parking entrances capture license plates and match them against payment or access databases — enabling seamless, contactless billing and traffic regulation.

### 🧍 Face Recognition – *Access Control & Smart Security*
Face recognition plays a key role in modern security systems. It enables secure access to offices, research labs, or data centers by authenticating individuals in real-time, often replacing traditional ID badges or PIN codes. In the context of cybersecurity, it supports multi-factor authentication and surveillance in sensitive environments.

---

## 📸 System Overview

This project simulates such scenarios on a **Raspberry Pi**, a compact and affordable edge device. Using a USB camera, the system captures image data and processes it using:

- **YOLOv8n** (optimized for fast object detection on CPU)  
- **`face_recognition`** for identifying known individuals  
- **Tesseract OCR** for reading text from license plates  
- Optional comparison with **MediaPipe** for performance benchmarking

**System Flow:**
1. Raspberry Pi boots into Python environment  
2. Camera captures live images  
3. YOLO detects persons or license plates  
4. Relevant regions (e.g. faces or plates) are cropped  
5. Face recognition or OCR extracts identity  
6. Results are logged or visualized live

## 📦 Requirements

- Raspberry Pi (with Raspberry Pi OS)
- USB-compatible camera
- HDMI monitor, keyboard, mouse, power supply
- Basic Python & Linux knowledge

You’ll also need to clone the repo:

```bash
git clone https://github.com/TOPR-yoloteam/YOLO.git
cd YOLO
```

## 📝 Full Hands-On-Tutorial (PDF)

For a detailed step-by-step guide including exercises, code walkthroughs, and troubleshooting tips, please refer to the full tutorial document:

📄 [**Download Hands-On Tutorial (PDF)**](https://github.com/TOPR-yoloteam/YOLO/tree/main-1/src/utils/TOPR_TeamD_Hands-On-Tutorial.pdf)

## 🎴 Promo poster

You can view the promotional poster here:
![Promo Poster](https://raw.githubusercontent.com/TOPR-yoloteam/YOLO/main-1/src/utils/Promo_Poster_en.pdf)



