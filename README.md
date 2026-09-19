# 🏢 Workplace Activity & Ergonomic Safety Monitoring System

<div align="center">

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg?style=for-the-badge)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Computer%20Vision-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Workplace%20AI-5C3EE8.svg?style=for-the-badge)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**An intelligent computer vision system tracking desk occupancy, employee postures, and activity patterns in corporate office environments.**

</div>

---

## 📌 Overview

Designed for modern smart offices, this system processes video surveillance streams to monitor desk occupancy rates, detect unsafe posture habits, and analyze space utilization for facilities management.

---

## 🎯 Monitored Activities

- 💻 **Working at Desk**: Identifies focused workstation occupancy.
- 🧍 **Standing / Moving**: Tracks hallway movement and collaboration areas.
- 🪑 **Empty Workstation**: Calculates real-time facility seat utilization.

---

## 🛠️ Quick Start

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Detection on Sample Video**:
   ```bash
   python src/detect.py
   ```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
