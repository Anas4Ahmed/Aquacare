# 🌊 Aquacare — Autonomous Water Cleaning Robot

> Intelligent AI-powered aquatic waste management and water quality monitoring system.

🏆 1st Place — Innoctech 2024 (700+ Teams)  
📜 Indian Patent Published: Patent No. 202411104044 A  
📄 Research Published in Springer Journal Proceedings

---

# 📖 Overview

Aquacare is an autonomous robotic system designed for aquatic ecosystem preservation.  
The system uses AI-powered computer vision to detect and classify floating waste in real-time and autonomously navigates to collect it.

The robot integrates:

- YOLOv8 object detection
- Embedded control systems
- MQTT communication
- Water quality monitoring
- GPS tracking
- Autonomous navigation

Aquacare aims to provide scalable smart-cleaning solutions for lakes, ponds, rivers, and urban water bodies.

---

# 🚀 Features

- 🧠 Real-time waste detection using YOLOv8
- ♻️ Autonomous garbage collection mechanism
- 🌡️ Water quality monitoring
  - pH
  - turbidity
  - odor analysis
- 📡 MQTT-based communication
- 📍 GPS tracking
- 📊 Live monitoring dashboard
- 🔋 Embedded autonomous decision system
- 🌊 Aquatic ecosystem protection

---

# 🏗️ Architecture

```text
Camera Feed
     ↓
YOLOv8 Detection Engine
     ↓
Embedded Decision Logic
     ↓
ESP8266 Communication
     ↓
Motor Control System
     ↓
Garbage Collection Mechanism
     ↓
Dashboard + MQTT Monitoring
```

---

# 🛠️ Tech Stack

## AI / ML
- YOLOv8
- OpenCV
- Scikit-learn
- Python

## Embedded Systems
- ESP8266
- Sensors
- Motor Drivers

## Communication
- MQTT Protocol
- Wi-Fi Communication

## Dashboard
- Web-based monitoring system

## Operating System
- Linux Ubuntu / Fedora

---

# 🤖 Hardware Components

- ESP8266
- Camera Module
- Motors
- Motor Driver
- pH Sensor
- Turbidity Sensor
- GPS Module
- Power Supply System

---

# 🧠 AI/ML Pipeline

1. Dataset collection
2. Image preprocessing
3. Annotation and labeling
4. YOLOv8 training
5. Real-time inference
6. Waste classification
7. Autonomous navigation decision

---

# 📂 Dataset Information

- Dataset Size: 3,000+ images
- Multi-class waste categories
- Custom preprocessing and augmentation
- Real-world aquatic waste samples

---

# ⚙️ Working Flow

1. Camera captures live water surface
2. YOLOv8 detects floating waste
3. Embedded system calculates movement
4. Robot navigates autonomously
5. Waste gets collected
6. Sensor data sent to dashboard
7. Live monitoring updates continuously

---

# 📊 Performance

- 🎯 Detection Accuracy: 88%+
- ⚡ Real-time inference
- 🌊 Stable aquatic navigation
- 📡 Reliable MQTT communication

---

# 🏆 Awards & Achievements

- 🥇 1st Place — Innoctech 2024
- 📜 Indian Patent Granted
- 📄 Springer Research Publication

---

# 💻 Installation

```bash
git clone https://github.com/yourusername/Aquacare.git
cd Aquacare
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run detection system:

```bash
python detect.py
```

Start dashboard:

```bash
python dashboard.py
```

---

# 🔮 Future Scope

- Solar-powered charging
- Swarm robotics integration
- Advanced aquatic AI analytics
- Edge AI acceleration
- Satellite-linked monitoring systems

---

# 👨‍💻 Contributors

- MD Anas Ahmed

---

# 📜 License

This project is licensed for research and educational purposes.
