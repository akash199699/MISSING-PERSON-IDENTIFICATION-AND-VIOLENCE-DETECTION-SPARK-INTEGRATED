# 🧠 CNN-Based Framework for Missing Person Identification and Violence Detection Using Deep Learning and Distributed Processing

## 📌 Project Title
**CNN-Based Framework for Missing Person Identification and Violence Detection Using Deep Learning and Distributed Processing**

## 🧠 Introduction / Overview
This project leverages deep learning and distributed computing to solve two critical problems in surveillance systems:
1. **Missing Person Identification** – Detect and recognize missing individuals using CNN-based face recognition.
2. **Violence Detection** – Analyze video frames to detect violent activities using a trained CNN model.

The project integrates a user-friendly UI with a backend powered by **Apache Spark** for distributed processing, making it scalable for large-scale surveillance footage analysis.

---

## ⚙️ Working / Functionality

### Core Features:
- **Face Recognition**: Detect and match faces using a trained CNN model.
- **Violence Detection**: Classify actions as violent or non-violent using video analysis.
- **UI Panel**: A simple Python UI for interacting with both functionalities.
- **Distributed Processing**: Spark-based data processing for efficient performance on larger datasets.
- **Report Generation**: Create summaries/statistics of detections.

### Components:
- `ui.py`: Main UI interface.
- `missing_person_detection.py`: Contains face recognition logic.
- `violence_detection.py`: Contains violence classification logic.
- `spark_processing.py`: Spark-based processing logic.
- `report_generation.py`: Generates detection reports.
- `start_cluster.py`: Starts Spark cluster for distributed mode.

---

## 🧱 Architecture / Flow
                     ┌────────────────────────────┐
                     │       User Interface       │
                     │     (Tkinter-based UI)     │
                     └────────────┬───────────────┘
                                  │
                     ┌────────────▼───────────────┐
                     │ Input Media (Images/Videos)│
                     └────────────┬───────────────┘
                                  │
          ┌───────────────────────▼──────────────────────┐
          │            Preprocessing Module              │
          │  (Resizing, Normalization, Face Extraction)  │
          └───────────────────────┬──────────────────────┘
                                  │
     ┌────────────────────────────▼──────────────────────────────┐
     │                  Deep Learning Inference                  │
     │  ┌────────────────────────┐  ┌──────────────────────────┐ │
     │  │  Face Recognition (CNN)│  │ Violence Detection (CNN) │ │
     │  └────────────────────────┘  └──────────────────────────┘ │
     └────────────────────────────┬──────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Distributed Processing    │
                    │     (Apache Spark)        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     Report Generation     │
                    │    (PDF, Stats, Logging)  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Output Results       │
                    │   (UI + Logs + Reports)   │
                    └───────────────────────────┘

---

## 🔧 Technologies Used

- **Python**
- **OpenCV**
- **TensorFlow / Keras**
- **PyTorch / facenet-pytorch**
- **Apache Spark (PySpark)**
- **Tkinter (for UI)**
- **NumPy, Matplotlib**
- **FPDF (PDF Report)**

---

## 🛠️ Setup Instructions

### 🔹 Prerequisites
- Python 3.8+
- Java 8+
- Apache Spark
- Virtual environment (recommended)

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```


## 🙌 Credits
- Akash Krishna – Project Developer
- (B.Tech AI & ML, KTU – 6th Semester)
- 📧 Email: akash199699@gmail.com
- 🔗 GitHub: @akash199699

This project was developed as part of the mini project under the university curriculum.
Special thanks to our mentors for their guidance, and teammates Anandhu S Kumar and Jewel Saji for their collaboration.
