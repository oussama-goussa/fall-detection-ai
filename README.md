# 🚨 AI-Powered Fall Detection System

*Real-time elderly safety monitoring using MPU6050 sensor and artificial intelligence*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Arduino](https://img.shields.io/badge/Arduino-Uno-green)
![Virtual Serial](https://img.shields.io/badge/Virtual_Serial-Port-blueviolet)

## 📖 Overview

This project implements an intelligent embedded system that detects falls in real-time using an MPU6050 inertial sensor and artificial intelligence. The device provides immediate audio and visual alerts through a buzzer, LED, and LCD display, specifically designed to assist vulnerable elderly individuals.

## 🎯 Key Features

- **🤖 AI-Powered Detection**: Neural network model for accurate fall classification
- **⏱️ Real-time Monitoring**: Continuous sensor data analysis
- **🚨 Multi-Modal Alerts**: Audio (buzzer) + Visual (LED + LCD) notifications
- **📊 Data Processing**: Advanced preprocessing with SMOTE for balanced classes
- **🔌 Virtual Serial Communication**: Connect Wokwi simulation with Python using Virtual Serial Port

## 🔗 Virtual Serial Setup

### Required Software
- **Virtual Serial Port Driver**: [Download here](https://www.virtual-serial-port.org/)
- **Wokwi Simulation**: [Live Simulation](https://wokwi.com/projects/419729501675955201)

### Configuration Steps

1. **Install Virtual Serial Port Driver**
   - Download and install from the official website
   - Create a virtual port pair (e.g., COM2 ↔ COM3)

2. **Wokwi Arduino Code Setup**
```cpp
// In your Arduino code, set the serial communication
void setup() {
  Serial.begin(9600);
  // ... other setup code
}

void loop() {
  // Read sensor data
  int accX = analogRead(A0);
  int accY = analogRead(A1);
  int accZ = analogRead(A2);
  
  // Send data to Python via serial
  Serial.print(accX);
  Serial.print(",");
  Serial.print(accY);
  Serial.print(",");
  Serial.print(accZ);
  Serial.println();
  
  delay(100);
}
```

3. **Python Serial Configuration**
```python
import serial
import time

# Configure virtual serial port
ser = serial.Serial('COM2', 9600, timeout=1)  # Match Wokwi output port
print("Connected to virtual serial port COM2")

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line:
            print(f"Received: {line}")
            # Process data for AI prediction
```

## 🛠️ System Architecture

```
Wokwi Simulation → Virtual Serial Port → Python AI → Fall Detection → Alerts
      ↑               (COM2 ↔ COM3)         ↓
MPU6050 Data                             Real-time Prediction
```

## 📁 Project Structure

```
fall-detection-ai/
├── models/
│   ├── acc_gyr_model.h5
│   └── label_encoder.pkl
├── python/
│   ├── training.py
│   └── application_detection_des_chutes.py
├── data/
│   └── acc_gyr.csv
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Arduino Uno, MPU6050, LCD 16x2, LED, Buzzer
- **Software**: 
  - Python 3.8+
  - Arduino IDE
  - Virtual Serial Port Driver
  - Wokwi account for simulation

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/oussama-goussa/fall-detection-ai.git
cd fall-detection-ai
```

2. **Install Python dependencies**
```bash
pip install tensorflow pandas numpy scikit-learn imbalanced-learn joblib pyserial
```

3. **Setup Virtual Serial Port**
   - Install Virtual Serial Port Driver
   - Create a pair: COM2 (Python) ↔ COM3 (Wokwi)
   - Configure ports in both Wokwi and Python code

4. **Configure Wokwi Simulation**
   - Open [Wokwi Project](https://wokwi.com/projects/419729501675955201)
   - Add serial communication code to Arduino sketch
   - Set baud rate to 9600

5. **Run the System**
```bash
cd python
python application_detection_des_chutes.py
```

## 🔧 Virtual Serial Configuration

### Port Settings
```python
# In application_detection_des_chutes.py
ser = serial.Serial(
    port='COM2',           # Virtual port for Python
    baudrate=9600,         # Match Arduino baud rate
    timeout=1,             # Read timeout
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)
```

### Data Format
Arduino sends data in CSV format:
```
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```

## 🤖 AI Model Details

### Neural Network Architecture
```python
model = models.Sequential([
    layers.Flatten(input_shape=(400, 6)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(7, activation='softmax')
])
```

## 🎮 Wokwi Integration

### Simulation Setup
1. **Open Wokwi Project**: [https://wokwi.com/projects/419729501675955201](https://wokwi.com/projects/419729501675955201)
2. **Add Serial Output**: Modify Arduino code to send sensor data
3. **Configure Virtual Port**: Connect Wokwi to virtual COM port
4. **Start Python Script**: Run detection application

### Expected Output
```
Waiting for data to start the prediction...
Received: 1.23,0.45,-9.81,0.12,0.08,0.03
Predicted Label: Walking
Received: 8.95,2.34,-1.23,4.56,3.21,2.89
Predicted Label: Falling
🚨 FALL DETECTED! Activating alerts...
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port Connection Failed**
   - Check if virtual ports are properly paired
   - Verify port names in both Wokwi and Python code
   - Ensure no other application is using the ports

2. **No Data Received**
   - Check baud rate matching (9600)
   - Verify Wokwi simulation is running
   - Test virtual port connection

3. **Model Loading Errors**
   - Ensure model files are in `./models/` directory
   - Check file paths in Python code
   - Verify TensorFlow version compatibility

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Real-time Detection | < 2 seconds |
| Model Accuracy | > 95% |
| Serial Data Rate | 9600 baud |
| Virtual Port Latency | < 100ms |

## 🔗 Useful Links

- **GitHub Repository**: [https://github.com/oussama-goussa/fall-detection-ai](https://github.com/oussama-goussa/fall-detection-ai)
- **Wokwi Simulation**: [https://wokwi.com/projects/419729501675955201](https://wokwi.com/projects/419729501675955201)
- **Virtual Serial Port**: [https://www.virtual-serial-port.org/](https://www.virtual-serial-port.org/)

## ⚠️ Disclaimer

This system is designed as an assistive technology and should not replace professional medical care or supervision. Always test the system thoroughly before deployment.

---

<div align="center">

**Made with ❤️ for Elderly Safety**

*If this project helps you, please give it a ⭐!*

[![GitHub stars](https://img.shields.io/github/stars/oussama-goussa/fall-detection-ai?style=social)](https://github.com/oussama-goussa/fall-detection-ai)

</div>
