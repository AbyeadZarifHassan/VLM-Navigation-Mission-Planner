# Advancing UAV Flight and Obstacle Avoidance with Cross-Modal Deep Learning Integration

> **B.Sc. Thesis** — Department of Robotics and Mechatronics Engineering, University of Dhaka (April 2026)

A cross-modal framework that integrates **Vision–Language Models (VLMs)** with **monocular depth estimation** to enable real-time, semantic obstacle avoidance and waypoint mission execution on a lightweight UAV. The system streams live FPV video from a Pixhawk 6C quadrotor to an off-board ground station, where a depth map and a structured textual prompt are jointly consumed by a VLM to produce MAVLink-compatible navigation commands.

This repository contains the **Mission Planner integration code, ground station application, depth + VLM reasoning pipeline, and MAVLink control layer** developed for the thesis.

---

## Authors

- **Samiha Tarannum Noor** (Roll: 2320307) — [@Tarannum-Samiha](https://github.com/Tarannum-Samiha)
- **Abyead Zarif Hassan** (Roll: 2320305) — [@AbyeadZarifHassan](https://github.com/AbyeadZarifHassan)

**Supervisor:** Dr. Md Mehedi Hasan, Assistant Professor, Department of Robotics and Mechatronics Engineering, University of Dhaka.

---

## Abstract

Unmanned Aerial Vehicles (UAVs) are increasingly deployed for surveillance, infrastructure inspection, and disaster response, yet classical GPS or geometry-based navigation methods provide only limited semantic understanding of unstructured environments. This thesis proposes an integrated cross-modal framework in which a UAV streams real-time video to an off-board ground station running a Vision–Language reasoning module fused with monocular depth estimation through **DepthAnything-V2**. The framework is validated through a two-phase research design: (i) a controlled Unity–ROS 2 simulation benchmark evaluating six contemporary VLMs across nine factorially-controlled scenes that vary obstacle density, lighting, weather, and obstacle behaviour, and (ii) a real-world deployment of the strongest cloud-based candidates on a physical Pixhawk 6C quadrotor on the University of Dhaka campus.

---

## Key Contributions

- A **two-phase evaluation framework** (simulation screening → real-world validation) for benchmarking foundation models in embodied UAV autonomy.
- A **multi-VLM benchmark** spanning six contemporary models across both proprietary and open-source deployment categories — the first reported in the literature for UAV obstacle avoidance.
- A **model-agnostic command-parsing layer** that uniformly translates VLM reasoning outputs into 11 discrete MAVLink-compatible motion primitives.
- A **ground station application with integrated Mission Planner** supporting QGC WPL 110 waypoint interchange, Leaflet map composition, and live obstacle-avoidance overlay during AUTO missions.
- A **finite state machine for obstacle avoidance** with hierarchical proximity thresholds, altitude safety envelope, VLM-timeout heuristic fallback, and operator manual override.

---

## System Architecture

The proposed pipeline consists of three primary stages executed off-board on a ground station:

1. **Depth Estimation Pipeline** — Live RGB frames from the FPV camera are processed by **DepthAnything-V2** (ViT-L/14 encoder + DPT decoder) to produce a dense pixel-wise depth map.
2. **Vision–Language Reasoning Framework** — The RGB frame and depth map are fused into a structured prompt over a 3 × 3 spatial grid. The active VLM applies Chain-of-Thought reasoning to produce a discrete navigation command.
3. **Navigation Override and MAVLink Control Loop** — The selected command is translated into a MAVLink message and transmitted to the Pixhawk 6C flight controller over a bidirectional telemetry link.

```
[FPV Camera] -> [DepthAnything-V2] -> [3x3 Depth Grid + RGB] -> [VLM Reasoning] -> [MAVLink Command] -> [Pixhawk 6C]
                                                                                          ^
                                                                    [Ground Station GUI + Mission Planner]
```

---

## Models Evaluated

| Category | Model | Purpose |
|---|---|---|
| Depth Estimation | DepthAnything-V2 (ViT-L/14 + DPT) | Monocular depth map generation |
| Cloud VLM | GPT-5 | Reasoning (highest-performing) |
| Cloud VLM | GPT-4o | Reasoning (environmental invariance) |
| Cloud VLM | Gemini 2.0 Flash | Reasoning (mid-tier) |
| Open-source VLM | LLaVA-7B | Local reasoning |
| Open-source VLM | Llama 3.2-Vision | Local reasoning |
| Open-source VLM | Qwen2.5-VL-3B | Local reasoning |

---

## Hardware Configuration

| Component | Specification |
|---|---|
| Flight Controller | Pixhawk 6C |
| Platform | Quadrotor UAV |
| Sensors | FPV camera (monocular RGB), onboard IMU, GPS |
| Communication | MAVLink over telemetry radio |
| Ground Station | Off-board PC with NVIDIA RTX-3050 (4 GB VRAM) |
| Simulation | Unity 2021.3+ with ROS 2 Humble bridge |

---

## Repository Structure

```
VLM_Navigation_with_Mission_Planner_Integration/
├── app/                    # Main application modules
├── main.py                 # Entry point
├── app_config.json         # Configuration
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE                 # MIT License
└── README.md
```

> **Companion repository:** The Unity simulation environment and standalone VLM agent scripts used in the simulation phase are maintained separately at [Drone-Sim-Thesis](https://github.com/AbyeadZarifHassan/Drone-Sim-Thesis).

---

## Key Results

### Simulation Phase (265 episodes, 9 scenes, 478 shared decision points)

| Model | Accuracy | False Positive | Path Efficiency | Latency |
|---|---|---|---|---|
| GPT-5 | 73.3% | 7.3% | 0.871 | 1.34 s |
| GPT-4o | 60.0% | 10.2% | 0.826 | 1.56 s |
| Gemini 2.0 Flash | 28.9% | 26.6% | — | 2.61 s |
| Qwen2.5-VL-3B | — | — | — | 143.4 s |
| Llama 3.2-Vision | — | — | — | 248.2 s |
| LLaVA-7B | — | — | — | 365.6 s |

### Real-World Phase (Pixhawk 6C, University of Dhaka campus)

| Model | Avoidance Accuracy | Mean Confidence | Mean Latency | Temporal Consistency |
|---|---|---|---|---|
| GPT-5 | 95.6% | 91.2% | 0.834 s | 0.93 |
| GPT-4o | 93.6% | 86.4% | 0.949 s | 0.89 |
| Gemini 2.0 Flash | 73.3% | 67.2% | 0.722 s | 0.64 |
| Heuristic Baseline | 77.8% | — | 0.305 s | — |

The full real-time pipeline operated at **29.4 fps with sub-120 ms depth processing latency** under genuine flight conditions.

---

## Requirements

- Python 3.9+
- PyTorch (for DepthAnything-V2 inference)
- pymavlink (MAVLink communication)
- opencv-python (frame processing)
- API keys for OpenAI (GPT-4o, GPT-5) and Google AI Studio (Gemini 2.0 Flash) as needed
- Mission Planner or compatible ArduPilot ground station (for waypoint cross-verification)
- For open-source VLMs: local Ollama or HuggingFace transformers setup

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AbyeadZarifHassan/VLM_Navigation_with_Mission_Planner_Integration.git
cd VLM_Navigation_with_Mission_Planner_Integration
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### 4. Launch the ground station application

```bash
python main.py
```

> Refer to the in-app help and the thesis (Chapter 3.13) for detailed operator control channels and panel descriptions.

---

## Command Vocabulary

The VLM produces one of the following discrete navigation commands per decision cycle:

```
{ Forward, Left, Right, Up, Down,
  Upper Right, Upper Left, Lower Right, Lower Left,
  Yaw 90° Clockwise, Yaw 90° Anti-clockwise }
```

---

## Citation

If you reference this work in your research, please cite it as a B.Sc. thesis from the University of Dhaka:

> Noor, S. T., & Hassan, A. Z. (2026). *Advancing UAV Flight and Obstacle Avoidance with Cross-Modal Deep Learning Integration* [B.Sc. Thesis, University of Dhaka]. Department of Robotics and Mechatronics Engineering.

BibTeX:

```bibtex
@mastersthesis{noor_hassan_2026_vlm_uav,
  author  = {Noor, Samiha Tarannum and Hassan, Abyead Zarif},
  title   = {Advancing UAV Flight and Obstacle Avoidance with Cross-Modal Deep Learning Integration},
  school  = {University of Dhaka},
  year    = {2026},
  type    = {B.Sc. Thesis},
  address = {Dhaka, Bangladesh}
}
```

> **Note:** This work is currently part of an undergraduate thesis submission. A peer-reviewed publication may follow. Please contact the authors before citing for substantial use.

---

## Acknowledgements

We extend our sincere gratitude to our supervisor, **Dr. Md Mehedi Hasan**, Assistant Professor, Department of Robotics and Mechatronics Engineering, University of Dhaka, for his invaluable guidance, intellectual rigour, and constant encouragement throughout this research. We also thank the faculty and staff of the department for providing the academic foundation, laboratory resources, and inspiring research environment that made this work possible.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details. The academic work and findings remain the intellectual property of the authors and the University of Dhaka.

---

## Contact

- **Samiha Tarannum Noor** — via GitHub [@Tarannum-Samiha](https://github.com/Tarannum-Samiha)
- **Abyead Zarif Hassan** — abyeadzarif3139@gmail.com — [@AbyeadZarifHassan](https://github.com/AbyeadZarifHassan)
