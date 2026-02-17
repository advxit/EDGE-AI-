# Temporal Trade-offs in Spiking Neural Networks for Edge-Aware Neuromorphic Systems

## 📌 Overview

This project investigates the design, training, optimization, and deployment feasibility of **Leaky Integrate-and-Fire (LIF) based Spiking Neural Networks (SNNs)** with a focus on **temporal performance vs hardware efficiency trade-offs**.

Spiking Neural Networks operate using discrete spikes and temporal membrane dynamics, making them highly suitable for **neuromorphic computing, edge AI, and hardware-aware machine learning systems**.

This repository implements an end-to-end pipeline — from SNN training to edge deployment formats — while analyzing implications from **VLSI and analog neuromorphic perspectives**.

---

## 🎯 Objectives

* Implement LIF-based SNN architectures
* Train SNNs using surrogate gradient learning
* Evaluate timestep-dependent performance trade-offs
* Analyze spike sparsity as an energy proxy
* Study hardware-aware compute implications
* Export trained models to deployment formats
* Explore neuromorphic hardware mapping feasibility

---

## 🧠 Core Neuromorphic Concepts

* Membrane integration & leakage dynamics
* Spike-driven event computation
* Temporal evidence accumulation
* Surrogate gradient backpropagation
* Sparse activation patterns
* Energy-efficient inference

---

## 🏗️ Model Architecture

Baseline pipeline:

Input → Fully Connected Layer → LIF Neuron → Fully Connected Layer → LIF Output Layer

Key configurable parameters:

* Simulation timesteps (T)
* Membrane decay constant (β)
* Threshold dynamics
* Surrogate gradient function
* Temporal batch normalization

---

## 📊 Experimental Focus

Primary research axis:

```
T ∈ {5, 10, 25, 50}
```

For each configuration we measure:

* Classification Accuracy
* Inference Latency
* Total Spike Count
* Average Firing Rate
* Spike Sparsity

These metrics allow evaluation of **accuracy vs efficiency trade-offs**.

---

## ⚡ Hardware & VLSI-Aware Analysis

To bridge algorithm design with hardware feasibility:

* MAC vs AC operation comparison
* Memory access estimation
* Temporal compute overhead
* Spike-driven sparsity analysis

This provides insights into neuromorphic hardware efficiency without requiring full RTL implementation.

---

## 🔌 Analog Neuromorphic Interpretation

LIF neuron behavior maps to analog primitives:

| SNN Component      | Analog Equivalent           |
| ------------------ | --------------------------- |
| Membrane Potential | Capacitor Voltage           |
| Synaptic Input     | Current Injection           |
| Leakage            | Resistive/Subthreshold Leak |
| Thresholding       | Comparator                  |
| Spike              | Digital Pulse               |
| Reset              | Capacitor Discharge         |

Temporal spike sparsity implies reduced switching activity and analog energy consumption.

---

## 🛠️ Tech Stack

* PyTorch
* snnTorch
* SpikingJelly
* CUDA (GPU training)
* NumPy
* Matplotlib
* ONNX
* TensorFlow Lite

---

## 🧪 Training Methodology

Training incorporates:

* Surrogate gradient learning
* Adam optimizer
* Temporal spike accumulation
* Backpropagation through time

Optional enhancements:

* BatchNorm Through Time (BNTT)
* Spike regularization
* Threshold tuning

---

## 🔄 Deployment Pipeline

```
PyTorch SNN
     ↓
ONNX Export
     ↓
TensorFlow Lite Conversion
     ↓
Edge Inference Evaluation
```

This enables compatibility testing with lightweight inference environments.

---

## 🧭 Neuromorphic Hardware Outlook

Future hardware mapping targets include:

* Intel Loihi
* FPGA-based neuromorphic accelerators
* Mixed-signal analog SNN implementations

These explorations focus on translating spike-driven models to hardware substrates.

---

## 📡 Event-Based Vision Extension

Planned dataset expansion:

* DVS Gesture Dataset
* Event-driven temporal encoding
* Frame vs event batching comparison

This will enable evaluation on true neuromorphic sensory data.

---

## 📂 Repository Structure

```
├── lif_basics/              # Single neuron simulations
├── snn_models/              # Network architectures
├── surrogate_training/     # Gradient-based learning
├── experiments/             # Timestep trade-off studies
├── metrics/                 # Spike & latency logging
├── deployment/              # ONNX & TFLite export
├── hardware_analysis/       # VLSI & analog evaluation
└── README.md
```

---

## 📈 Project Status

* ✅ LIF neuron modeling
* ✅ SNN architecture implementation
* ⏳ Surrogate gradient training
* ⏳ Temporal trade-off experiments
* ⏳ Hardware-aware analysis
* ⏳ Deployment benchmarking
* ⏳ DVS dataset integration

Estimated completion: ~70%

---

## 🤝 Future Extensions

* SpikingJelly comparative frameworks
* Quantization-aware SNN training
* Mixed-signal neuromorphic mapping
* Event-driven real-time inference
* Edge deployment benchmarking

---

## 📚 References

Key inspirations include:

* Neuromorphic computing literature
* Surrogate gradient SNN training
* Edge AI deployment workflows
* Hardware-aware neural network design

Full citations will accompany the research publication.

---

## 👨‍💻 Author

Advait Rao
Electronics & Engineering
Neuromorphic Systems • Edge AI • Hardware-Aware AI

---

## 📜 License

Released under the MIT License.
