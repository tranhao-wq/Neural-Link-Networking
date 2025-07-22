# Neural Link Networking 🧠⚡

A real-time neural network visualization tool that demonstrates the training process of a feed-forward neural network for geometric shape classification. Watch your AI learn in real-time!

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This application creates a live, animated visualization of a neural network learning to classify simple geometric shapes (circles, squares, triangles). The visualization shows real-time neuron activations, training metrics, and the learning process as it happens.

## ✨ Features

- **Real-time Training Visualization**: Watch neurons activate and learn in real-time
- **Interactive Network Diagram**: Color-coded neurons showing activation levels
- **Live Metrics**: Cost and accuracy plots updating with each training batch
- **Synthetic Data Generation**: Automatically generates training data with shape variations
- **Educational Tool**: Perfect for understanding how neural networks learn

## 🖼️ Application Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Batch: 143 Example: 567                                 │
├──────────────┬──────────────────────────────────────────────────────────────┤
│   Cost       │                                                              │
│    2.0 ┌─────┤                    Neural Network Diagram                    │
│        │     │                                                              │
│    1.0 │  ╲  │    ●──●──●──● Circle                                         │
│        │   ╲ │   ╱│╲╱│╲╱│╲╱                                                 │
│    0.0 └─────┤  ● │ ● │ ● │ ● Square                                        │
│   0  200 400 │   ╲│╱╲│╱╲│╱╲│                                                │
├──────────────┤    ●──●──●──● Triangle                                       │
│  Accuracy %  │   ╱│╲╱│╲╱│╲╱                                                 │
│    1.0 ┌─────┤  ● │ ● │ ● │ ● I don't know                                  │
│        │  ╱──│   ╲│╱╲│╱╲│╱╲│                                                │
│    0.5 │ ╱   │    ●──●──●──●                                                │
│        │╱    │                                                              │
│    0.0 └─────┤  Input  Hidden Hidden Output                                 │
│   0  200 400 │  Layer   L1     L2    Layer                                  │
├──────────────┤                                                              │
│ Input Image  │  ● Yellow = High Activation                                  │
│  ┌────────┐  │  ● Orange = Medium Activation                                │
│  │   ○    │  │  ● Red = Low Activation                                      │
│  │        │  │  ✗ = Inactive Neuron                                         │
│  │        │  │                                                              │
│  └────────┘  │                                                              │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch matplotlib pillow numpy
```

### Installation & Running

1. **Clone the repository:**
```bash
git clone https://github.com/tranhao-wq/Neural-Link-Networking.git
cd Neural-Link-Networking
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python neural_network_visualizer.py
```

## 🏗️ Architecture

### Neural Network Structure
```
Input Layer:    1600 neurons (40x40 flattened image)
                    ↓
Hidden Layer 1:   12 neurons (ReLU activation)
                    ↓
Hidden Layer 2:   12 neurons (ReLU activation)
                    ↓
Output Layer:      4 neurons (Softmax activation)
                    ↓
Classes: [Circle, Square, Triangle, "I don't know"]
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32 images
- **Data**: Synthetically generated 40x40 grayscale images

## 📊 Visualization Components

### 1. Cost Plot (Top-Left)
- **X-axis**: Training iterations
- **Y-axis**: Loss value (0.0 - 2.0)
- **Updates**: After each training batch

### 2. Accuracy Plot (Middle-Left)
- **X-axis**: Training iterations  
- **Y-axis**: Accuracy percentage (0.0 - 1.0)
- **Updates**: After each training batch

### 3. Input Image Display (Bottom-Left)
- **Size**: 40x40 pixels
- **Format**: Grayscale
- **Content**: Current example being processed

### 4. Neural Network Diagram (Right)
- **Neurons**: Color-coded by activation level
  - 🟡 **Yellow**: High activation (>0.3)
  - 🟠 **Orange**: Medium activation (0.05-0.3)
  - 🔴 **Red**: Low activation (<0.05)
  - ❌ **X Mark**: Very low activation (<0.05)
- **Connections**: Lines between all neurons in adjacent layers
- **Labels**: Output neurons labeled with class names
- **Values**: Numerical activation values displayed for select neurons

## 🎨 Shape Generation

The application generates four types of training data:

### Circle
```
    ●●●●●
  ●       ●
 ●         ●
●           ●
 ●         ●
  ●       ●
    ●●●●●
```

### Square
```
●●●●●●●●●
●         ●
●         ●
●         ●
●         ●
●         ●
●●●●●●●●●
```

### Triangle
```
      ●
     ● ●
    ●   ●
   ●     ●
  ●       ●
 ●●●●●●●●●
```

### "I don't know" (Random Noise)
```
  ●   ●     ●
●   ●   ●
    ●     ●   ●
●     ●
  ●       ●
```

## 🔧 Customization

### Modify Network Architecture
```python
# In ShapeClassifier.__init__()
self.fc1 = nn.Linear(1600, 24)  # Increase hidden layer size
self.fc2 = nn.Linear(24, 24)
self.fc3 = nn.Linear(24, 4)
```

### Adjust Training Parameters
```python
# In NeuralNetworkVisualizer.__init__()
self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # Change learning rate
```

### Modify Visualization Colors
```python
# In get_neuron_color()
if activation < 0.05:
    return 'blue'      # Change low activation color
elif activation < 0.3:
    return 'green'     # Change medium activation color
else:
    return 'red'       # Change high activation color
```

## 📈 Performance Metrics

- **Training Speed**: ~10-20 batches per second
- **Convergence**: Typically reaches >90% accuracy within 100-200 iterations
- **Memory Usage**: ~50-100MB RAM
- **CPU Usage**: Moderate (single-threaded)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Tran The Hao**
- GitHub: [@tranhao-wq](https://github.com/tranhao-wq)
- Project Link: [https://github.com/tranhao-wq/Neural-Link-Networking](https://github.com/tranhao-wq/Neural-Link-Networking)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Matplotlib developers for powerful visualization tools
- The open-source community for inspiration and support

## 📚 Educational Use

This project is perfect for:
- **Students** learning about neural networks
- **Educators** demonstrating AI concepts
- **Researchers** prototyping visualization techniques
- **Developers** understanding real-time ML visualization

---

⭐ **Star this repository if you found it helpful!**