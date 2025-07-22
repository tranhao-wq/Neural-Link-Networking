import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw
import random
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier, self).__init__()
        self.fc1 = nn.Linear(1600, 12)  # 40x40 = 1600 input neurons
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 4)     # 4 output classes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.view(-1, 1600)  # Flatten
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.softmax(self.fc3(x2))
        return x3, [x.view(-1, 1600), x1, x2, x3]  # Return activations for visualization

class DataGenerator:
    def __init__(self):
        self.classes = ['Circle', 'Square', 'Triangle', 'I dont know']
        
    def generate_shape(self, shape_type, size=40):
        img = Image.new('L', (size, size), 0)  # Black background
        draw = ImageDraw.Draw(img)
        
        # Add some randomness
        center_x = random.randint(size//4, 3*size//4)
        center_y = random.randint(size//4, 3*size//4)
        radius = random.randint(8, 15)
        thickness = random.randint(2, 4)
        
        if shape_type == 0:  # Circle
            bbox = [center_x-radius, center_y-radius, center_x+radius, center_y+radius]
            draw.ellipse(bbox, outline=255, width=thickness)
        elif shape_type == 1:  # Square
            half_size = radius
            bbox = [center_x-half_size, center_y-half_size, center_x+half_size, center_y+half_size]
            draw.rectangle(bbox, outline=255, width=thickness)
        elif shape_type == 2:  # Triangle
            points = [
                (center_x, center_y-radius),
                (center_x-radius, center_y+radius//2),
                (center_x+radius, center_y+radius//2)
            ]
            draw.polygon(points, outline=255, width=thickness)
        else:  # Random noise
            for _ in range(random.randint(50, 150)):
                x, y = random.randint(0, size-1), random.randint(0, size-1)
                draw.point((x, y), fill=random.randint(100, 255))
        
        return np.array(img) / 255.0  # Normalize to [0, 1]
    
    def generate_batch(self, batch_size=32):
        images = []
        labels = []
        for _ in range(batch_size):
            shape_type = random.randint(0, 3)
            img = self.generate_shape(shape_type)
            images.append(img)
            labels.append(shape_type)
        
        return torch.FloatTensor(images), torch.LongTensor(labels)

class NeuralNetworkVisualizer:
    def __init__(self):
        self.model = ShapeClassifier()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data_generator = DataGenerator()
        
        # Training metrics
        self.costs = []
        self.accuracies = []
        self.batch_num = 0
        self.example_num = 0
        
        # Current visualization data
        self.current_image = None
        self.current_activations = None
        
        self.setup_plot()
        
    def setup_plot(self):
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create subplots
        self.ax_cost = plt.subplot2grid((3, 4), (0, 0))
        self.ax_accuracy = plt.subplot2grid((3, 4), (1, 0))
        self.ax_image = plt.subplot2grid((3, 4), (2, 0))
        self.ax_network = plt.subplot2grid((3, 4), (0, 1), colspan=3, rowspan=3)
        
        # Setup cost plot
        self.ax_cost.set_title('Cost')
        self.ax_cost.set_xlim(0, 500)
        self.ax_cost.set_ylim(0, 2.0)
        self.ax_cost.set_xlabel('Iterations')
        self.ax_cost.set_ylabel('cost')
        
        # Setup accuracy plot
        self.ax_accuracy.set_title('Accuracy %')
        self.ax_accuracy.set_xlim(0, 500)
        self.ax_accuracy.set_ylim(0, 1.0)
        self.ax_accuracy.set_xlabel('Iterations')
        self.ax_accuracy.set_ylabel('accuracy')
        
        # Setup image display
        self.ax_image.set_xlim(0, 40)
        self.ax_image.set_ylim(0, 40)
        self.ax_image.set_aspect('equal')
        
        # Setup network diagram
        self.ax_network.set_xlim(-1, 4)
        self.ax_network.set_ylim(-1, 17)
        self.ax_network.set_aspect('equal')
        self.ax_network.axis('off')
        
        plt.tight_layout()
        
    def get_neuron_color(self, activation):
        # Red (low) -> Orange -> Yellow (high)
        if activation < 0.05:
            return 'red'
        elif activation < 0.3:
            return 'orange'
        else:
            return 'yellow'
    
    def draw_network(self):
        self.ax_network.clear()
        self.ax_network.set_xlim(-1, 4)
        self.ax_network.set_ylim(-1, 17)
        self.ax_network.axis('off')
        
        if self.current_activations is None:
            return
            
        layer_sizes = [16, 12, 12, 4]  # Simplified input visualization
        layer_positions = [0, 1, 2, 3]
        
        # Draw connections first (behind neurons)
        for i in range(len(layer_sizes) - 1):
            for j in range(layer_sizes[i]):
                for k in range(layer_sizes[i + 1]):
                    y1 = j * (16 / layer_sizes[i])
                    y2 = k * (16 / layer_sizes[i + 1])
                    self.ax_network.plot([layer_positions[i], layer_positions[i + 1]], 
                                       [y1, y2], 'k-', alpha=0.1, linewidth=0.5)
        
        # Draw neurons
        for layer_idx, (size, x_pos) in enumerate(zip(layer_sizes, layer_positions)):
            activations = self.current_activations[layer_idx] if layer_idx < len(self.current_activations) else None
            
            for neuron_idx in range(size):
                y_pos = neuron_idx * (16 / size)
                
                if activations is not None and layer_idx > 0:
                    if layer_idx == 1:  # First hidden layer
                        activation = activations[0, neuron_idx].item()
                    elif layer_idx == 2:  # Second hidden layer  
                        activation = activations[0, neuron_idx].item()
                    else:  # Output layer
                        activation = activations[0, neuron_idx].item()
                else:
                    activation = 0.5  # Default for input layer visualization
                
                color = self.get_neuron_color(activation)
                circle = Circle((x_pos, y_pos), 0.15, color=color, ec='black')
                self.ax_network.add_patch(circle)
                
                # Add X for very low activation
                if activation < 0.05 and layer_idx > 0:
                    self.ax_network.text(x_pos, y_pos, 'X', ha='center', va='center', 
                                       fontsize=8, fontweight='bold')
                
                # Add activation values for some neurons
                if layer_idx > 0 and neuron_idx % 3 == 0:
                    self.ax_network.text(x_pos + 0.25, y_pos, f'{activation:.4f}', 
                                       fontsize=8, va='center')
        
        # Add output labels
        labels = ['Circle', 'Square', 'Triangle', 'I dont know']
        for i, label in enumerate(labels):
            y_pos = i * (16 / 4)
            self.ax_network.text(3.5, y_pos, label, fontsize=10, va='center')
    
    def train_step(self):
        # Generate batch
        images, labels = self.data_generator.generate_batch(32)
        
        # Training step
        self.optimizer.zero_grad()
        outputs, _ = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item()
        
        # Store metrics
        self.costs.append(loss.item())
        self.accuracies.append(accuracy)
        
        # Get single example for visualization
        single_image = images[0:1]
        self.current_image = single_image[0].numpy()
        
        # Get activations for visualization
        with torch.no_grad():
            _, activations = self.model(single_image)
            self.current_activations = activations
        
        self.batch_num += 1
        self.example_num = (self.example_num + 1) % 1000
    
    def update_plots(self):
        # Update cost plot
        self.ax_cost.clear()
        self.ax_cost.set_title('Cost')
        self.ax_cost.set_xlim(0, max(500, len(self.costs)))
        self.ax_cost.set_ylim(0, 2.0)
        self.ax_cost.set_xlabel('Iterations')
        self.ax_cost.set_ylabel('cost')
        if self.costs:
            self.ax_cost.plot(self.costs, 'b-')
        
        # Update accuracy plot
        self.ax_accuracy.clear()
        self.ax_accuracy.set_title('Accuracy %')
        self.ax_accuracy.set_xlim(0, max(500, len(self.accuracies)))
        self.ax_accuracy.set_ylim(0, 1.0)
        self.ax_accuracy.set_xlabel('Iterations')
        self.ax_accuracy.set_ylabel('accuracy')
        if self.accuracies:
            self.ax_accuracy.plot(self.accuracies, 'g-')
        
        # Update image display
        self.ax_image.clear()
        if self.current_image is not None:
            self.ax_image.imshow(self.current_image, cmap='gray', origin='upper')
        self.ax_image.set_xlim(0, 40)
        self.ax_image.set_ylim(0, 40)
        
        # Update status text
        self.fig.suptitle(f'Batch: {self.batch_num} Example: {self.example_num}', 
                         fontsize=14, y=0.95)
    
    def animate(self, frame):
        self.train_step()
        self.update_plots()
        self.draw_network()
        return []
    
    def run(self):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=100, blit=False)
        plt.show()
        return ani

if __name__ == "__main__":
    visualizer = NeuralNetworkVisualizer()
    ani = visualizer.run()