import numpy as np
import struct
import tkinter as tk
from tkinter import Canvas, Button, ALL
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

# Constants
INPUT_SIZE = 784  
HIDDEN_SIZE = 128  
OUTPUT_SIZE = 10 
RANDOM_SEED = 42
NUM_CLASSES = 10

# MNIST Data Loading
def load_images(filename):
    with open(filename, "rb") as f:
        data = f.read()
        _, num_images, rows, cols = struct.unpack(">IIII", data[:16])
        images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows * cols)
        images = images.astype(np.float32) / 255.0  # Normalize
    return images

def load_labels(filename):
    with open(filename, "rb") as f:
        data = f.read()
        _, num_labels = struct.unpack(">II", data[:8])
        labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

# Neural Network Initializations and Definitions
def initialize_network_parameters():
    np.random.seed(RANDOM_SEED)
    W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2. / INPUT_SIZE)
    b1 = np.zeros((1, HIDDEN_SIZE))
    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2. / HIDDEN_SIZE)
    b2 = np.zeros((1, OUTPUT_SIZE))
    return W1, b1, W2, b2

W1, b1, W2, b2 = initialize_network_parameters()

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(Z.dtype)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=1, keepdims=True)

def cross_entropy_loss(Y_true, Y_pred):
    m = Y_true.shape[0]
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
    return loss

def cross_entropy_loss_derivative(Y_true, Y_pred):
    return Y_pred - Y_true

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[0]
    
    dZ2 = cross_entropy_loss_derivative(Y, A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

def update_parameters(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

def train_neural_network(X, Y, learning_rate=0.1, epochs=100):
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2)
        update_parameters(dW1, db1, dW2, db2, learning_rate)
        if epoch % 10 == 0:
            loss = cross_entropy_loss(Y, A2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
def save_weights(filename='weights.npz'):
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)

def load_weights(filename='weights.npz'):
    global W1, b1, W2, b2
    weights = np.load(filename)
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']


class DigitsPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Digits Predictor")

        # Canvas setup
        self.canvas = Canvas(self.root, bg="white", width=280, height=280)
        self.canvas.pack(pady=20)

        # Brush settings
        self.brush_color = "black"
        self.brush_width = 10
        self.eraser_width = 30
        self.last_x, self.last_y = None, None
        
        # Store the image and initialize the ImageDraw object
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind drawing methods to mouse actions
        self.canvas.bind("<Button-1>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons setup
        self.pen_button = Button(self.root, text="Lápiz", command=self.use_pen)
        self.pen_button.pack(side=tk.LEFT, padx=10)

        self.eraser_button = Button(self.root, text="Goma", command=self.use_eraser)
        self.eraser_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = Button(self.root, text="Predecir", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = Button(self.root, text="Limpiar", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)

    def start_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval((x-2, y-2, x+2, y+2), fill=self.brush_color, width=self.brush_width)
        self.draw.line([(self.last_x, self.last_y), (x, y)], fill=self.brush_color, width=self.brush_width)
        self.last_x, self.last_y = x, y

    def use_pen(self):
        self.brush_color = "black"
        self.brush_width = 10

    def use_eraser(self):
        self.brush_color = "white"
        self.brush_width = self.eraser_width

    def predict_digit(self):
        # Convert the drawing into a 28x28 image
        img = self.image.copy()
        img = img.resize((28, 28)).convert('L')
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        img_data = np.array(img)
        img_data = img_data.reshape(1, 28*28)
        img_data = 1 - img_data.astype(np.float32) / 255.0  # Invert colors and normalize

        # Use the neural network to predict the digit
        _, _, _, predictions = forward_propagation(img_data)
        predicted_digit = np.argmax(predictions)

        # Show the prediction
        self.root.title(f"Predicted: {predicted_digit}")

    def clear_canvas(self):
        self.canvas.delete(ALL)
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.root.title("Digits Predictor")

if __name__ == '__main__':
    # Load MNIST data
    path = input("Ingrese la ruta de la carpeta MNIST DATABASE: ")
    X_train = load_images(path + "/train-images.idx3-ubyte")
    y_train = load_labels(path + "/train-labels.idx1-ubyte")
    y_train_one_hot = np.eye(NUM_CLASSES)[y_train]
    
    # Ask user to either train the network or load existing weights
    choice = input("¿Desea entrenar la red neuronal (E) o cargar los pesos existentes (C)? ").lower()
    
    if choice == 'e':
        # Train the neural network
        train_neural_network(X_train, y_train_one_hot)
        save_weights()
    elif choice == 'c':
        try:
            load_weights()
            print("Pesos cargados exitosamente.")
        except FileNotFoundError:
            print("Archivo de pesos no encontrado. Entrenando red neuronal...")
            train_neural_network(X_train, y_train_one_hot)
            save_weights()
    
    # Run the GUI application
    root = tk.Tk()
    app = DigitsPredictor(root)
    root.mainloop()

