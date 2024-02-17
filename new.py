import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd

class SimpleCNN:
    def __init__(self, num_classes):
        # Initialize weights and biases for the layers
        self.conv1_weights = np.random.randn(32, 1, 3, 3) / np.sqrt(32 * 3 * 3)
        self.conv1_bias = np.zeros((32, 1))

        self.conv2_weights = np.random.randn(64, 32, 3, 3) / np.sqrt(64 * 3 * 3)
        self.conv2_bias = np.zeros((64, 1))

        self.fc1_weights = np.random.randn(128, 64 * 7 * 7) / np.sqrt(128 * 7 * 7)
        self.fc1_bias = np.zeros((128, 1))

        self.fc2_weights = np.random.randn(num_classes, 128) / np.sqrt(128)
        self.fc2_bias = np.zeros((num_classes, 1))

    def forward(self, x):
        # Convolutional layer 1
        print("Input shape:", x.shape)
        conv1_output = self.convolution(x, self.conv1_weights, self.conv1_bias)
        print("Conv1 output shape:", conv1_output.shape)
        relu1_output = self.relu(conv1_output)
        pool1_output = self.max_pooling(relu1_output)

        # Convolutional layer 2
        conv2_output = self.convolution(pool1_output, self.conv2_weights, self.conv2_bias)
        print("Conv2 output shape:", conv2_output.shape)
        relu2_output = self.relu(conv2_output)
        pool2_output = self.max_pooling(relu2_output)

        # Flatten and fully connected layer 1
        fc1_input = pool2_output.reshape(pool2_output.shape[0], -1)
        print("FC1 input shape:", fc1_input.shape)
        fc1_output = self.fc_layer(fc1_input, self.fc1_weights, self.fc1_bias)
        relu3_output = self.relu(fc1_output)

        # Fully connected layer 2
        fc2_output = self.fc_layer(relu3_output, self.fc2_weights, self.fc2_bias)

        return fc2_output

    def convolution(self, x, weights, bias):
        _, _, height, width = x.shape
        _, _, kernel_size, _ = weights.shape
        output_height = height - kernel_size + 1
        output_width = width - kernel_size + 1

        unfolded_input = self.unfold(x, kernel_size)
        unfolded_weights = weights.reshape(weights.shape[0], -1)

        # Check if dimensions are compatible
        assert unfolded_input.shape[1] == unfolded_weights.shape[0], "Incompatible dimensions for convolution"

        conv_output = (unfolded_input @ unfolded_weights.T + bias).reshape(weights.shape[0], output_height, output_width)

        return conv_output



    def unfold(self, x, kernel_size):
        _, _, height, width = x.shape
        unfolded_input = np.lib.stride_tricks.sliding_window_view(x, (1, 1, kernel_size, kernel_size)).reshape(-1, kernel_size * kernel_size)
        return unfolded_input

    def max_pooling(self, x):
        _, _, height, width = x.shape
        pool_size = 2
        output_height = height // pool_size
        output_width = width // pool_size

        unfolded_input = self.unfold(x, pool_size)
        pooled_output = np.max(unfolded_input, axis=1).reshape(-1, output_height, output_width)

        return pooled_output

    def fc_layer(self, x, weights, bias):
        return (weights @ x.T + bias).T

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Load and preprocess the data using pandas
def load_dataset(root='data/train', transform=None):
    data = {'path': [], 'label': []}
    for class_label, class_name in enumerate(os.listdir(root)):
        class_path = os.path.join(root, class_name)
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            data['path'].append(image_path)
            data['label'].append(class_label)
    df = pd.DataFrame(data)
    return df

# Training the model
def train_model(model, df, num_epochs, learning_rate, batch_size):
    class_labels = {}  # Dictionary to store class index and corresponding name
    total_batches = len(df) // batch_size
    for epoch in range(num_epochs):
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_data = df.iloc[start_idx:end_idx]

            images = []
            labels = []

            for i, row in batch_data.iterrows():
                try:
                    image = Image.open(row['path']).convert('L')
                    label = row['label']
                    images.append(image)
                    labels.append(label)
                    # Update class_labels dictionary
                    class_labels[label] = get_class_name(label)
                except UnidentifiedImageError:
                    print(f"Corrupt image detected: {row['path']}. Skipping.")

            if len(images) == 0:
                continue  # Skip empty batches

            # Resize images to a common size (e.g., 28x28)
            images = [img.resize((28, 28)) for img in images]
            # Convert the list of images to a numpy array
            images = np.array([np.array(img) for img in images]).reshape(-1, 1, 28, 28)
            labels = np.array(labels)


            # Normalize images to [0, 1]
            images = images / 255.0

            model.forward(images)
            loss = cross_entropy_loss(model.fc2_output, labels)

            # Backpropagation (gradient descent)
            gradients = backward_pass(model, labels)
            update_weights(model, gradients, learning_rate)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}")

    # Save the trained model and class labels
    np.save('class_labels.npy', class_labels)
    save_model(model, path='trained_model.npy')

def cross_entropy_loss(predictions, targets):
    # Convert targets to one-hot encoding
    one_hot_targets = np.eye(num_classes)[targets]

    # Calculate cross-entropy loss
    loss = -np.sum(one_hot_targets * np.log(predictions + 1e-15)) / len(targets)

    return loss

def backward_pass(model, targets):
    # Calculate gradients for the weights and biases
    gradients = {}

    # Gradients for fully connected layer 2
    gradients['fc2_weights'] = (model.softmax(model.fc2_output) - np.eye(num_classes)[targets]).T @ model.relu(model.fc1_output)
    gradients['fc2_bias'] = np.sum(model.softmax(model.fc2_output) - np.eye(num_classes)[targets], axis=0, keepdims=True).T

    # Gradients for fully connected layer 1
    gradients['fc1_weights'] = (model.softmax(model.fc2_output) - np.eye(num_classes)[targets]) @ model.fc2_weights
    gradients['fc1_weights'] = gradients['fc1_weights'] * (model.fc1_input > 0)
    gradients['fc1_weights'] = gradients['fc1_weights'].T @ model.fc1_input.reshape(-1, 1)

    gradients['fc1_bias'] = np.sum((model.softmax(model.fc2_output) - np.eye(num_classes)[targets]) @ model.fc2_weights, axis=0, keepdims=True).T

    return gradients

def update_weights(model, gradients, learning_rate):
    # Update weights and biases using gradient descent
    model.fc2_weights -= learning_rate * gradients['fc2_weights']
    model.fc2_bias -= learning_rate * gradients['fc2_bias']

    model.fc1_weights -= learning_rate * gradients['fc1_weights']
    model.fc1_bias -= learning_rate * gradients['fc1_bias']

# Save the trained model
def save_model(model, path='trained_model.npy'):
    np.save(path, model.__dict__)

# Load the trained model
def load_model(path='trained_model.npy'):
    return SimpleCNN(0).__dict__.update(np.load(path, allow_pickle=True).item())

# Test the model
def test_model(model, image_path, class_labels):
    try:
        image = np.array(Image.open(image_path).convert('L'))
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        image = image / 255.0  # Normalize to [0, 1]

        output = model.forward(image)
        probabilities = model.softmax(output)
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[0, predicted_class]
        class_name = class_labels.get(predicted_class, "Unknown")

        print(f"Predicted Class: {predicted_class} - {class_name}")
        print(f"Prediction Confidence: {confidence * 100:.2f}%")
    except UnidentifiedImageError:
        print(f"Corrupt image detected: {image_path}. Cannot make a prediction.")

def get_class_name(class_index):
    # Function to get class name from class index
    return f"Class_{class_index}"

# Initialize the model


# ... (Rest of the code remains the same)
# Load and preprocess the data using pandas
train_dataset = load_dataset(root='data/train')

# Initialize the model
num_classes = len(np.unique(train_dataset['label']))
model = SimpleCNN(num_classes)

# Ask user for choice
choice = input("Do you want to train or test the model? (Type 'train' or 'test'): ")

if choice.lower() == 'train':
    df = load_dataset(root='data/train')
    custom_epochs = int(input("Enter the number of epochs: "))
    learning_rate = float(input("Enter the learning rate: "))
    batch_size = int(input("Enter the batch size: "))
    train_model(model, df, custom_epochs, learning_rate, batch_size)
    save_model(model, path='trained_model.npy')

elif choice.lower() == 'test':
    model = load_model(path='trained_model.npy')
    class_labels = np.load('class_labels.npy', allow_pickle=True).item()
    image_path = input("Enter the path of the image you want to test: ")
    test_model(model, image_path, class_labels)

else:
    print("Invalid choice. Please type 'train' or 'test'")

