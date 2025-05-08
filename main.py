
import numpy as np
from neuralNetwork import NN
import pprint
from matplotlib.animation import FuncAnimation

def generate_noisy_data(func, x_start, x_end, num_points, noise_level):
    x_values = np.linspace(x_start, x_end, num_points)
    y_values = func(x_values)
    noise = np.random.normal(0, noise_level, size=y_values.shape)
    noisy_y_values = y_values + noise
    x_array = x_values.reshape(-1, 1)
    y_array = noisy_y_values.reshape(-1, 1)
    return x_array, y_array

def two_times_x_plus_one(x):
    return 2 * x + 1

def x_squared(x):
    return x ** 2

EPOCHS = 10000

if __name__ == "__main__":
    nn = NN()
    nn.addInputLayer(1)
    nn.addHiddenLayer(10)
    nn.addHiddenLayer(10)
    nn.addOutputLayer(1)
    
    x, y = generate_noisy_data(x_squared, 0, 10, 100, 0.5)
    
    inputs = np.array(x, dtype=float)
    targets = np.array(y, dtype=float)

    nn.train(inputs, targets, epochs=EPOCHS)
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Target values')
    line, = ax.plot([], [], color='red', label='Estimated values')
    ax.legend()
    ax.set_title("Training Progress")
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output (y)")

    def init():
        line.set_data([], [])
        return line,

    def update(epoch):
        estimated_values = nn.trainingHistory[epoch]
        line.set_data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], estimated_values)
        ax.set_title(f"Training Progress - Epoch {epoch + 1}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(nn.trainingHistory), init_func=init, blit=True, repeat=False, interval=10000 / len(nn.trainingHistory))
    plt.show()
    