import matplotlib.pyplot as plt
from backprop_network import *
from backprop_data import *


def b():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rates = [0.001, 0.01, 0.1, 1, 10]

    # Network configuration
    layer_dims = [784, 40, 10]

    # Training
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    for learning_rate in learning_rates:
        net = Network(layer_dims)
        res = net.train(
            x_train,
            y_train,
            epochs,
            batch_size,
            learning_rate,
            x_test,
            y_test,
        )
        train_accuracies.append(res[3])
        train_losses.append(res[1])
        test_accuracies.append(res[4])
    epochs = [i for i in range(1, epochs + 1)]

    # Train accuracy plotting
    for i, rate in enumerate(learning_rates):
        plt.plot(epochs, train_accuracies[i], label=f"Learning rate: {rate}")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.show()

    # Train loss plotting
    for i, rate in enumerate(learning_rates):
        plt.plot(epochs, train_losses[i], label=f"Learning rate: {rate}")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.show()

    # Test accuracy plotting
    for i, rate in enumerate(learning_rates):
        plt.plot(epochs, test_accuracies[i], label=f"Learning rate: {rate}")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()


def c():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(50000, 10000)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 40, 10]

    # Training
    net = Network(layer_dims)
    res = net.train(
        x_train,
        y_train,
        epochs,
        batch_size,
        learning_rate,
        x_test,
        y_test,
    )

    test_accuracy = res[4][-1]
    print(f"Final epoch test accuracy using learning rate 0.1: {test_accuracy:.2f}")


def d():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(50000, 10000)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 10]

    # Training
    net = Network(layer_dims)
    res = net.train(
        x_train,
        y_train,
        epochs,
        batch_size,
        learning_rate,
        x_test,
        y_test,
    )

    # Train accuracy plotting
    plt.plot([i for i in range(1, epochs + 1)], res[3])
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.show()

    # Test accuracy plotting
    plt.plot([i for i in range(1, epochs + 1)], res[4])
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.show()

    # Weights plotting
    for i, row in enumerate(res[0]["W1"]):
        plt.imshow(row.reshape(28, 28), interpolation="nearest")
        plt.title(f"W[{i}]")
        plt.show()
