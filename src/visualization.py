import matplotlib.pyplot as plt


def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Training Accuracy")

    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig("accuracy_graph.png")

    plt.show()


def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Training Loss")

    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("loss_graph.png")

    plt.show()