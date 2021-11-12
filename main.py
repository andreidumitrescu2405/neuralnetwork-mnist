from model import NeuralNetwork
import torchvision
import torch
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Our model
    model = NeuralNetwork()

    # Loss function
    loss = torch.nn.NLLLoss()

    # Hyperparametrii 
    lr = 0.03
    epochs = 5

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)

    # Transforms
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Datasets
    dataset_train = torchvision.datasets.MNIST("./files", train=True, download=True, transform=transforms)
    dataset_valid = torchvision.datasets.MNIST("./files", train=False, download=True, transform=transforms)

    # Dataloader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=32)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, shuffle=False, batch_size=1)

    # Training
    errors_train = []
    errors_valid = []

    # Training loop
    for epoch in range(epochs):
        errors_temporal = []
        model.train()
        for image, label in dataloader_train:
            # Flatten
            image = image.view(32, 784)

            # Clear gradients
            optimizer.zero_grad()

            # Prediction
            pred = model(image)

            # Calcul erorii
            error = loss(pred, label)
            errors_temporal.append(error.item())

            # Compute gradients
            error.backward()

            # Update weights
            optimizer.step()

        # Validation
        errors_temporal_valid = []
        for image, label in dataloader_valid:
            # Flatten
            image = image.view(1, 784)

            # Prediction
            pred = model(image)

            # Calcul erorii
            error = loss(pred, label)
            errors_temporal_valid.append(error.item())

        # Metrics
        error_medie_per_epoca = sum(errors_temporal) / len(errors_temporal)
        error_medie_per_epoca_valid = sum(errors_temporal_valid) / len(errors_temporal_valid)

        errors_train.append(error_medie_per_epoca)
        errors_valid.append(error_medie_per_epoca_valid)

        print(f"Epoch: {epoch}. Error per epochs: {error_medie_per_epoca}. Error per epoch valid: {error_medie_per_epoca_valid}")

    plt.plot(errors_train, label="Error train")
    plt.plot(errors_valid, label="Error validation")
    plt.legend()
    plt.show()






