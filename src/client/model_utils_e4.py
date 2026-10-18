import torch
import torch.nn as nn


class FashionMNISTCNN(nn.Module):
    """
    Small CNN for the independent Fashion-MNIST FL benchmark.

    Input:
        [batch, 1, 28, 28]

    Output:
        logits for 10 Fashion-MNIST classes
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    model = FashionMNISTCNN()

    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)

    print("Fashion-MNIST E1 model validation")
    print("---------------------------------")
    print("Input shape :", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters  :", count_parameters(model))

    assert output.shape == (8, 10)

    print()
    print("Model validation: PASS")
