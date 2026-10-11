"""A minimal two-layer neural network trained on XOR with batch backpropagation."""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class XorNetwork:
    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_weights = rng.normal(0.0, 0.8, size=(3, 4))
        self.output_weights = rng.normal(0.0, 0.8, size=(5, 1))

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        biased_input = np.column_stack((features, np.ones(len(features))))
        hidden = sigmoid(biased_input @ self.hidden_weights)
        biased_hidden = np.column_stack((hidden, np.ones(len(hidden))))
        output = sigmoid(biased_hidden @ self.output_weights)
        return hidden, output

    def train(self, features: np.ndarray, targets: np.ndarray, *, epochs: int = 12_000, rate: float = 0.8) -> float:
        biased_input = np.column_stack((features, np.ones(len(features))))
        for _ in range(epochs):
            hidden, output = self.forward(features)
            output_delta = (output - targets) * output * (1.0 - output)
            biased_hidden = np.column_stack((hidden, np.ones(len(hidden))))
            hidden_delta = (output_delta @ self.output_weights[:-1].T) * hidden * (1.0 - hidden)

            self.output_weights -= rate * (biased_hidden.T @ output_delta) / len(features)
            self.hidden_weights -= rate * (biased_input.T @ hidden_delta) / len(features)

        _, output = self.forward(features)
        return float(np.mean((targets - output) ** 2))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.forward(features)[1] >= 0.5).astype(int)


if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    network = XorNetwork()
    loss = network.train(x, y)
    predictions = network.predict(x)
    assert np.array_equal(predictions, y.astype(int))
    print(f"xor loss: {loss:.6f}")
    print("predictions:", predictions.ravel().tolist())
