from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPSILON = 1e-9


@dataclass(slots=True)
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < EPSILON] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _relu_grad(values: np.ndarray) -> np.ndarray:
    return (values > 0.0).astype(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / (exps.sum(axis=1, keepdims=True) + EPSILON)


class _BaseMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        output_dim: int,
        learning_rate: float,
        l2: float,
        seed: int,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scaler = StandardScaler()
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.loss_history: list[float] = []
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        layer_sizes = [self.input_dim, *self.hidden_sizes, self.output_dim]
        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / max(fan_in, 1))
            weight = self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            bias = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

    def _forward(self, values: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [values]
        pre_activations: list[np.ndarray] = []

        hidden = values
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z_values = hidden @ weight + bias
            pre_activations.append(z_values)
            hidden = _relu(z_values)
            activations.append(hidden)

        output = hidden @ self.weights[-1] + self.biases[-1]
        pre_activations.append(output)
        activations.append(output)
        return activations, pre_activations

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        output_gradient: np.ndarray,
    ) -> None:
        delta = output_gradient.astype(np.float32)

        for layer_index in range(len(self.weights) - 1, -1, -1):
            activation_prev = activations[layer_index]
            weight_gradient = activation_prev.T @ delta + self.l2 * self.weights[layer_index]
            bias_gradient = delta.sum(axis=0)

            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * _relu_grad(pre_activations[layer_index - 1])

            self.weights[layer_index] -= self.learning_rate * weight_gradient
            self.biases[layer_index] -= self.learning_rate * bias_gradient

    def _iterate_batches(self, total_rows: int, batch_size: int) -> list[np.ndarray]:
        indices = self.rng.permutation(total_rows)
        return [indices[start : start + batch_size] for start in range(0, total_rows, batch_size)]


class MLPClassifier(_BaseMLP):
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        batch_size: int,
        class_weights: np.ndarray | None = None,
    ) -> "MLPClassifier":
        features = features.astype(np.float32)
        scaled = self.scaler.fit_transform(features)
        labels = labels.astype(np.int64)

        weight_lookup = (
            np.ones(int(labels.max()) + 1, dtype=np.float32)
            if class_weights is None
            else class_weights.astype(np.float32)
        )

        for _ in range(epochs):
            epoch_losses = []
            for batch_indices in self._iterate_batches(len(scaled), batch_size):
                batch_x = scaled[batch_indices]
                batch_y = labels[batch_indices]

                activations, pre_activations = self._forward(batch_x)
                logits = activations[-1]
                probabilities = _softmax(logits)

                one_hot = np.eye(self.output_dim, dtype=np.float32)[batch_y]
                sample_weights = weight_lookup[batch_y]
                normalizer = sample_weights.sum() + EPSILON
                loss = -np.sum(
                    sample_weights * np.log(probabilities[np.arange(len(batch_y)), batch_y] + EPSILON)
                ) / normalizer
                loss += 0.5 * self.l2 * sum((weight * weight).sum() for weight in self.weights)
                epoch_losses.append(float(loss))

                gradient = (probabilities - one_hot) * sample_weights[:, None] / normalizer
                self._backward(activations, pre_activations, gradient)

            self.loss_history.append(float(np.mean(epoch_losses)))

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features.astype(np.float32))
        activations, _ = self._forward(scaled)
        return _softmax(activations[-1])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    def save(self, path: Path) -> None:
        payload = {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
            "hidden_sizes": np.array(self.hidden_sizes, dtype=np.int32),
            "output_dim": np.array([self.output_dim], dtype=np.int32),
        }
        for index, weight in enumerate(self.weights):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = self.biases[index]
        np.savez(path, **payload)


class MLPRegressor(_BaseMLP):
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> "MLPRegressor":
        features = features.astype(np.float32)
        targets = targets.astype(np.float32)
        scaled = self.scaler.fit_transform(features)

        for _ in range(epochs):
            epoch_losses = []
            for batch_indices in self._iterate_batches(len(scaled), batch_size):
                batch_x = scaled[batch_indices]
                batch_y = targets[batch_indices]

                activations, pre_activations = self._forward(batch_x)
                predictions = activations[-1]
                errors = predictions - batch_y
                loss = np.mean(errors * errors)
                loss += 0.5 * self.l2 * sum((weight * weight).sum() for weight in self.weights)
                epoch_losses.append(float(loss))

                gradient = 2.0 * errors / max(len(batch_indices), 1)
                self._backward(activations, pre_activations, gradient)

            self.loss_history.append(float(np.mean(epoch_losses)))

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features.astype(np.float32))
        activations, _ = self._forward(scaled)
        return activations[-1]

    def save(self, path: Path) -> None:
        payload = {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
            "hidden_sizes": np.array(self.hidden_sizes, dtype=np.int32),
            "output_dim": np.array([self.output_dim], dtype=np.int32),
        }
        for index, weight in enumerate(self.weights):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = self.biases[index]
        np.savez(path, **payload)
