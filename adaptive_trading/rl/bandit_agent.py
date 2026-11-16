from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import RLConfig
from ..mlp import MLPRegressor


class ContextualBanditAgent:
    def __init__(self, input_dim: int, config: RLConfig) -> None:
        self.config = config
        self.model = MLPRegressor(
            input_dim=input_dim,
            hidden_sizes=config.hidden_sizes,
            output_dim=len(config.action_names),
            learning_rate=config.learning_rate,
            l2=config.l2,
            seed=config.seed,
        )

    def fit(
        self,
        state_matrix: np.ndarray,
        reward_table: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        clipped_rewards = np.clip(
            reward_table.astype(np.float32),
            -self.config.reward_clip,
            self.config.reward_clip,
        )
        if action_mask is not None:
            action_mask = action_mask.astype(bool)
            clipped_rewards = np.where(action_mask, clipped_rewards, -self.config.reward_clip)
        self.model.fit(
            features=state_matrix,
            targets=clipped_rewards,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
        )
        return {
            "final_loss": float(self.model.loss_history[-1]) if self.model.loss_history else 0.0,
            "epochs": float(self.config.epochs),
        }

    def predict_scores(self, state_matrix: np.ndarray) -> np.ndarray:
        return self.model.predict(state_matrix)

    def _masked_scores(self, scores: np.ndarray, action_mask: np.ndarray | None) -> np.ndarray:
        if action_mask is None:
            return scores

        masked_scores = scores.copy()
        invalid_mask = ~action_mask.astype(bool)
        masked_scores[invalid_mask] = -np.inf

        no_valid_actions = ~action_mask.any(axis=1)
        if np.any(no_valid_actions):
            masked_scores[no_valid_actions, 0] = 0.0

        return masked_scores

    def choose_actions(
        self,
        state_matrix: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> list[str]:
        scores = self.predict_scores(state_matrix)
        masked_scores = self._masked_scores(scores, action_mask)
        action_indices = masked_scores.argmax(axis=1)
        return [self.config.action_names[int(index)] for index in action_indices]

    def choose_action_with_metadata(
        self,
        state_matrix: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores = self.predict_scores(state_matrix)
        raw_indices = scores.argmax(axis=1)
        masked_scores = self._masked_scores(scores, action_mask)
        chosen_indices = masked_scores.argmax(axis=1)
        return scores, raw_indices, chosen_indices

    def save(self, path: Path) -> None:
        self.model.save(path)
