from __future__ import annotations

import json
import math
import shutil
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harmonic.swing_detector import detect_swings

from .config import PatternGalleryConfig, PipelineConfig

EPSILON = 1e-9


def _slugify(value: str) -> str:
    text = value.strip().lower()
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "-", "_"}:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "pattern"


def _timestamp_label(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d_%H%M%S")


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _style_color(name: str) -> str:
    palette = {
        "entry": "#1f77b4",
        "stop": "#d62728",
        "tp1": "#2ca02c",
        "tp2": "#9467bd",
        "signal": "#ff7f0e",
        "bull": "#2e8b57",
        "bear": "#c0392b",
        "swing": "#111827",
        "grid": "#d1d5db",
        "text": "#111827",
    }
    return palette[name]


class PatternGalleryBuilder:
    def __init__(self, repo_root: Path, pipeline_config: PipelineConfig, artifact_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.pipeline_config = pipeline_config
        self.config: PatternGalleryConfig = pipeline_config.gallery
        self.artifact_root = Path(artifact_root)
        self.output_dir = self.artifact_root / self.config.output_dir

    def build(
        self,
        market_frames: dict[str, pd.DataFrame],
        harmonic_signals: pd.DataFrame,
        neural_signals: pd.DataFrame,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {
                "enabled": False,
                "output_dir": str(self.output_dir),
                "harmonic_images": 0,
                "neural_images": 0,
                "neural_classes": 0,
            }

        target_dir = self.output_dir
        staging_dir = target_dir.parent / f"{target_dir.name}__staging"
        backup_dir = target_dir.parent / f"{target_dir.name}__backup"

        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)

        self.output_dir = staging_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, Any]] = []
        neural_member_rows: list[dict[str, Any]] = []

        try:
            harmonic_summary = self._render_harmonic_gallery(
                market_frames=market_frames,
                harmonic_signals=harmonic_signals,
                manifest_rows=manifest_rows,
            )
            neural_summary = self._render_neural_gallery(
                market_frames=market_frames,
                neural_signals=neural_signals,
                manifest_rows=manifest_rows,
                member_rows=neural_member_rows,
            )

            manifest_frame = pd.DataFrame(manifest_rows)
            manifest_path = self.output_dir / self.config.manifest_file
            manifest_frame.to_csv(manifest_path, index=False)

            class_members_frame = pd.DataFrame(neural_member_rows)
            class_members_path = self.output_dir / self.config.class_members_file
            class_members_frame.to_csv(class_members_path, index=False)

            summary = {
                "enabled": True,
                "output_dir": str(target_dir),
                "manifest_file": str(target_dir / self.config.manifest_file),
                "class_members_file": str(target_dir / self.config.class_members_file),
                "harmonic": harmonic_summary,
                "neural": neural_summary,
                "harmonic_images": int(harmonic_summary["image_count"]),
                "neural_images": int(neural_summary["image_count"]),
                "neural_classes": int(neural_summary["class_count"]),
            }
            with (self.output_dir / self.config.summary_file).open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            self.output_dir = target_dir
            raise

        self.output_dir = target_dir
        try:
            if target_dir.exists():
                target_dir.replace(backup_dir)
            staging_dir.replace(target_dir)
        except Exception:
            if backup_dir.exists() and not target_dir.exists():
                backup_dir.replace(target_dir)
            raise
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)
            shutil.rmtree(staging_dir, ignore_errors=True)

        return summary

    def _locate_signal_index(self, frame: pd.DataFrame, signal: pd.Series) -> int | None:
        explicit_index = signal.get("signal_index")
        if explicit_index is not None and not pd.isna(explicit_index):
            index = int(explicit_index)
            if 0 <= index < len(frame):
                return index

        time_values = frame["time"].to_numpy()
        timestamp = pd.Timestamp(signal["timestamp"]).to_datetime64()
        index = int(np.searchsorted(time_values, timestamp, side="left"))
        if index >= len(frame):
            return None
        if pd.Timestamp(frame.iloc[index]["time"]) == pd.Timestamp(signal["timestamp"]):
            return index
        if index > 0 and pd.Timestamp(frame.iloc[index - 1]["time"]) == pd.Timestamp(signal["timestamp"]):
            return index - 1
        return None

    def _window_slice(
        self,
        frame: pd.DataFrame,
        center_index: int,
        lookback: int,
        lookforward: int,
    ) -> tuple[pd.DataFrame, int]:
        start = max(0, center_index - lookback)
        end = min(len(frame), center_index + lookforward + 1)
        window = frame.iloc[start:end].reset_index(drop=True)
        return window, center_index - start

    def _harmonic_swings(
        self,
        frame: pd.DataFrame,
        signal_index: int,
        window_start_index: int,
    ) -> list[tuple[int, float, str]]:
        swings = detect_swings(frame)
        for position, swing in enumerate(swings):
            if swing[0] != signal_index:
                continue
            if position < 4:
                return []
            labels = ["X", "A", "B", "C", "D"]
            selected = swings[position - 4 : position + 1]
            return [
                (int(index - window_start_index), float(price), label)
                for label, (index, price, _) in zip(labels, selected)
            ]
        return []

    def _price_lines(self, signal: pd.Series) -> list[tuple[str, float, str]]:
        return [
            ("Entry", float(signal["entry"]), _style_color("entry")),
            ("Stop", float(signal["stop_loss"]), _style_color("stop")),
            ("TP1", float(signal["target_1"]), _style_color("tp1")),
            ("TP2", float(signal["target_2"]), _style_color("tp2")),
        ]

    def _svg_chart(
        self,
        frame: pd.DataFrame,
        output_path: Path,
        title: str,
        subtitle: str,
        signal_local_index: int,
        price_lines: list[tuple[str, float, str]],
        swing_points: list[tuple[int, float, str]] | None = None,
    ) -> None:
        width = self.config.chart_width
        height = self.config.chart_height
        margin_left = 88
        margin_right = 140
        margin_top = 72
        margin_bottom = 64
        chart_width = max(width - margin_left - margin_right, 100)
        chart_height = max(height - margin_top - margin_bottom, 100)

        lows = frame["low"].astype(float).to_numpy()
        highs = frame["high"].astype(float).to_numpy()
        opens = frame["open"].astype(float).to_numpy()
        closes = frame["close"].astype(float).to_numpy()

        extra_prices = [price for _, price, _ in price_lines]
        if swing_points:
            extra_prices.extend(price for _, price, _ in swing_points)

        min_price = min(float(lows.min()), min(extra_prices))
        max_price = max(float(highs.max()), max(extra_prices))
        price_padding = max((max_price - min_price) * 0.08, EPSILON)
        min_price -= price_padding
        max_price += price_padding

        def y_map(price: float) -> float:
            ratio = (price - min_price) / (max_price - min_price + EPSILON)
            return margin_top + chart_height * (1.0 - ratio)

        step = chart_width / max(len(frame), 1)
        body_width = max(3.0, min(step * 0.65, 16.0))

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{margin_left}" y="28" font-size="22" font-family="Segoe UI, Arial, sans-serif" fill="{_style_color("text")}">{escape(title)}</text>',
            f'<text x="{margin_left}" y="52" font-size="13" font-family="Segoe UI, Arial, sans-serif" fill="#4b5563">{escape(subtitle)}</text>',
        ]

        for level in range(6):
            y_value = margin_top + chart_height * (level / 5.0)
            price = max_price - (max_price - min_price) * (level / 5.0)
            svg_lines.append(
                f'<line x1="{margin_left}" y1="{y_value:.2f}" x2="{margin_left + chart_width}" y2="{y_value:.2f}" '
                f'stroke="{_style_color("grid")}" stroke-width="1" stroke-dasharray="4 4" />'
            )
            svg_lines.append(
                f'<text x="12" y="{y_value + 4:.2f}" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="#6b7280">{price:.2f}</text>'
            )

        signal_x = margin_left + signal_local_index * step + step * 0.5
        svg_lines.append(
            f'<line x1="{signal_x:.2f}" y1="{margin_top}" x2="{signal_x:.2f}" y2="{margin_top + chart_height}" '
            f'stroke="{_style_color("signal")}" stroke-width="1.5" stroke-dasharray="6 4" />'
        )

        for index, (open_, high_, low_, close_) in enumerate(zip(opens, highs, lows, closes)):
            x_center = margin_left + index * step + step * 0.5
            color = _style_color("bull") if close_ >= open_ else _style_color("bear")
            y_high = y_map(float(high_))
            y_low = y_map(float(low_))
            y_open = y_map(float(open_))
            y_close = y_map(float(close_))
            body_top = min(y_open, y_close)
            body_height = max(abs(y_open - y_close), 1.0)

            svg_lines.append(
                f'<line x1="{x_center:.2f}" y1="{y_high:.2f}" x2="{x_center:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1.5" />'
            )
            svg_lines.append(
                f'<rect x="{x_center - body_width / 2:.2f}" y="{body_top:.2f}" width="{body_width:.2f}" height="{body_height:.2f}" '
                f'fill="{color}" fill-opacity="0.85" stroke="{color}" stroke-width="1" />'
            )

        for label, price, color in price_lines:
            y_value = y_map(price)
            svg_lines.append(
                f'<line x1="{margin_left}" y1="{y_value:.2f}" x2="{margin_left + chart_width}" y2="{y_value:.2f}" '
                f'stroke="{color}" stroke-width="1.3" stroke-dasharray="8 4" />'
            )
            svg_lines.append(
                f'<text x="{margin_left + chart_width + 8}" y="{y_value + 4:.2f}" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="{color}">{escape(label)} {price:.2f}</text>'
            )

        if swing_points:
            point_text = " ".join(
                f"{margin_left + local_index * step + step * 0.5:.2f},{y_map(price):.2f}"
                for local_index, price, _ in swing_points
            )
            svg_lines.append(
                f'<polyline points="{point_text}" fill="none" stroke="{_style_color("swing")}" stroke-width="2.2" />'
            )
            for local_index, price, label in swing_points:
                x_value = margin_left + local_index * step + step * 0.5
                y_value = y_map(price)
                svg_lines.append(
                    f'<circle cx="{x_value:.2f}" cy="{y_value:.2f}" r="4.5" fill="#ffffff" stroke="{_style_color("swing")}" stroke-width="2" />'
                )
                svg_lines.append(
                    f'<text x="{x_value + 7:.2f}" y="{y_value - 7:.2f}" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="{_style_color("swing")}">{escape(label)}</text>'
                )

        svg_lines.append("</svg>")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(svg_lines), encoding="utf-8")

    def _render_harmonic_gallery(
        self,
        market_frames: dict[str, pd.DataFrame],
        harmonic_signals: pd.DataFrame,
        manifest_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if harmonic_signals.empty:
            return {"image_count": 0, "classes": {}}

        classes: dict[str, int] = {}
        image_count = 0

        for _, signal in harmonic_signals.sort_values(["timeframe", "pattern", "timestamp"]).iterrows():
            timeframe = str(signal["timeframe"])
            pattern = str(signal.get("pattern", "unknown"))
            frame = market_frames[timeframe]
            signal_index = self._locate_signal_index(frame, signal)
            if signal_index is None:
                continue

            lookback = int(self.config.harmonic_context_bars[timeframe])
            lookforward = int(self.config.harmonic_forward_bars[timeframe])
            start = max(0, signal_index - lookback)
            window, local_index = self._window_slice(frame, signal_index, lookback, lookforward)
            swings = self._harmonic_swings(frame, signal_index, window_start_index=start)

            folder = self.output_dir / "harmonic" / timeframe / _slugify(pattern)
            filename = f"{_timestamp_label(signal['timestamp'])}_{signal['direction']}_{signal_index:07d}.svg"
            output_path = folder / filename
            title = f"Harmonic {pattern.title()} {str(signal['direction']).upper()} [{timeframe}]"
            subtitle = (
                f"timestamp={pd.Timestamp(signal['timestamp'])} "
                f"confidence={float(signal['confidence']):.2%} rr={float(signal['risk_reward']):.2f}"
            )
            self._svg_chart(
                frame=window,
                output_path=output_path,
                title=title,
                subtitle=subtitle,
                signal_local_index=local_index,
                price_lines=self._price_lines(signal),
                swing_points=swings,
            )

            class_name = f"{timeframe}/{pattern}"
            classes[class_name] = classes.get(class_name, 0) + 1
            image_count += 1
            manifest_rows.append(
                {
                    "source": "harmonic_rule_engine",
                    "class_name": class_name,
                    "timeframe": timeframe,
                    "timestamp": str(pd.Timestamp(signal["timestamp"])),
                    "direction": str(signal["direction"]),
                    "confidence": float(signal["confidence"]),
                    "risk_reward": float(signal["risk_reward"]),
                    "image_path": str(output_path.relative_to(self.output_dir)),
                    "signal_index": signal_index,
                    "is_representative": True,
                    "class_rank": classes[class_name],
                }
            )

        return {"image_count": image_count, "classes": classes}

    def _neural_shape_vector(self, frame: pd.DataFrame, signal_index: int) -> np.ndarray | None:
        lookback = max(int(self.config.neural_context_bars), 8)
        start = signal_index - lookback + 1
        end = signal_index + int(self.config.neural_forward_bars) + 1
        if start < 0 or end > len(frame):
            return None
        close_values = frame.iloc[start:end]["close"].astype(float).to_numpy()
        if len(close_values) < 8:
            return None

        normalized = close_values / max(abs(close_values[0]), EPSILON) - 1.0
        scale = max(np.max(np.abs(normalized)), EPSILON)
        return (normalized / scale).astype(np.float32)

    def _kmeans(
        self,
        values: np.ndarray,
        cluster_count: int,
        seed: int,
        max_iterations: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(values) == 0:
            return np.empty((0, values.shape[1])), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        rng = np.random.default_rng(seed)
        cluster_count = max(1, min(cluster_count, len(values)))
        initial_indices = rng.choice(len(values), size=cluster_count, replace=False)
        centers = values[initial_indices].copy()

        labels = np.zeros(len(values), dtype=np.int32)
        for _ in range(max_iterations):
            squared_distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            next_labels = squared_distances.argmin(axis=1).astype(np.int32)
            if np.array_equal(labels, next_labels):
                labels = next_labels
                break
            labels = next_labels

            for cluster_index in range(cluster_count):
                members = values[labels == cluster_index]
                if len(members) == 0:
                    farthest_index = squared_distances.min(axis=1).argmax()
                    centers[cluster_index] = values[farthest_index]
                else:
                    centers[cluster_index] = members.mean(axis=0)

        final_distances = np.sqrt(((values - centers[labels]) ** 2).sum(axis=1))
        return centers, labels, final_distances.astype(np.float32)

    def _render_neural_gallery(
        self,
        market_frames: dict[str, pd.DataFrame],
        neural_signals: pd.DataFrame,
        manifest_rows: list[dict[str, Any]],
        member_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if neural_signals.empty:
            return {"image_count": 0, "class_count": 0, "classes": {}}

        master_timeframe = self.pipeline_config.data.master_timeframe
        frame = market_frames[master_timeframe]
        neural_signals = neural_signals.copy()
        neural_signals["signal_index"] = [
            self._locate_signal_index(frame, signal)
            for _, signal in neural_signals.iterrows()
        ]
        neural_signals = neural_signals.dropna(subset=["signal_index"]).copy()
        if neural_signals.empty:
            return {"image_count": 0, "class_count": 0, "classes": {}}

        classes: dict[str, int] = {}
        image_count = 0

        for direction in ["long", "short"]:
            direction_frame = neural_signals.loc[neural_signals["direction"] == direction].copy().reset_index(drop=True)
            if direction_frame.empty:
                continue

            shapes: list[np.ndarray] = []
            valid_positions: list[int] = []
            for row_position, row in direction_frame.iterrows():
                shape = self._neural_shape_vector(frame, int(row["signal_index"]))
                if shape is None:
                    continue
                shapes.append(shape)
                valid_positions.append(row_position)

            if not shapes:
                continue

            direction_frame = direction_frame.iloc[valid_positions].reset_index(drop=True)
            shape_matrix = np.vstack(shapes)
            cluster_count = min(int(self.config.neural_cluster_count), len(direction_frame))
            _, labels, distances = self._kmeans(
                values=shape_matrix,
                cluster_count=cluster_count,
                seed=self.pipeline_config.neural.seed,
            )

            direction_frame["visual_class"] = [
                f"{direction}_pattern_{int(label) + 1:02d}" for label in labels
            ]
            direction_frame["cluster_distance"] = distances

            for _, member in direction_frame.iterrows():
                member_rows.append(
                    {
                        "timestamp": str(pd.Timestamp(member["timestamp"])),
                        "direction": direction,
                        "timeframe": str(member["timeframe"]),
                        "confidence": float(member["confidence"]),
                        "risk_reward": float(member["risk_reward"]),
                        "signal_index": int(member["signal_index"]),
                        "visual_class": str(member["visual_class"]),
                        "cluster_distance": float(member["cluster_distance"]),
                    }
                )

            for class_name, class_frame in direction_frame.groupby("visual_class"):
                class_frame = class_frame.sort_values(
                    ["cluster_distance", "confidence"],
                    ascending=[True, False],
                ).reset_index(drop=True)
                classes[class_name] = int(len(class_frame))
                folder = self.output_dir / "non_harmonic" / class_name
                examples_to_render = min(len(class_frame), int(self.config.neural_examples_per_class))

                for rank in range(examples_to_render):
                    signal = class_frame.iloc[rank]
                    signal_index = int(signal["signal_index"])
                    lookback = max(int(self.config.neural_context_bars), 8)
                    lookforward = int(self.config.neural_forward_bars)
                    window, local_index = self._window_slice(
                        frame=frame,
                        center_index=signal_index,
                        lookback=lookback - 1,
                        lookforward=lookforward,
                    )
                    filename = f"{rank + 1:02d}_{_timestamp_label(signal['timestamp'])}_{signal_index:07d}.svg"
                    output_path = folder / filename
                    title = f"Neural {class_name.replace('_', ' ').title()}"
                    subtitle = (
                        f"timestamp={pd.Timestamp(signal['timestamp'])} "
                        f"confidence={float(signal['confidence']):.2%} rr={float(signal['risk_reward']):.2f}"
                    )
                    self._svg_chart(
                        frame=window,
                        output_path=output_path,
                        title=title,
                        subtitle=subtitle,
                        signal_local_index=local_index,
                        price_lines=self._price_lines(signal),
                        swing_points=None,
                    )

                    image_count += 1
                    manifest_rows.append(
                        {
                            "source": "non_harmonic_neural_net",
                            "class_name": class_name,
                            "timeframe": str(signal["timeframe"]),
                            "timestamp": str(pd.Timestamp(signal["timestamp"])),
                            "direction": str(signal["direction"]),
                            "confidence": float(signal["confidence"]),
                            "risk_reward": float(signal["risk_reward"]),
                            "image_path": str(output_path.relative_to(self.output_dir)),
                            "signal_index": signal_index,
                            "is_representative": rank == 0,
                            "class_rank": rank + 1,
                        }
                    )

        return {
            "image_count": image_count,
            "class_count": len(classes),
            "classes": classes,
        }
