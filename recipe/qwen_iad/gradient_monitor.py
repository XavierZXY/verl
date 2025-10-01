#!/usr/bin/env python3
"""
Gradient monitoring tool for analyzing gradient norms during training
Helps identify when large grad_norm occurs with identical prompts
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientMonitor:
    """Monitor and analyze gradient norms during training"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_history = defaultdict(list)
        self.step_count = 0

    def log_gradients(self, model: nn.Module, step: int, prefix: str = ""):
        """Log gradient norms for all model parameters"""
        total_norm = 0.0
        param_count = 0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
                param_count += 1

                # Group by layer type
                layer_type = self._get_layer_type(name)
                if layer_type not in layer_norms:
                    layer_norms[layer_type] = []
                layer_norms[layer_type].append(param_norm)

                # Log individual parameter gradients for critical layers
                if any(critical in name.lower() for critical in ["attention", "vision", "embed"]):
                    self.gradient_history[f"{prefix}{name}"].append(
                        {
                            "step": step,
                            "norm": param_norm,
                            "shape": list(param.shape),
                            "mean": param.grad.data.mean().item(),
                            "std": param.grad.data.std().item(),
                            "max": param.grad.data.max().item(),
                            "min": param.grad.data.min().item(),
                        }
                    )

        total_norm = total_norm**0.5

        # Log layer-wise statistics
        for layer_type, norms in layer_norms.items():
            self.gradient_history[f"{prefix}layer_{layer_type}"].append(
                {
                    "step": step,
                    "mean_norm": np.mean(norms),
                    "max_norm": np.max(norms),
                    "std_norm": np.std(norms),
                    "count": len(norms),
                }
            )

        # Log overall statistics
        self.gradient_history[f"{prefix}total_norm"].append(
            {"step": step, "norm": total_norm, "param_count": param_count}
        )

        return total_norm

    def _get_layer_type(self, param_name: str) -> str:
        """Categorize parameter by layer type"""
        name_lower = param_name.lower()
        if "embed" in name_lower:
            return "embedding"
        elif "attention" in name_lower or "attn" in name_lower:
            return "attention"
        elif "vision" in name_lower or "visual" in name_lower:
            return "vision"
        elif "mlp" in name_lower or "ffn" in name_lower:
            return "mlp"
        elif "norm" in name_lower or "ln" in name_lower:
            return "normalization"
        elif "lora" in name_lower:
            return "lora"
        else:
            return "other"

    def detect_gradient_explosion(self, threshold: float = 10.0) -> list[dict]:
        """Detect steps where gradient explosion occurred"""
        explosions = []

        for key, history in self.gradient_history.items():
            if "total_norm" in key:
                for entry in history:
                    if entry["norm"] > threshold:
                        explosions.append(
                            {"step": entry["step"], "norm": entry["norm"], "type": "total_norm_explosion"}
                        )

        return explosions

    def analyze_gradient_patterns(self) -> dict:
        """Analyze gradient patterns to identify issues"""
        analysis = {
            "gradient_explosions": self.detect_gradient_explosion(),
            "layer_statistics": {},
            "recommendations": [],
        }

        # Analyze layer-wise patterns
        for key, history in self.gradient_history.items():
            if key.startswith("layer_"):
                layer_type = key.replace("layer_", "")
                norms = [entry["mean_norm"] for entry in history]

                analysis["layer_statistics"][layer_type] = {
                    "mean_norm": np.mean(norms),
                    "std_norm": np.std(norms),
                    "max_norm": np.max(norms),
                    "min_norm": np.min(norms),
                    "variance": np.var(norms),
                }

        # Generate recommendations
        total_norms = [entry["norm"] for entry in self.gradient_history.get("total_norm", [])]
        if total_norms:
            max_norm = max(total_norms)
            mean_norm = np.mean(total_norms)
            std_norm = np.std(total_norms)

            if max_norm > 10.0:
                analysis["recommendations"].append(
                    f"Large gradient norm detected (max: {max_norm:.2f}). "
                    "Consider reducing learning rate or increasing gradient clipping."
                )

            if std_norm > mean_norm:
                analysis["recommendations"].append(
                    f"High gradient variance detected (std: {std_norm:.2f}, mean: {mean_norm:.2f}). "
                    "This suggests unstable training, possibly due to identical prompts."
                )

            # Check for vision layer issues
            vision_stats = analysis["layer_statistics"].get("vision", {})
            if vision_stats.get("variance", 0) > vision_stats.get("mean_norm", 0) ** 2:
                analysis["recommendations"].append(
                    "Vision layers show high gradient variance. "
                    "Consider adding image augmentation or prompt diversification."
                )

        return analysis

    def save_analysis(self, filename: str = "gradient_analysis.json"):
        """Save gradient analysis to file"""
        analysis = self.analyze_gradient_patterns()

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        analysis = convert_numpy(analysis)

        with open(self.log_dir / filename, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Gradient analysis saved to {self.log_dir / filename}")
        return analysis

    def plot_gradient_trends(self, save_plots: bool = True):
        """Plot gradient trends over training steps"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Gradient Analysis During Training", fontsize=16)

        # Plot 1: Total gradient norm over time
        if "total_norm" in self.gradient_history:
            steps = [entry["step"] for entry in self.gradient_history["total_norm"]]
            norms = [entry["norm"] for entry in self.gradient_history["total_norm"]]

            axes[0, 0].plot(steps, norms, "b-", alpha=0.7)
            axes[0, 0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Clip threshold")
            axes[0, 0].set_title("Total Gradient Norm")
            axes[0, 0].set_xlabel("Training Step")
            axes[0, 0].set_ylabel("Gradient Norm")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Layer-wise gradient norms
        layer_data = {}
        for key, history in self.gradient_history.items():
            if key.startswith("layer_"):
                layer_type = key.replace("layer_", "")
                steps = [entry["step"] for entry in history]
                norms = [entry["mean_norm"] for entry in history]
                layer_data[layer_type] = (steps, norms)

        for layer_type, (steps, norms) in layer_data.items():
            axes[0, 1].plot(steps, norms, label=layer_type, alpha=0.7)

        axes[0, 1].set_title("Layer-wise Gradient Norms")
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Mean Gradient Norm")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Gradient norm distribution
        if "total_norm" in self.gradient_history:
            norms = [entry["norm"] for entry in self.gradient_history["total_norm"]]
            axes[1, 0].hist(norms, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
            axes[1, 0].axvline(x=np.mean(norms), color="r", linestyle="--", label=f"Mean: {np.mean(norms):.2f}")
            axes[1, 0].axvline(x=np.median(norms), color="g", linestyle="--", label=f"Median: {np.median(norms):.2f}")
            axes[1, 0].set_title("Gradient Norm Distribution")
            axes[1, 0].set_xlabel("Gradient Norm")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Gradient variance over time
        if layer_data:
            for layer_type, (steps, norms) in layer_data.items():
                if len(norms) > 10:  # Need enough points for rolling variance
                    rolling_var = pd.Series(norms).rolling(window=10).var()
                    axes[1, 1].plot(steps, rolling_var, label=f"{layer_type} variance", alpha=0.7)

            axes[1, 1].set_title("Gradient Variance (Rolling Window)")
            axes[1, 1].set_xlabel("Training Step")
            axes[1, 1].set_ylabel("Gradient Variance")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(self.log_dir / "gradient_analysis.png", dpi=300, bbox_inches="tight")
            logger.info(f"Gradient plots saved to {self.log_dir / 'gradient_analysis.png'}")

        plt.show()


def analyze_training_logs(log_file: str, output_dir: str):
    """Analyze training logs for gradient patterns"""
    monitor = GradientMonitor(output_dir)

    # Parse training logs (this would need to be adapted based on actual log format)
    try:
        with open(log_file) as f:
            for line_num, line in enumerate(f):
                if "grad_norm" in line.lower():
                    # Extract gradient norm from log line
                    # This is a simplified parser - adapt based on actual log format
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "grad_norm" in part.lower() and i + 1 < len(parts):
                                norm_value = float(parts[i + 1])
                                monitor.gradient_history["total_norm"].append(
                                    {"step": line_num, "norm": norm_value, "param_count": 0}
                                )
                                break
                    except (ValueError, IndexError):
                        continue

        # Generate analysis
        analysis = monitor.save_analysis()
        monitor.plot_gradient_trends()

        print("\n=== Gradient Analysis Summary ===")
        print(f"Gradient explosions detected: {len(analysis['gradient_explosions'])}")

        if analysis["recommendations"]:
            print("\nRecommendations:")
            for rec in analysis["recommendations"]:
                print(f"- {rec}")

        return analysis

    except FileNotFoundError:
        logger.error(f"Log file not found: {log_file}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor and analyze gradient norms during training")
    parser.add_argument("--log_file", type=str, help="Path to training log file")
    parser.add_argument(
        "--output_dir", type=str, default="./gradient_analysis", help="Output directory for analysis results"
    )

    args = parser.parse_args()

    if args.log_file:
        analyze_training_logs(args.log_file, args.output_dir)
    else:
        print("Usage: python gradient_monitor.py --log_file <path_to_log> --output_dir <output_dir>")
        print("\nThis tool helps analyze gradient patterns during training to identify issues")
        print("caused by identical prompts or other training instabilities.")
