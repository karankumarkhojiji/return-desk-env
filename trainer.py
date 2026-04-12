"""
trainer.py — Multi-episode training harness for ReturnDeskEnv

Runs N episodes of the deterministic policy (or optionally an LLM agent)
across all 7 tasks, records per-episode scores, and plots a learning curve.

Usage:
    # Deterministic baseline across all tasks (no API key needed)
    python trainer.py --episodes 70 --use-llm false

    # LLM agent (requires HF_TOKEN in .env)
    python trainer.py --episodes 50 --use-llm true --model Qwen/Qwen2.5-72B-Instruct

Output:
    outputs/training_curve.png  — Learning curve plot
    outputs/baseline_scores.json — Full episode results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure package root is importable when run directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from tasks.catalog import list_task_ids, get_task
    from server.environment import ReturnDeskEnvironment, CurriculumState
    from inference import _deterministic_policy
except ImportError:
    from return_desk_env.tasks.catalog import list_task_ids, get_task
    from return_desk_env.server.environment import ReturnDeskEnvironment, CurriculumState
    from return_desk_env.inference import _deterministic_policy


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Runs N training episodes using the deterministic policy.

    Episode scheduling:
    - When mode="round_robin", cycles evenly through all 7 tasks.
    - When mode="curriculum", auto-advances difficulty based on rolling mean.
    - When mode="random", samples tasks uniformly at random (with seed).

    Records per-episode scores and produces:
    - A learning curve plot (matplotlib)
    - A JSON summary file
    """

    def __init__(
        self,
        mode: str = "round_robin",
        episodes: int = 70,
        seed_start: int = 42,
        output_dir: str = "outputs",
    ) -> None:
        self.mode = mode
        self.episodes = episodes
        self.seed_start = seed_start
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self._task_ids = sorted(list_task_ids())
        self._curriculum = CurriculumState(window=10)
        self._results: List[Dict[str, Any]] = []

    def _pick_task(self, episode: int, seed: int) -> Optional[str]:
        if self.mode == "round_robin":
            return self._task_ids[episode % len(self._task_ids)]
        elif self.mode == "curriculum":
            return None  # let env pick via curriculum_state
        else:  # random
            import random
            return random.Random(seed).choice(self._task_ids)

    def run(self) -> List[Dict[str, Any]]:
        print(f"\n{'='*60}")
        print(f"  ReturnDeskEnv Trainer | mode={self.mode} | episodes={self.episodes}")
        print(f"{'='*60}")

        for ep in range(self.episodes):
            seed = self.seed_start + ep
            task_id = self._pick_task(ep, seed)

            env = ReturnDeskEnvironment()
            reset_kwargs: Dict[str, Any] = {"seed": seed}
            if task_id:
                reset_kwargs["task_id"] = task_id
            if self.mode == "curriculum":
                reset_kwargs["mode"] = "curriculum"
                reset_kwargs["curriculum_state"] = self._curriculum

            obs = env.reset(**reset_kwargs)
            actual_task_id = obs.task_id

            rewards: List[float] = []
            done = False
            step = 0

            while not done:
                action_dict = _deterministic_policy(obs)
                try:
                    from models import ReturnDeskAction  # type: ignore
                except ImportError:
                    from return_desk_env.models import ReturnDeskAction
                action = ReturnDeskAction(**action_dict)
                obs = env.step(action)
                r = obs.reward or 0.0
                rewards.append(r)
                done = obs.done
                step += 1

            score = obs.final_score or 0.0
            self._curriculum.record(score)

            result = {
                "episode": ep + 1,
                "task_id": actual_task_id,
                "difficulty": obs.difficulty,
                "score": round(score, 4),
                "steps": step,
                "rewards": [round(r, 4) for r in rewards],
                "grader_breakdown": obs.grader_breakdown,
                "curriculum_mean": self._curriculum.rolling_mean,
            }
            self._results.append(result)

            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            print(
                f"  Ep {ep+1:>3}/{self.episodes}  {actual_task_id:<28}  "
                f"score={score:.3f}  [{bar}]  steps={step:>2}  "
                f"rolling={self._curriculum.rolling_mean:.3f}"
            )

        print(f"\n{'='*60}")
        print(f"  Final mean score: {self._final_mean:.4f}")
        print(f"  Curriculum final difficulty: {self._curriculum.select_difficulty()}")
        print(f"{'='*60}\n")

        return self._results

    @property
    def _final_mean(self) -> float:
        if not self._results:
            return 0.0
        return round(sum(r["score"] for r in self._results) / len(self._results), 4)

    def save_results(self) -> Path:
        out = self.output_dir / "baseline_scores.json"
        payload = {
            "trainer_mode": self.mode,
            "episodes": self.episodes,
            "final_mean_score": self._final_mean,
            "curriculum_summary": self._curriculum.summary(),
            "results": self._results,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved results -> {out}")
        return out

    def plot(self) -> Optional[Path]:
        """Generate a learning curve plot. Requires matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("  matplotlib not installed — skipping plot. Run: pip install matplotlib")
            return None

        episodes = [r["episode"] for r in self._results]
        scores = [r["score"] for r in self._results]
        rolling: List[float] = []
        window = 10
        for i in range(len(scores)):
            w = scores[max(0, i - window + 1): i + 1]
            rolling.append(round(sum(w) / len(w), 4))

        # Colour by task difficulty
        TASK_COLORS = {
            "easy": "#22c55e",
            "medium": "#f59e0b",
            "hard": "#ef4444",
            "extreme": "#8b5cf6",
        }
        dot_colors = [TASK_COLORS.get(r["difficulty"], "#64748b") for r in self._results]

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#151821")

        ax.scatter(episodes, scores, c=dot_colors, s=60, alpha=0.7, zorder=3, label="Episode score")
        ax.plot(episodes, rolling, color="#6366f1", linewidth=2.5, label=f"Rolling mean (w={window})", zorder=4)
        ax.axhline(y=0.80, color="#6366f1", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(1, 0.815, "Hard→Extreme threshold (0.82)", color="#6366f1", alpha=0.5, fontsize=8)

        ax.set_xlabel("Episode", color="#e2e8f0", fontsize=11)
        ax.set_ylabel("Score", color="#e2e8f0", fontsize=11)
        ax.set_title("ReturnDeskEnv — Deterministic Policy Learning Curve", color="#e2e8f0", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#64748b")
        ax.set_ylim(0, 1.05)
        ax.grid(True, color="#252a3a", linewidth=0.5, alpha=0.5)

        # Legend
        patches = [mpatches.Patch(color=c, label=d.capitalize()) for d, c in TASK_COLORS.items()]
        patches.append(plt.Line2D([0], [0], color="#6366f1", linewidth=2.5, label=f"Rolling mean (w={window})"))
        ax.legend(handles=patches, facecolor="#1c2030", edgecolor="#252a3a", labelcolor="#e2e8f0", loc="lower right")

        for spine in ax.spines.values():
            spine.set_edgecolor("#252a3a")

        plt.tight_layout()
        out = self.output_dir / "training_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved learning curve -> {out}")
        return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ReturnDeskEnv training harness")
    parser.add_argument("--episodes", type=int, default=70, help="Number of episodes to run")
    parser.add_argument("--mode", choices=["round_robin", "curriculum", "random"], default="round_robin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = parser.parse_args()

    trainer = Trainer(
        mode=args.mode,
        episodes=args.episodes,
        seed_start=args.seed,
        output_dir=args.output_dir,
    )
    trainer.run()
    trainer.save_results()
    if not args.no_plot:
        trainer.plot()


if __name__ == "__main__":
    main()
