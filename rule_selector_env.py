"""Conservative Rule-Selector PPO environment for HGCR-PPO Stage F."""

from __future__ import annotations

import pickle
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import numpy as np

from candidate_generator import fifo_ranked, generate_candidates, greedy_ect_ranked, lookahead_ranked, minload_ranked
from mlp_models import load_checkpoint
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from stage_c_utils import extract_candidate_features, process_types_for_instance


BASE_RULES = ["fifo", "lookahead", "greedy_ect", "minload", "mlp_ranker_soft_ce"]
PAIRWISE_RULE = "mlp_ranker_pairwise"
ALL_RULES = [*BASE_RULES, PAIRWISE_RULE]


class RuleSelectorEnv:
    """Wrap ``RollingSchedulingEnv`` so PPO selects a dispatching rule id.

    The wrapped environment still executes the original action
    ``(job_id, split_num)``. The PPO action is a smaller and more conservative
    ``rule_id``; each available rule recommends one job from HybridTopK.
    """

    def __init__(
        self,
        instance=None,
        top_k: int = 5,
        mlp_soft_model_path: str | None = None,
        mlp_pairwise_model_path: str | None = None,
        baseline_cache_dir: str | Path = "data/cache/stage_F/fifo_baselines",
        seed: int = 42,
    ):
        self.top_k = max(1, int(top_k))
        self.rng = random.Random(seed)
        self.baseline_cache_dir = Path(baseline_cache_dir)
        self.mlp_soft_model = self._try_load_model(mlp_soft_model_path)
        self.mlp_pairwise_model = self._try_load_model(mlp_pairwise_model_path)
        self.rule_names = list(ALL_RULES)
        self.env = RollingSchedulingEnv(instance)
        self.instance = self.env.instance
        self.process_types = process_types_for_instance(self.instance)
        self.fifo_cmax = 1.0
        self.last_decision: Dict = {}
        self.rule_counts: Counter[str] = Counter()
        self.reset(instance)

    @property
    def action_dim(self) -> int:
        return len(self.rule_names)

    @property
    def state_dim(self) -> int:
        return int(len(self._build_state(self._build_decision())))

    def reset(self, instance=None) -> np.ndarray:
        if instance is not None:
            self.instance = instance
        self.env.reset(self.instance)
        self.instance = self.env.instance
        self.process_types = process_types_for_instance(self.instance)
        self.fifo_cmax = max(1e-6, self._fifo_baseline_cmax())
        self.last_decision = {}
        self.rule_counts = Counter()
        return self._build_state(self._build_decision())

    def step(self, action: int):
        if self.env.is_done():
            state = self._build_state(self._build_decision())
            return state, 0.0, True, self._info({"final_makespan": self.env.current_cmax})

        decision = self._build_decision()
        mask = decision["action_mask"]
        rule_id = int(action)
        if rule_id < 0 or rule_id >= len(self.rule_names) or not bool(mask[rule_id]):
            raise ValueError(f"Invalid or masked rule action {action}. mask={mask.tolist()}")

        rule_name = self.rule_names[rule_id]
        job_id = decision["rule_jobs"][rule_name]
        old_cmax = float(self.env.current_cmax)
        _, _, done, env_info = self.env.step((job_id, choose_split_num(self.env, job_id)))
        new_cmax = float(self.env.current_cmax)

        reward = -(new_cmax - old_cmax) / self.fifo_cmax
        if done:
            reward += -(new_cmax - self.fifo_cmax) / self.fifo_cmax

        self.rule_counts[rule_name] += 1
        info = self._info(
            {
                **env_info,
                "selected_rule": rule_name,
                "selected_rule_id": rule_id,
                "selected_job_id": job_id,
                "step_reward": reward,
                "old_cmax": old_cmax,
                "new_cmax": new_cmax,
                "fifo_cmax": self.fifo_cmax,
            }
        )
        next_decision = self._build_decision()
        return self._build_state(next_decision), float(reward), done, info

    def action_mask(self) -> np.ndarray:
        return self._build_decision()["action_mask"].copy()

    def greedy_rule_id(self, preferred_rule: str = "fifo") -> int:
        decision = self._build_decision()
        mask = decision["action_mask"]
        if preferred_rule in self.rule_names:
            idx = self.rule_names.index(preferred_rule)
            if bool(mask[idx]):
                return idx
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("No available rule action.")
        return int(valid[0])

    def rule_distribution(self) -> Dict[str, float]:
        total = sum(self.rule_counts.values())
        if total <= 0:
            return {rule: 0.0 for rule in self.rule_names}
        return {rule: float(self.rule_counts[rule]) / total for rule in self.rule_names}

    def _build_decision(self) -> Dict:
        if self.env.is_done():
            empty_mask = np.zeros(len(self.rule_names), dtype=bool)
            return {"candidates": [], "rule_jobs": {rule: None for rule in self.rule_names}, "action_mask": empty_mask}

        candidates = generate_candidates(
            self.env,
            candidate_mode="hybrid_topk",
            top_k=self.top_k,
            fallback_to_all=True,
        )
        candidate_set = set(candidates)
        rule_jobs = {
            "fifo": self._first_in_candidates(fifo_ranked(self.env), candidate_set),
            "lookahead": self._first_in_candidates(lookahead_ranked(self.env), candidate_set),
            "greedy_ect": self._first_in_candidates(greedy_ect_ranked(self.env), candidate_set),
            "minload": self._first_in_candidates(minload_ranked(self.env), candidate_set),
            "mlp_ranker_soft_ce": self._ranker_job(self.mlp_soft_model, candidates),
            "mlp_ranker_pairwise": self._ranker_job(self.mlp_pairwise_model, candidates),
        }
        mask = np.array([rule_jobs[rule] is not None for rule in self.rule_names], dtype=bool)
        return {"candidates": candidates, "rule_jobs": rule_jobs, "action_mask": mask}

    def _build_state(self, decision: Dict) -> np.ndarray:
        candidates = decision["candidates"]
        now = float(self.env._current_decision_time())
        max_release = max((float(job.release_time) for job in self.instance.jobs), default=1.0) or 1.0
        max_jobs = max(1, len(self.instance.jobs))
        cmax_norm = float(self.env.current_cmax) / self.fifo_cmax
        remaining_norm = len(self.env.unscheduled_jobs) / max_jobs

        waiting = [max(0.0, now - self.env.job_by_id[j].release_time) for j in self.env.get_schedulable_jobs()]
        machine_times = list(self.env.machine_available_time.values()) or [0.0]
        loads = self._machine_busy_loads()
        process_counts = [
            sum(1 for job_id in self.env.unscheduled_jobs if self.env.job_by_id[job_id].process_type == process_type)
            / max_jobs
            for process_type in self.process_types
        ]

        global_features = [
            cmax_norm,
            remaining_norm,
            (mean(waiting) if waiting else 0.0) / max_release,
            (max(waiting) if waiting else 0.0) / max_release,
            mean(machine_times) / self.fifo_cmax,
            (pstdev(machine_times) if len(machine_times) > 1 else 0.0) / self.fifo_cmax,
            (pstdev(loads) if len(loads) > 1 else 0.0) / self.fifo_cmax,
            len(candidates) / float(self.top_k),
            self._agreement_ratio(decision["rule_jobs"]),
        ]

        candidate_features = self._candidate_feature_map(candidates)
        state: List[float] = [*global_features, *process_counts]
        for rule in self.rule_names:
            job_id = decision["rule_jobs"].get(rule)
            available = 1.0 if job_id is not None else 0.0
            rank_norm = candidates.index(job_id) / max(1, len(candidates) - 1) if job_id in candidates else 0.0
            state.extend([available, rank_norm])
            state.extend(candidate_features.get(job_id, self._zero_candidate_summary()))
        return np.asarray(state, dtype=np.float32)

    def _candidate_feature_map(self, candidates: Sequence[str]) -> Dict[str, List[float]]:
        if not candidates:
            return {}
        raw_rows = extract_candidate_features(self.env, candidates)
        max_release = max((float(job.release_time) for job in self.instance.jobs), default=1.0) or 1.0
        max_processing = max(
            (
                float(self.instance.processing_time[job.job_id][machine_id])
                for job in self.instance.jobs
                for machine_id in job.candidate_machines
            ),
            default=1.0,
        ) or 1.0
        max_machines = max(1, len(self.instance.machines))
        max_split = max((job.max_split_num for job in self.instance.jobs), default=1) or 1
        out: Dict[str, List[float]] = {}
        for job_id, row in zip(candidates, raw_rows):
            out[job_id] = [
                float(row[0]),
                float(row[1]) / max_processing,
                float(row[2]) / max_processing,
                float(row[3]) / max_processing,
                float(row[4]) / max_machines,
                float(row[5]) / max_split,
                float(row[6]) / max_release,
                float(row[7]),
                float(row[8]),
                float(row[9]),
                float(row[10]),
                float(row[11]),
                float(row[12]) / self.fifo_cmax,
                float(row[13]) / max_release,
                float(row[14]) / self.fifo_cmax,
                float(row[15]) / self.fifo_cmax,
                float(row[16]) / self.fifo_cmax,
            ]
        return out

    @staticmethod
    def _zero_candidate_summary() -> List[float]:
        return [0.0] * 17

    def _ranker_job(self, model, candidates: Sequence[str]) -> str | None:
        if model is None or not candidates:
            return None
        try:
            import torch

            features = torch.tensor([extract_candidate_features(self.env, candidates)], dtype=torch.float32)
            with torch.no_grad():
                scores = model(features)[0]
            idx = int(torch.argmax(scores).item())
            return str(candidates[idx])
        except Exception:
            return None

    def _fifo_baseline_cmax(self) -> float:
        key = getattr(self.instance, "instance_id", None) or getattr(self.instance, "name", "instance")
        cache_path = self.baseline_cache_dir / f"{key}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            return float(cached["fifo_cmax"])

        trial = RollingSchedulingEnv(self.instance)
        trial.reset(self.instance)
        while not trial.is_done():
            ranked = fifo_ranked(trial)
            if not ranked:
                raise RuntimeError("FIFO baseline found no schedulable job.")
            job_id = ranked[0]
            trial.step((job_id, choose_split_num(trial, job_id)))

        self.baseline_cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"fifo_cmax": float(trial.current_cmax)}, f)
        return float(trial.current_cmax)

    def _machine_busy_loads(self) -> List[float]:
        busy = {machine.machine_id: 0.0 for machine in self.instance.machines}
        for subtask in self.env.subtasks:
            busy[subtask.machine_id] += float(subtask.duration)
        return list(busy.values()) or [0.0]

    def _info(self, extra: Dict | None = None) -> Dict:
        decision = self._build_decision()
        info = {
            "state_dim": self.state_dim if not self.env.is_done() else len(self._build_state(decision)),
            "action_dim": self.action_dim,
            "rule_names": list(self.rule_names),
            "action_mask": decision["action_mask"].tolist(),
            "candidates": list(decision["candidates"]),
            "rule_jobs": dict(decision["rule_jobs"]),
            "rule_distribution": self.rule_distribution(),
        }
        if extra:
            info.update(extra)
        return info

    @staticmethod
    def _first_in_candidates(ranked: Sequence[str], candidate_set: set[str]) -> str | None:
        for job_id in ranked:
            if job_id in candidate_set:
                return job_id
        return None

    @staticmethod
    def _agreement_ratio(rule_jobs: Dict[str, str | None]) -> float:
        jobs = [job for job in rule_jobs.values() if job is not None]
        if len(jobs) <= 1:
            return 1.0 if jobs else 0.0
        counts = Counter(jobs)
        return max(counts.values()) / float(len(jobs))

    @staticmethod
    def _try_load_model(path: str | None):
        if not path:
            return None
        model_path = Path(path)
        if not model_path.exists():
            print(f"Warning: ranker checkpoint not found, rule will be masked: {model_path}")
            return None
        return load_checkpoint(model_path)

