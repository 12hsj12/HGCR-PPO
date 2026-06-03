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
DELTA_RULES = [
    "keep_ranker",
    "switch_to_fifo",
    "switch_to_lookahead",
    "switch_to_greedy_ect",
    "switch_to_minload",
    "switch_to_pairwise_ranker",
]
ACTION_MODES = ["rule_selector", "delta_rule"]
CONSERVATIVE_MODES = ["none", "ranker_fallback", "ranker_penalty"]
BASELINE_TYPES = ["fifo", "ranker"]
RANKER_RULES = {"mlp_ranker_soft_ce", "mlp_ranker_pairwise", "keep_ranker", "switch_to_pairwise_ranker"}


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
        ranker_baseline_cache_dir: str | Path = "data/cache/stage_F/ranker_baselines",
        action_mode: str = "rule_selector",
        conservative_mode: str = "none",
        fallback_threshold: float = 0.6,
        switch_penalty: float = 0.01,
        baseline_type: str = "fifo",
        include_pairwise_ranker: bool = False,
        seed: int = 42,
    ):
        if action_mode not in ACTION_MODES:
            raise ValueError(f"Unknown action_mode {action_mode!r}. Expected one of {ACTION_MODES}.")
        if conservative_mode not in CONSERVATIVE_MODES:
            raise ValueError(f"Unknown conservative_mode {conservative_mode!r}. Expected one of {CONSERVATIVE_MODES}.")
        if baseline_type not in BASELINE_TYPES:
            raise ValueError(f"Unknown baseline_type {baseline_type!r}. Expected one of {BASELINE_TYPES}.")
        self.top_k = max(1, int(top_k))
        self.rng = random.Random(seed)
        self.baseline_cache_dir = Path(baseline_cache_dir)
        self.ranker_baseline_cache_dir = Path(ranker_baseline_cache_dir)
        self.action_mode = action_mode
        self.conservative_mode = conservative_mode
        self.fallback_threshold = float(fallback_threshold)
        self.switch_penalty = float(switch_penalty)
        self.baseline_type = baseline_type
        self.include_pairwise_ranker = bool(include_pairwise_ranker)
        self.mlp_soft_model = self._try_load_model(mlp_soft_model_path)
        self.mlp_pairwise_model = self._try_load_model(mlp_pairwise_model_path)
        if action_mode == "delta_rule":
            self.rule_names = list(DELTA_RULES if self.include_pairwise_ranker else DELTA_RULES[:-1])
        else:
            self.rule_names = list(ALL_RULES)
        self.env = RollingSchedulingEnv(instance)
        self.instance = self.env.instance
        self.process_types = process_types_for_instance(self.instance)
        self.fifo_cmax = 1.0
        self.ranker_cmax = 1.0
        self.baseline_cmax = 1.0
        self.last_decision: Dict = {}
        self.rule_counts: Counter[str] = Counter()
        self.raw_rule_counts: Counter[str] = Counter()
        self.fallback_count = 0
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
        self.ranker_cmax = max(1e-6, self._ranker_baseline_cmax()) if self.baseline_type == "ranker" else self.fifo_cmax
        self.baseline_cmax = self.ranker_cmax if self.baseline_type == "ranker" else self.fifo_cmax
        self.last_decision = {}
        self.rule_counts = Counter()
        self.raw_rule_counts = Counter()
        self.fallback_count = 0
        return self._build_state(self._build_decision())

    def step(self, action: int, action_probability: float | None = None):
        if self.env.is_done():
            state = self._build_state(self._build_decision())
            return state, 0.0, True, self._info({"final_makespan": self.env.current_cmax})

        decision = self._build_decision()
        mask = decision["action_mask"]
        rule_id = int(action)
        if rule_id < 0 or rule_id >= len(self.rule_names) or not bool(mask[rule_id]):
            raise ValueError(f"Invalid or masked rule action {action}. mask={mask.tolist()}")

        raw_action_name = self.rule_names[rule_id]
        raw_rule_name = self._target_rule_for_action(raw_action_name)
        executed_rule_name = self._resolve_executed_rule(raw_rule_name, decision, action_probability)
        executed_action_name = self._action_name_for_rule(executed_rule_name)
        job_id = decision["rule_jobs"][executed_rule_name]
        old_cmax = float(self.env.current_cmax)
        _, _, done, env_info = self.env.step((job_id, choose_split_num(self.env, job_id)))
        new_cmax = float(self.env.current_cmax)

        reward = -(new_cmax - old_cmax) / self.baseline_cmax
        if self.conservative_mode == "ranker_penalty" and not self._is_ranker_rule(raw_rule_name):
            reward -= self.switch_penalty
        if done:
            reward += -(new_cmax - self.baseline_cmax) / self.baseline_cmax

        self.raw_rule_counts[raw_action_name] += 1
        self.rule_counts[executed_action_name] += 1
        info = self._info(
            {
                **env_info,
                "selected_rule": executed_action_name,
                "selected_rule_id": rule_id,
                "raw_rule": raw_action_name,
                "executed_rule": executed_action_name,
                "raw_target_rule": raw_rule_name,
                "executed_target_rule": executed_rule_name,
                "selected_job_id": job_id,
                "step_reward": reward,
                "old_cmax": old_cmax,
                "new_cmax": new_cmax,
                "fifo_cmax": self.fifo_cmax,
                "ranker_cmax": self.ranker_cmax,
                "baseline_cmax": self.baseline_cmax,
                "baseline_type": self.baseline_type,
                "fallback_count": self.fallback_count,
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

    def raw_rule_distribution(self) -> Dict[str, float]:
        total = sum(self.raw_rule_counts.values())
        if total <= 0:
            return {rule: 0.0 for rule in self.rule_names}
        return {rule: float(self.raw_rule_counts[rule]) / total for rule in self.rule_names}

    def diagnostics(self) -> Dict[str, float | Dict[str, float]]:
        executed = self.rule_distribution()
        raw = self.raw_rule_distribution()
        total_decisions = sum(self.rule_counts.values())
        total = max(1, total_decisions)
        ranker_ratio = sum(value for rule, value in executed.items() if self._is_ranker_action(rule))
        return {
            "ppo_raw_rule_distribution": raw,
            "executed_rule_distribution": executed,
            "executed_rule_counts": {rule: float(self.rule_counts[rule]) for rule in self.rule_names},
            "total_decisions": float(total_decisions),
            "rule_ranker_ratio": ranker_ratio,
            "rule_non_ranker_ratio": max(0.0, 1.0 - ranker_ratio),
            "fallback_count": float(self.fallback_count),
            "fallback_ratio": float(self.fallback_count) / total,
            "keep_ranker_ratio": executed.get("keep_ranker", 0.0),
            "switch_to_fifo_ratio": executed.get("switch_to_fifo", 0.0),
            "switch_to_lookahead_ratio": executed.get("switch_to_lookahead", 0.0),
            "switch_to_greedy_ratio": executed.get("switch_to_greedy_ect", 0.0),
            "switch_to_minload_ratio": executed.get("switch_to_minload", 0.0),
            "switch_to_pairwise_ratio": executed.get("switch_to_pairwise_ranker", 0.0),
            "effective_switch_ratio": sum(
                executed.get(rule, 0.0)
                for rule in [
                    "switch_to_fifo",
                    "switch_to_lookahead",
                    "switch_to_greedy_ect",
                    "switch_to_minload",
                    "switch_to_pairwise_ranker",
                ]
            ),
        }

    def _build_decision(self) -> Dict:
        if self.env.is_done():
            empty_mask = np.zeros(len(self.rule_names), dtype=bool)
            return {
                "candidates": [],
                "rule_jobs": {rule: None for rule in ALL_RULES},
                "action_jobs": {rule: None for rule in self.rule_names},
                "action_mask": empty_mask,
            }

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
        action_jobs = {action_name: rule_jobs[self._target_rule_for_action(action_name)] for action_name in self.rule_names}
        mask = np.array([action_jobs[rule] is not None for rule in self.rule_names], dtype=bool)
        return {"candidates": candidates, "rule_jobs": rule_jobs, "action_jobs": action_jobs, "action_mask": mask}

    def _build_state(self, decision: Dict) -> np.ndarray:
        candidates = decision["candidates"]
        now = float(self.env._current_decision_time())
        max_release = max((float(job.release_time) for job in self.instance.jobs), default=1.0) or 1.0
        max_jobs = max(1, len(self.instance.jobs))
        cmax_norm = float(self.env.current_cmax) / self.baseline_cmax
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
            mean(machine_times) / self.baseline_cmax,
            (pstdev(machine_times) if len(machine_times) > 1 else 0.0) / self.baseline_cmax,
            (pstdev(loads) if len(loads) > 1 else 0.0) / self.baseline_cmax,
            len(candidates) / float(self.top_k),
            self._agreement_ratio(decision["rule_jobs"]),
        ]

        candidate_features = self._candidate_feature_map(candidates)
        state: List[float] = [*global_features, *process_counts]
        for rule in self.rule_names:
            job_id = decision["action_jobs"].get(rule)
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
                float(row[12]) / self.baseline_cmax,
                float(row[13]) / max_release,
                float(row[14]) / self.baseline_cmax,
                float(row[15]) / self.baseline_cmax,
                float(row[16]) / self.baseline_cmax,
            ]
        return out

    @staticmethod
    def _zero_candidate_summary() -> List[float]:
        return [0.0] * 17

    def _ranker_job(self, model, candidates: Sequence[str]) -> str | None:
        return self._ranker_job_for_env(model, self.env, candidates)

    @staticmethod
    def _ranker_job_for_env(model, env, candidates: Sequence[str]) -> str | None:
        if model is None or not candidates:
            return None
        try:
            import torch

            features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32)
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

    def _ranker_baseline_cmax(self) -> float:
        if self.mlp_soft_model is None:
            raise RuntimeError(
                "baseline_type='ranker' requires a valid --mlp_soft_model_path so ranker_Cmax can be computed."
            )
        key = getattr(self.instance, "instance_id", None) or getattr(self.instance, "name", "instance")
        cache_path = self.ranker_baseline_cache_dir / f"{key}_topk{self.top_k}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            return float(cached["ranker_cmax"])

        trial = RollingSchedulingEnv(self.instance)
        trial.reset(self.instance)
        while not trial.is_done():
            candidates = generate_candidates(
                trial,
                candidate_mode="hybrid_topk",
                top_k=self.top_k,
                fallback_to_all=True,
            )
            job_id = self._ranker_job_for_env(self.mlp_soft_model, trial, candidates)
            if job_id is None:
                raise RuntimeError(
                    "Failed to compute ranker baseline: MLP-Ranker did not return a valid job "
                    f"for instance {key}."
                )
            trial.step((job_id, choose_split_num(trial, job_id)))

        self.ranker_baseline_cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"ranker_cmax": float(trial.current_cmax)}, f)
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
            "action_jobs": dict(decision["action_jobs"]),
            "rule_distribution": self.rule_distribution(),
            "ppo_raw_rule_distribution": self.raw_rule_distribution(),
            "executed_rule_distribution": self.rule_distribution(),
            "action_mode": self.action_mode,
            "conservative_mode": self.conservative_mode,
            "baseline_type": self.baseline_type,
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

    def _resolve_executed_rule(self, raw_rule_name: str, decision: Dict, action_probability: float | None) -> str:
        ranker_rule = "mlp_ranker_soft_ce"
        should_fallback = (
            self.conservative_mode == "ranker_fallback"
            and not self._is_ranker_rule(raw_rule_name)
            and decision["rule_jobs"].get(ranker_rule) is not None
            and action_probability is not None
            and float(action_probability) < self.fallback_threshold
        )
        if should_fallback:
            self.fallback_count += 1
            return ranker_rule
        return raw_rule_name

    def _target_rule_for_action(self, action_name: str) -> str:
        if self.action_mode == "rule_selector":
            return action_name
        return {
            "keep_ranker": "mlp_ranker_soft_ce",
            "switch_to_fifo": "fifo",
            "switch_to_lookahead": "lookahead",
            "switch_to_greedy_ect": "greedy_ect",
            "switch_to_minload": "minload",
            "switch_to_pairwise_ranker": "mlp_ranker_pairwise",
        }[action_name]

    def _action_name_for_rule(self, rule_name: str) -> str:
        if self.action_mode == "rule_selector":
            return rule_name
        return {
            "mlp_ranker_soft_ce": "keep_ranker",
            "fifo": "switch_to_fifo",
            "lookahead": "switch_to_lookahead",
            "greedy_ect": "switch_to_greedy_ect",
            "minload": "switch_to_minload",
            "mlp_ranker_pairwise": "switch_to_pairwise_ranker",
        }[rule_name]

    @staticmethod
    def _is_ranker_rule(rule_name: str) -> bool:
        return rule_name in {"mlp_ranker_soft_ce", "mlp_ranker_pairwise"}

    @staticmethod
    def _is_ranker_action(action_name: str) -> bool:
        return action_name in RANKER_RULES

    @staticmethod
    def _try_load_model(path: str | None):
        if not path:
            return None
        model_path = Path(path)
        if not model_path.exists():
            print(f"Warning: ranker checkpoint not found, rule will be masked: {model_path}")
            return None
        return load_checkpoint(model_path)

