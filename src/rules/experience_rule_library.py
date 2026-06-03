"""Experience-guided rule library for HGCR-PPO Stage F2-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from candidate_generator import fifo_ranked, greedy_ect_ranked, lookahead_ranked, minload_ranked
from mlp_models import load_checkpoint
from src.baselines.heuristics import choose_split_num
from stage_c_utils import extract_candidate_features


RULE_NAMES = [
    "mlp_ranker_soft_ce",
    "fifo",
    "lookahead",
    "greedy_ect",
    "minload",
    "min_cmax_onestep",
]
RULE_TIEBREAK = {
    "mlp_ranker_soft_ce": 0,
    "fifo": 1,
    "min_cmax_onestep": 2,
    "lookahead": 3,
    "greedy_ect": 4,
    "minload": 5,
}


@dataclass
class RuleRecommendation:
    rule_id: int
    rule_name: str
    job_id: str | None
    candidate_index: int
    is_valid: bool
    score_or_proxy: float
    fallback_reason: str


class ExperienceRuleLibrary:
    """Map rule ids to valid HybridTopK candidate jobs."""

    def __init__(
        self,
        ranker_model=None,
        ranker_ckpt: str | None = None,
        device: str = "cpu",
        include_gnn_v1: bool = False,
    ):
        if include_gnn_v1:
            raise NotImplementedError("GNN v1 is reserved for future ablation and is disabled by default.")
        self.device = device
        self.ranker_model = ranker_model or (load_checkpoint(ranker_ckpt, device=device) if ranker_ckpt else None)
        self.rule_names = list(RULE_NAMES)

    @property
    def action_dim(self) -> int:
        return len(self.rule_names)

    def recommend(self, env, candidates: Sequence[str]) -> List[RuleRecommendation]:
        candidate_list = list(candidates)
        candidate_set = set(candidate_list)
        if not candidate_list:
            return [
                RuleRecommendation(idx, name, None, -1, False, float("inf"), "empty_candidate_set")
                for idx, name in enumerate(self.rule_names)
            ]

        raw_jobs = {
            "mlp_ranker_soft_ce": self._ranker_job(env, candidate_list),
            "fifo": self._first_in_candidates(fifo_ranked(env), candidate_set),
            "lookahead": self._first_in_candidates(lookahead_ranked(env), candidate_set),
            "greedy_ect": self._first_in_candidates(greedy_ect_ranked(env), candidate_set),
            "minload": self._first_in_candidates(minload_ranked(env), candidate_set),
            "min_cmax_onestep": self._min_cmax_onestep_job(env, candidate_list),
        }

        out: List[RuleRecommendation] = []
        for rule_id, rule_name in enumerate(self.rule_names):
            job_id = raw_jobs.get(rule_name)
            if job_id in candidate_set:
                idx = candidate_list.index(job_id)
                out.append(
                    RuleRecommendation(
                        rule_id=rule_id,
                        rule_name=rule_name,
                        job_id=job_id,
                        candidate_index=idx,
                        is_valid=True,
                        score_or_proxy=self.one_step_proxy(env, job_id),
                        fallback_reason="",
                    )
                )
            else:
                out.append(
                    RuleRecommendation(
                        rule_id=rule_id,
                        rule_name=rule_name,
                        job_id=None,
                        candidate_index=-1,
                        is_valid=False,
                        score_or_proxy=float("inf"),
                        fallback_reason="not_in_hybrid_topk_or_model_unavailable",
                    )
                )
        return out

    def action_mask(self, recommendations: Sequence[RuleRecommendation]) -> np.ndarray:
        return np.asarray([rec.is_valid for rec in recommendations], dtype=bool)

    def choose_job_or_fallback(self, recommendations: Sequence[RuleRecommendation], rule_id: int) -> tuple[str, int]:
        if 0 <= int(rule_id) < len(recommendations) and recommendations[int(rule_id)].is_valid:
            rec = recommendations[int(rule_id)]
            return str(rec.job_id), int(rec.rule_id)
        base = recommendations[0]
        if not base.is_valid:
            raise RuntimeError("Base rule mlp_ranker_soft_ce is invalid; check --ranker_ckpt and HybridTopK candidates.")
        return str(base.job_id), 0

    def label_from_proxy(self, recommendations: Sequence[RuleRecommendation]) -> RuleRecommendation:
        valid = [rec for rec in recommendations if rec.is_valid]
        if not valid:
            raise RuntimeError("No valid rule recommendation available for labeling.")
        return min(valid, key=lambda rec: (rec.score_or_proxy, RULE_TIEBREAK.get(rec.rule_name, 999), rec.rule_id))

    def one_step_proxy(self, env, job_id: str) -> float:
        trial = env.clone()
        trial.step((job_id, choose_split_num(trial, job_id)))
        return float(trial.current_cmax)

    def state_features(self, env, candidates: Sequence[str], recommendations: Sequence[RuleRecommendation]) -> List[float]:
        candidate_count = len(candidates)
        total_jobs = max(1, len(env.instance.jobs))
        machine_times = list(env.machine_available_time.values()) or [0.0]
        base = [
            float(env.current_cmax) / max(1.0, max(machine_times + [env.current_cmax])),
            len(env.unscheduled_jobs) / total_jobs,
            candidate_count / max(1, candidate_count),
            float(np.mean(machine_times)),
            float(np.std(machine_times)),
        ]
        rec_features = []
        max_proxy = max([rec.score_or_proxy for rec in recommendations if rec.is_valid] or [1.0])
        for rec in recommendations:
            rec_features.extend(
                [
                    1.0 if rec.is_valid else 0.0,
                    rec.candidate_index / max(1, candidate_count - 1) if rec.candidate_index >= 0 else 0.0,
                    rec.score_or_proxy / max(1.0, max_proxy) if rec.is_valid else 0.0,
                ]
            )
        return [float(value) for value in [*base, *rec_features]]

    def _ranker_job(self, env, candidates: Sequence[str]) -> str | None:
        if self.ranker_model is None or not candidates:
            return None
        try:
            import torch

            features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                scores = self.ranker_model(features)[0]
            return str(candidates[int(torch.argmax(scores).item())])
        except Exception:
            return None

    def _min_cmax_onestep_job(self, env, candidates: Sequence[str]) -> str | None:
        if not candidates:
            return None
        return min(candidates, key=lambda job_id: (self.one_step_proxy(env, job_id), job_id))

    @staticmethod
    def _first_in_candidates(ranked: Sequence[str], candidate_set: set[str]) -> str | None:
        for job_id in ranked:
            if job_id in candidate_set:
                return job_id
        return None

