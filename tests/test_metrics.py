from src.baselines.heuristics import run_heuristic
from src.evaluation.metrics import compute_metrics
from src.instances.instance_generator import generate_instance


def test_metrics_have_expected_keys_and_ranges():
    env = run_heuristic(generate_instance("small", seed=5), "GreedyECT", seed=5).env
    metrics = compute_metrics(env)
    assert metrics["Cmax_roll"] > 0
    assert metrics["average_completion_time"] > 0
    assert metrics["average_waiting_time"] >= 0
    assert 0 <= metrics["machine_utilization"] <= 1
    assert 0 <= metrics["split_task_ratio"] <= 1
    assert metrics["total_split_count"] >= 0
