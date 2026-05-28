from src.baselines.heuristics import run_heuristic
from src.instances.instance_generator import generate_instance


def test_fifo_schedule_is_feasible():
    instance = generate_instance("small", seed=11)
    result = run_heuristic(instance, "FIFO", seed=11)
    env = result.env
    machine_by_id = instance.machine_by_id

    for job_schedule in env.scheduled_jobs.values():
        job = env.job_by_id[job_schedule.job_id]
        assert abs(sum(subtask.ratio for subtask in job_schedule.subtasks) - 1.0) < 1e-9
        assert job_schedule.completion_time == max(subtask.completion_time for subtask in job_schedule.subtasks)
        for subtask in job_schedule.subtasks:
            assert subtask.machine_id in job.candidate_machines
            assert machine_by_id[subtask.machine_id].process_type == job.process_type
            assert subtask.start_time >= job.release_time
            assert subtask.ratio > 0

    for machine in instance.machines:
        tasks = sorted([s for s in env.subtasks if s.machine_id == machine.machine_id], key=lambda s: s.start_time)
        for left, right in zip(tasks, tasks[1:]):
            assert left.completion_time <= right.start_time + 1e-9

    assert env.current_cmax == max(job.completion_time for job in env.scheduled_jobs.values())
