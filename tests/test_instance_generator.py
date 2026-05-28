from src.instances.instance_generator import generate_instance


def test_instance_generation_is_reproducible():
    a = generate_instance("small", seed=7)
    b = generate_instance("small", seed=7)
    assert [job.release_time for job in a.jobs] == [job.release_time for job in b.jobs]
    assert a.processing_time == b.processing_time


def test_jobs_only_have_same_process_candidates():
    instance = generate_instance("medium", seed=3)
    machine_by_id = instance.machine_by_id
    for job in instance.jobs:
        assert job.candidate_machines
        for machine_id in job.candidate_machines:
            assert machine_by_id[machine_id].process_type == job.process_type
            assert machine_id in instance.processing_time[job.job_id]
