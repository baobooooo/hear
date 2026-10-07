"""Opt-in vLLM worker: wait for admission without repeating CUDA initialization."""
import math
import time

from vllm.v1.worker import gpu_worker


def wait_for_memory(request_memory, snapshot, cache_config, *, timeout=3600,
                    sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout
    announced = False
    while True:
        try:
            return request_memory(snapshot, cache_config)
        except ValueError as error:
            required = math.ceil(snapshot.total_memory * cache_config.gpu_memory_utilization)
            if (snapshot.free_memory >= required
                    or not str(error).startswith('Free memory on device')):
                raise
            if clock() >= deadline:
                raise TimeoutError('GPU memory admission deadline exceeded') from error
            if not announced:
                print('Waiting for GPU memory; retaining initialized CUDA context and original memory requirement.', flush=True)
                announced = True
            sleep(1)
            # Refresh the same object also held by Worker.init_snapshot, so later
            # profiling does not use the earlier contended-memory baseline.
            snapshot.measure()


class WaitForMemoryWorker(gpu_worker.Worker):
    def init_device(self):
        original = gpu_worker.request_memory
        gpu_worker.request_memory = lambda snapshot, config: wait_for_memory(original, snapshot, config)
        try:
            return super().init_device()
        finally:
            gpu_worker.request_memory = original
