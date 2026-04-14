import os
if os.environ.get("JAX_PLATFORMS") is None:
    os.environ["JAX_PLATFORMS"] = "cpu"

import atexit
import queue
import signal
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


class MultiProcessLoader:
    def __init__(self, config, *args, **kwargs):
        from configs.configurations import TrainConfig, BenchmarkConfig
        if isinstance(config, (TrainConfig, BenchmarkConfig)):
            global_config = config.g
            loader_config = config.loader

            self.n_workers   = loader_config.N_workers
            self.n_chunks    = loader_config.N_chunks
            self.n_batches   = global_config.batch_size
            self.buffer_size = loader_config.buffer_size

            self.data_generator      = loader_config.data_generator
            self.data_generator_init = loader_config.data_generator_init

            self.H         = global_config.H
            self.W         = global_config.W
            self.C         = global_config.C
            self.seed      = global_config.seed
            self.N_targets = 2
            self.config    = config
        else:
            self.n_workers           = kwargs.get("n_workers", 4)
            self.n_chunks            = kwargs.get("n_chunks", 4)
            self.n_batches           = kwargs.get("n_batches", 4)
            self.buffer_size         = kwargs.get("buffer_size", 2)
            self.seed                = kwargs.get("seed", 0)
            self.data_generator      = kwargs["data_generator"]
            self.data_generator_init = kwargs.get("data_generator_init", None)
            self.config              = config

            if self.config is None:
                raise ValueError("Insert configs for data generator")

            self.H         = self.config["H"]
            self.W         = self.config["W"]
            self.C         = self.config["C"]
            self.N_targets = 2

        self.shape_features = (self.buffer_size, self.n_chunks, self.n_batches, self.H, self.W, self.C)
        self.shape_targets  = (self.buffer_size, self.n_chunks, self.n_batches, self.H, self.W, self.N_targets)

        self.dtype = np.float32

        nbytes_features   = int(np.prod(self.shape_features) * np.dtype(self.dtype).itemsize)
        self.shm_features = shared_memory.SharedMemory(create=True, size=nbytes_features)
        self.features_array = np.ndarray(self.shape_features, dtype=self.dtype, buffer=self.shm_features.buf)

        nbytes_targets   = int(np.prod(self.shape_targets) * np.dtype(self.dtype).itemsize)
        self.shm_targets = shared_memory.SharedMemory(create=True, size=nbytes_targets)
        self.targets_array = np.ndarray(self.shape_targets, dtype=self.dtype, buffer=self.shm_targets.buf)

        self.task_queue     = mp.Queue()
        self.ready_queue    = mp.Queue()
        self.stop_event     = mp.Event()
        self.pending_counts = mp.Array("i", self.buffer_size, lock=True)
        self.workers        = []
        self.cleaned_up     = False

        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT,  self.signal_handler)

    def signal_handler(self, *args, **kwargs):
        print("\nCleaning up shared memory.")
        self.cleanup()
        sys.exit(1)

    @staticmethod
    def worker(
        rng_seed, shm_features, shm_targets,
        shape_features, shape_targets, dtype,
        task_queue, ready_queue, stop_event, pending_counts,
        data_generator, data_generator_init, config, i,
    ):
        os.environ["JAX_PLATFORMS"] = "cpu"
        _CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "loader")
        import jax
        print(f"Worker {i} devices: {jax.devices()}")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR
        jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

        shm_feat = shared_memory.SharedMemory(name=shm_features)
        shm_tgt  = shared_memory.SharedMemory(name=shm_targets)

        features_array = np.ndarray(shape_features, dtype=dtype, buffer=shm_feat.buf)
        targets_array  = np.ndarray(shape_targets,  dtype=dtype, buffer=shm_tgt.buf)

        rng = np.random.default_rng([rng_seed, i])

        worker_state = None
        if data_generator_init is not None:
            worker_state = data_generator_init(config, i, True)

        try:
            while True:
                try:
                    job = task_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if job is None:
                    break

                buffer_idx, chunk_idx = job
                seed = int(rng.integers(0, 2**31))

                features, targets = data_generator(seed, worker_state)

                features_array[buffer_idx, chunk_idx] = np.asarray(features, dtype=dtype)
                targets_array[buffer_idx, chunk_idx]  = np.asarray(targets,  dtype=dtype)

                with pending_counts.get_lock():
                    pending_counts[buffer_idx] -= 1
                    if pending_counts[buffer_idx] == 0:
                        ready_queue.put(buffer_idx)
        finally:
            shm_feat.close()
            shm_tgt.close()

    def start_workers(self):
        if self.workers:
            return
        for i in range(self.n_workers):
            p = mp.Process(
                target=MultiProcessLoader.worker,
                args=(
                    self.seed,
                    self.shm_features.name,
                    self.shm_targets.name,
                    self.shape_features,
                    self.shape_targets,
                    self.dtype,
                    self.task_queue,
                    self.ready_queue,
                    self.stop_event,
                    self.pending_counts,
                    self.data_generator,
                    self.data_generator_init,
                    self.config,
                    i,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        for buffer_idx in range(self.buffer_size):
            self.schedule_buffer_fill(buffer_idx)

    def schedule_buffer_fill(self, buffer_idx):
        with self.pending_counts.get_lock():
            self.pending_counts[buffer_idx] = self.n_chunks
        for chunk_idx in range(self.n_chunks):
            self.task_queue.put((buffer_idx, chunk_idx))

    def acquire_buffer(self):
        buffer_idx = self.ready_queue.get()
        return (
            buffer_idx,
            self.features_array[buffer_idx],
            self.targets_array[buffer_idx],
        )

    def release_buffer(self, buffer_idx):
        self.schedule_buffer_fill(buffer_idx)

    def stop_workers(self):
        if not self.workers:
            return

        for _ in self.workers:
            self.task_queue.put(None)

        for w in self.workers:
            w.join()

        self.workers.clear()

        self.shm_features.close()
        self.shm_features.unlink()

        self.shm_targets.close()
        self.shm_targets.unlink()

    def cleanup(self):
        if self.cleaned_up:
            return
        self.cleaned_up = True
        self.stop_workers()

    def benchmark(self, n_rounds=20):
        print(f"\n{'='*60}")
        print(f"{'Benchmark: Loader':^60}")
        print(f"{'='*60}")
        print(f"  rounds         : {n_rounds}")
        print(f"  n_workers      : {self.n_workers}")
        print(f"  n_chunks       : {self.n_chunks}")
        print(f"  n_batches      : {self.n_batches}")
        print(f"  buffer_size    : {self.buffer_size}")
        print(f"  features shape : {self.shape_features[1:]}")
        print(f"  targets shape  : {self.shape_targets[1:]}")
        print(f"{'─'*60}")

        print(f"{f'Pre jitting {self.n_workers} workers':^60}")

        t_warmup = time.perf_counter()
        self.start_workers()
        buf_idx, features, targets = self.acquire_buffer()
        print(f"{'─'*60}")
        print(f"  First buffer ready in {time.perf_counter() - t_warmup:.1f}s")
        self.release_buffer(buf_idx)
        time.sleep(0.5)

        print(f"{'─'*60}")
        print(f"  {'round':>5}  {'acquire(s)':>10}  {'release(s)':>10}  {'samp/s':>9}")
        print(f"{'─'*60}")

        samples_per_buffer = self.n_chunks * self.n_batches

        benchmark_times = {
            "acquire": [],
            "release": [],
            "total":   [],
        }

        start_time = time.perf_counter()
        for i in range(n_rounds):
            t0 = time.perf_counter()
            buf_idx, features, targets = self.acquire_buffer()
            t1 = time.perf_counter()
            benchmark_times["acquire"].append(t1 - t0)

            self.release_buffer(buf_idx)
            t2 = time.perf_counter()
            benchmark_times["release"].append(t2 - t1)
            benchmark_times["total"].append(t2 - t0)

            wait = t1 - t0
            sps  = samples_per_buffer / wait if wait > 0 else float("inf")
            print(f"  {i+1:>5}  {(t1-t0):>10.4f}  {(t2-t1):>10.4f}  {sps:>9.1f}")

        end_time = time.perf_counter()

        steady        = benchmark_times["acquire"][1:]
        avg           = float(np.mean(steady))
        std           = float(np.std(steady))
        best          = float(np.min(steady))
        worst         = float(np.max(steady))
        avg_sps       = samples_per_buffer / avg if avg > 0 else float("inf")
        total_elapsed = end_time - start_time

        print(f"{'─'*60}")
        print(f"Finished {n_rounds} rounds of loading data in {total_elapsed:.2f}s")
        print(f"{'─'*60}")
        print(f"{'Averages and stds':^60}")
        print(f"{'─'*60}")
        for k, v in benchmark_times.items():
            arr = np.array(v)
            print(f"{k:>20}: {arr.mean():.4f}s ± {arr.std():.4f}s")
        print(f"{'─'*60}")
        print(f"  steady-state wait  : {avg:.3f}s ± {std:.3f}s")
        print(f"  best / worst       : {best:.3f}s / {worst:.3f}s")
        print(f"  avg throughput     : {avg_sps:.1f} samples/s")
        print(f"{'─'*60}")

def data_generator_init(config, i, worker_init=True):
    from data.generator import SyntheticDataGenerator
    worker_data_generator = SyntheticDataGenerator(config, worker_init, i)
    return worker_data_generator

def data_generator(seed, worker_data_generator):
    rng = np.random.default_rng(seed)
    features, targets = worker_data_generator.generate_batch(rng=rng)
    return features, targets

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    from configs.default import CFG_TRAIN
    config = CFG_TRAIN
    loader = MultiProcessLoader(config)

    print("Starting workers (Atlas build + JIT warm-up per worker)")
    t0 = time.time()
    loader.start_workers()
    loader.benchmark()
    loader.stop_workers()