import os
import multiprocessing as mp

def subprocess_target(fn_name, config, platform):
    os.environ["JAX_PLATFORMS"] = platform
    import benchmark as bm
    fn = getattr(bm, fn_name)
    fn(config)


def run_in_process(fn_name, config, platform):
    p = mp.Process(target=subprocess_target, args=(fn_name, config, platform))
    p.start()
    p.join()


def print_device():
    import jax
    print(f"Device: {jax.devices()[0].platform}")


# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_atlas(config):
    print_device()
    from data.atlas import Atlas
    atlas = Atlas(config.atlas)
    atlas.benchmark(config.atlas, n_rounds=config.benchmark.n_rounds_atlas)


def benchmark_background(config):
    # TODO: implement background benchmark
    raise NotImplementedError("Needs to be implemented in the future.")


def benchmark_generator(config):
    print_device()
    from data.generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(config, N_images=config.benchmark.batch_size_generator)
    gen.benchmark(
        n_rounds=config.benchmark.n_rounds_generator,
        n_warmups=config.benchmark.n_warmups_generator,
        device=config.benchmark.device_generator,
    )


def benchmark_loader(config):
    print_device()
    from model.loader import MultiProcessLoader
    loader = MultiProcessLoader(config)
    loader.benchmark(n_rounds=config.benchmark.n_rounds_loader)
    loader.stop_workers()


def benchmark_network(config):
    print_device()
    from model.loader import MultiProcessLoader
    from model.network import NeuralNetwork
    loader = MultiProcessLoader(config)
    loader.start_workers()
    model = NeuralNetwork(config, loader=loader)
    model.benchmark(
        n_rounds=config.benchmark.n_rounds_network,
        n_warmups=config.benchmark.n_warmups_network,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config):
    b   = config.benchmark
    ran = False

    if b.benchmark_atlas:
        ran = True
        run_in_process("benchmark_atlas", config, b.device_atlas)

    if b.benchmark_background:
        ran = True
        run_in_process("benchmark_background", config, b.device_background)

    if b.benchmark_generator:
        ran = True
        run_in_process("benchmark_generator", config, b.device_generator)

    if b.benchmark_loader:
        ran = True
        run_in_process("benchmark_loader", config, b.device_loader)

    if b.benchmark_network:
        ran = True
        run_in_process("benchmark_network", config, b.device_network)

    if not ran:
        print("Nothing to benchmark.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    from configs.default import CFG_BENCHMARK
    main(CFG_BENCHMARK)