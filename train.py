import os
os.environ["JAX_PLATFORMS"]       = "cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"]            = "--xla_gpu_autotune_level=4"

import multiprocessing as mp


def main(config):
    from model.loader import MultiProcessLoader
    from model.network import NeuralNetwork

    if config.loader is not None:
        loader = MultiProcessLoader(config)
        loader.start_workers()
        model = NeuralNetwork(config, loader=loader)
    elif config.data is not None:
        raise NotImplementedError("Implement in the future for in memory dataset training.")
    else:
        raise ValueError("Insert type of data")
    try:
        model.train()
    except KeyboardInterrupt:
        print("Training interrupted")
        model.save_weights()
    finally:
        model.plot_training_history(log_scale=False)
        model.plot_lambdas_history()
        buf_idx, features, targets = loader.acquire_buffer()
        for i in range(model.train_batch_size):
            model.plot_predictions(features[0, i], targets[0, i])
        loader.release_buffer(buf_idx)
        loader.stop_workers()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    from configs.default import CFG_TRAIN
    import jax
    print(f"Main process devices: {jax.devices()}")

    main(CFG_TRAIN)