from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_method
import hydra
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver

def initialize_benchmark_drivers(benchmark: str, mode: str):
    """Initialize benchmark drivers based on benchmark type and mode."""
    if benchmark == "ParEval":
        driver = ParEvalDriver(mode)
        evaldriver = ParEvalDriver(mode)
    elif benchmark == "PolyBench":
        driver = PolyBenchDriver(mode)
        evaldriver = PolyBenchDriver(mode)
    else:
        print("Unknown Benchmark, program exits.")
        exit(0)

    return driver, evaldriver

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    """
    cfg.methodology now *is* the DictConfig that lives in
    configs/methodology/<chosen>.yaml, including its _target_.
    """
    # print("Selected strategy:", cfg.strategy._target_)

    print("Strategy selected:", cfg._target_.split('.')[1])
    print("Target function :", cfg._target_)

    print("Current Configuration: ")
    print(OmegaConf.to_yaml(cfg, resolve=True))   # nice for debugging

    # Initialize benchmark drivers
    driver, evaldriver = initialize_benchmark_drivers(cfg.benchmark, cfg.mode)

    # Add drivers to config so strategy can access them
    # Old approach: doesn't work because OmegaConf doesn't support non-primitive types
    # OmegaConf.set_struct(cfg, False)
    # cfg.driver = driver
    # cfg.evaldriver = evaldriver
    # OmegaConf.set_struct(cfg, True)

    # New approach: bypass OmegaConf type checking using object.__setattr__
    object.__setattr__(cfg, 'driver', driver)
    object.__setattr__(cfg, 'evaldriver', evaldriver)

    run_strategy_fn = get_method(cfg._target_)

    run_strategy_fn(cfg)

    # Hydra 1.2+ : call() will import the target and execute it,
    # passing the *methodology* subtree (not the whole cfg) as argument.
    # call(cfg.strategy)

if __name__ == "__main__":
    main()