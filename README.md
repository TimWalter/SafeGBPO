# Safeguarding Gradient-Based Policy Optimisation 

This is the codebase from the paper "Safeguarding Gradient-Based Policy Optimisation" 
## Usage

To train an agent, use the `train.py` script and fill the experiment queue:

```bash
python src/train.py
```



## Configuration

The framework uses Hydra for configuration management. This projects uses so-called *structured configs*, which means they are **typed** python dataclasses, if a type hint is **not present** the parameter is **ignored**. The *name* parameter is generally used to determine the class and therefore has to **match the class name exactly**. The main configuration file is `conf/config.py`, which includes other configuration files for specific components.

### Environment & Algorithm Configuration

Environment configurations are stored in `conf/env/`, while algorithm configurations are stored in `conf/algorithm/`. To use a specific environment and algorithm, specify it in the `config.py` file:

```python
defaults: list[Any] = field(default_factory=lambda: [
    "_self_",
    {"algorithm": "SHAC"},
    {"env": "CartPole"}
])
```
or utilise the `train.py` script.

## Extending the experiments

### Adding New Environments

1. Create a new environment class in `src/envs/` that inherits from `VectorTorchEnv`.
2. Implement the required methods, including `step()` and `reset()`.
3. Create a configuration file for the new environment in `conf/env/`.
4. Create a respective task that inherits from the environment and a safety specificiation in `src/tasks/`.
5. Create a configuration file for the new task in `conf/task/`.
5. Select the new task in the `config.py` file.
6. Adapt the logging procedure if wanted in [src/callbacks/train_callback.py](src/callbacks/train_callback.py)

### Adding New Algorithms

1. Create a new algorithm class in `src/algorithms/` that inherits from `ActorCriticAlgorithm`.
2. Implement the required methods, including `_learn_episode()`.
3. Create a configuration file for the new algorithm in `conf/algorithm/`.
4. Select the new algorithm in the `config.py` file.
5. Adapt the logging procedure if wanted in [src/callbacks/train_callback.py](src/callbacks/train_callback.py)

### Adding New Wrappers

1. Create a new wrapper class in `src/wrappers/` that inherits from `gymnasium.vector.VectorWrapper` or `SafetyWrapper`.
2. Implement the required methods.
3. Create a configuration file for the new wrapper in `conf/wrapper/`.
4. To use the new wrapper, add it to the `wrappers` list in the environment configuration file.

## Hyperparameter Optimization

To add hyperparameter search spaces for new algorithms, modify the `search_spaces.py` file: [src/search_spaces.py](src/search_spaces.py#L5-43)
Add a new function for your algorithm and include it in the `modifications` dictionary at the end of the file.
## Logging and Visualization

The framework uses Weights & Biases (wandb) for logging and visualization. Configure WandB in the section 
in the `config.py` file: [conf/config.py](conf/config.py#L7-14). Moreover, the main logging mechanisms
can be found in TrainCallback: [src/callbacks/train_callback.py](src/callbacks/train_callback.py). The login key has to be provided in `train.py`.

## Testing

Run the tests using pytest:

```bash
pytest tests/
```

This will run all tests.
