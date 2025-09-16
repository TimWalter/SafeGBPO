import torch
import wandb
import torchvision.utils
from beartype import beartype
from jaxtyping import jaxtyped
from optuna import Trial, TrialPruned
from wandb.sdk.wandb_run import Run

from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from envs.simulators.interfaces.simulator import Simulator


class Logger:
    """
    Logs evaluation and training data to Weights and Biases.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 agent: LearningAlgorithm,
                 env: Simulator,
                 eval_env: Simulator,
                 wandb_run: Run,
                 optuna_trial: Trial | None,
                 eval_freq: int,
                 fast_eval: bool):
        """
        Initializes the Logger.
        Args:
            agent: The model to train.
            env: The training environment.
            eval_env: The environment to evaluate the policy on.
            wandb_run: The Weights and Biases run object.
            optuna_trial: The Optuna trial object.
            eval_freq: The frequency at which to evaluate the policy.
            fast_eval (bool): Whether to produce video only on the final episode
        """
        self.model = agent
        self.env = env
        self.eval_env = eval_env
        self.wandb_run = wandb_run
        self.optuna_trial = optuna_trial
        self.eval_freq = eval_freq
        self.fast_eval = fast_eval

        self.best_reward = -torch.inf
        self.log_data = {}
        self.last_eval = 0

    @jaxtyped(typechecker=beartype)
    def on_learning_episode(self,
                            eps: int,
                            average_reward: float,
                            policy_loss: float,
                            value_loss: float,
                            num_learn_episodes: int):
        """
        Callback call to log and evaluate.

        Args:
            eps: Number of the current learning episode
            average_reward: The average reward of the current episode
            policy_loss: The policy loss
            value_loss: The value loss
            num_learn_episodes: The total number of learning episodes
        """
        self.log_data["train/Average Reward"] = average_reward
        if hasattr(self.env, "interventions"):
            self.log_data["train/Interventions"] = self.env.interventions
        for i, val in enumerate(self.model.policy.log_std.detach().cpu().numpy()):
            self.log_data[f"train/log(std_{i})"] = val
        self.log_data["train/Policy Loss"] = policy_loss
        self.log_data["train/Value Loss"] = value_loss

        samples = eps * self.model.interactions_per_episode
        if samples - self.last_eval >= self.eval_freq or eps == num_learn_episodes - 1:
            self.last_eval = samples
            eval_reward = self.evaluate_policy(eps, num_learn_episodes)

            if self.optuna_trial is not None:
                self.optuna_trial.report(eval_reward, eps)

        self.wandb_run.log(data=self.log_data, step=samples, commit=True)
        self.log_data = {}

        if self.optuna_trial is not None and self.optuna_trial.should_prune():
            self.wandb_run.finish()
            raise TrialPruned()

    @jaxtyped(typechecker=beartype)
    def evaluate_policy(self, eps: int, num_learn_episodes: int) -> float:
        """
        Evaluates the current policy on the evaluation environment.

        Args:
            eps: The current episode number.
            num_learn_episodes: The total number of learning episodes.
        Returns:
            float: The average reward obtained during the evaluation.
        """
        eval_reward = 0
        record = not self.fast_eval or eps == num_learn_episodes - 1
        frames = []

        observation, info = self.eval_env.eval_reset()
        terminal = False
        steps = 0
        while not terminal:
            action = self.model.policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = self.eval_env.step(action)
            terminal = (terminated | truncated)[0].item()
            if record:
                frame = torchvision.utils.make_grid(torch.stack(self.eval_env.render()),
                                                    nrow=int(torch.sqrt(torch.tensor([self.eval_env.num_envs]))))
                frames += [frame]

            eval_reward += reward.sum().item()
            steps += 1

        avg_eval_reward = eval_reward  / self.eval_env.num_envs / steps

        self.log_data["eval/Average Reward"] = avg_eval_reward

        if record and frames[0].numel() != 0:
            frames = torch.stack(frames).numpy()
            self.log_data["eval/Video"] = wandb.Video(frames, fps=60, format="mp4")

        if avg_eval_reward > self.best_reward:
            self.best_reward = avg_eval_reward

        return avg_eval_reward
