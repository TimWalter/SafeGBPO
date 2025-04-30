import io

import numpy as np
import plotly.graph_objects as go
import torch
import wandb
from PIL import Image
from jaxtyping import Float
from optuna import Trial, TrialPruned
from plotly.subplots import make_subplots
from torch import Tensor
from wandb.sdk.wandb_run import Run

from src.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class TrainCallback:
    """
    Callback for logging evaluation and training data to Weights and Biases.

    2 Principles of Callback
    Informative & Non-Intrusive

    Attributes:
        eval_env: The environment to evaluate the policy on.
        env_name: The name of the environment.
        wrapper_names: The names of the wrappers to apply to the environment.
        wandb_run: The Weights and Biases run object.
        optuna_trial: The Optuna trial object.
        eval_freq: The frequency at which to evaluate the policy.
        best_reward: The best reward achieved during evaluation.
        log_data: The data to log.
    """

    def __init__(self,
                 model: ActorCriticAlgorithm,
                 algo_name: str,
                 eval_env: TorchVectorEnv,
                 env_name: str,
                 wrapper_names: list[str],
                 wandb_run: Run,
                 optuna_trial: Trial | None,
                 eval_freq: int,
                 log_eval: bool,
                 log_vf: bool,
                 fast_eval: bool):
        """
        Initializes the LogEvalCallback.
        Args:
            model (ActorCriticAlgorithm): The model to train.
            algo_name (str): The name of the algorithm.
            eval_env (TorchVectorEnv): The environment to evaluate the policy on (num_envs=1).
            env_name (str): The name of the environment.
            wrapper_names (list[str]): The names of the wrappers to apply to the environment.
            wandb_run (Run): The Weights and Biases run object.
            optuna_trial (Trial | None): The Optuna trial object.
            eval_freq (int): The frequency at which to evaluate the policy.
            log_eval (bool): Whether to augment the frame by the evaluation figures.
            log_vf (bool): Whether to log the value function.
            fast_eval (bool): Whether to produce video only on the final episode
        """
        self.model = model
        self.algo_name = algo_name
        self.eval_env = eval_env
        self.eval_env.render_mode = "rgb_array"
        self.env_name = env_name
        self.wrapper_names = wrapper_names
        self.action_dim = eval_env.action_space.shape[1]
        self.obs_dim = eval_env.observation_space.shape[1]
        self.wandb_run = wandb_run
        self.optuna_trial = optuna_trial
        self.eval_freq = eval_freq
        self.log_eval = log_eval
        self.log_vf = log_vf
        self.fast_eval = fast_eval

        if self.log_eval:
            self.x = []
            self.y = {label: [] for label in self.customize_labels()}
            self.fig = None

        self.best_reward = -torch.inf
        self.log_data = {}
        self.last_eval = 0

    def customize_labels(self) -> list[str]:
        augment_labels = [
            "Reward",
            *[f"Action{i}" for i in range(self.action_dim)]
        ]
        if self.wrapper_names:
            augment_labels += [
                *[f"Safe Action{i}" for i in range(self.action_dim)],
                "Intervention"
            ]

        if "Pendulum" in self.env_name:
            augment_labels += ["Angle", "Angular Velocity"]
        elif "CartPole" in self.env_name:
            augment_labels += ["Velocity", "Angle", "Angular Velocity"]
        elif "Quadrotor" in self.env_name:
            augment_labels += [
                "Horizontal Position",
                "Vertical Position",
                "Roll",
                "Horizontal Velocity",
                "Vertical Velocity",
                "Roll Velocity"
            ]
        if self.algo_name == "SHAC":
            augment_labels += ["Target Value"]
        elif self.algo_name == "SAC":
            augment_labels += ["Value1", "Value2"]
        elif self.algo_name == "PPO":
            augment_labels += ["Value"]

        return augment_labels

    def initialize_plot(self) -> go.Figure:
        """
        Initializes the plot that will augment the frame.
        """

        fig = make_subplots(rows=len(self.y), cols=1, shared_xaxes=True,
                            subplot_titles=list(self.y.keys()))

        for i, (key, value) in enumerate(self.y.items()):
            fig.add_trace(go.Scatter(x=self.x, y=value, mode='lines', name=key),
                          row=i + 1, col=1
                          )
        return fig

    def on_learning_episode(self, eps: int, policy_loss: float,
                            value_loss: float, num_learn_episodes: int):
        """
        Custom callback call to log and evaluate.

        Args:
            eps (int): Number of the current learning episode
            policy_loss (float): The policy loss
            value_loss (float): The value loss
            num_learn_episodes (int): The total number of learning episodes
        """
        self.log_avg_episode_reward()
        if self.wrapper_names:
            self.log_interventions()
        for i, val in enumerate(self.model.policy.log_std.detach().cpu().numpy()):
            self.log_data[f"train/log_std{i}"] = val
        self.log_data["train/policy_loss"] = policy_loss
        self.log_data["train/value_loss"] = value_loss

        samples = eps * self.model.samples_per_episode
        if samples - self.last_eval >= self.eval_freq or eps == num_learn_episodes-1:
            self.last_eval = samples
            eval_reward = self.evaluate_policy(eps, num_learn_episodes)
            if self.log_vf:
                self.evaluate_value_function()

            if self.optuna_trial is not None:
                self.optuna_trial.report(eval_reward, eps)

        self.wandb_run.log(self.log_data, step=samples, commit=True)
        self.log_data = {}

        if self.optuna_trial is not None and self.optuna_trial.should_prune():
            self.wandb_run.finish()
            raise TrialPruned()

    def log_avg_episode_reward(self):
        """
        Logs the reward accumulated during the current episode.
        """
        reward_sum = self.model.buffer.rewards.sum().item()
        num_rewards = self.model.buffer.t.sum().item()
        self.log_data["train/avg_episode_reward"] = reward_sum / num_rewards

    def log_interventions(self):
        """
        Logs the number of interventions during the current episode.
        """
        self.log_data["train/interventions"] = self.model.env.interventions

    def augment_frame(self, frame: np.ndarray,
                      observation: Float[Tensor, "1 {self.obs_dim}"],
                      reward: Float[Tensor, "1"] | None = None,
                      action: Float[Tensor, "1 {self.action_dim}"] | None = None) \
            -> np.ndarray:
        """
        Augments the frame with the values to display.

        Args:
            frame: The frame to augment.
            observation: Step observation.
            reward: Step reward.
            action: Step action.

        Returns:
            np.ndarray: The augmented frame.
        """
        if reward is None:
            self.fig.update_layout(height=100 * len(self.y), width=frame.shape[1],
                                   showlegend=False)
        self.unpack_data(observation, reward, action)
        for i, (key, value) in enumerate(self.y.items()):
            self.fig.update_traces(
                selector=dict(name=key),
                x=self.x,
                y=value
            )
        augment = np.asarray(Image.open(io.BytesIO(self.fig.to_image(format="png"))))
        return np.vstack((frame, augment[..., :3]))

    def unpack_data(self, observation: Float[Tensor, "1 {self.obs_dim}"],
                    reward: Float[Tensor, "1"] | None,
                    action: Float[Tensor, "1 {self.action_dim}"] | None):
        if reward is None:
            self.x = [0]
            self.y = {label: [] for label in self.customize_labels()}
            self.y["Reward"] += [torch.nan]
            for i in range(self.action_dim):
                self.y[f"Action{i}"] += [torch.nan]
            if self.wrapper_names:
                for i in range(self.action_dim):
                    self.y[f"Safe Action{i}"] += [torch.nan]
                self.y["Intervention"] += [torch.nan]
            if self.algo_name == "SAC":
                self.y["Value1"] += [torch.nan]
                self.y["Value2"] += [torch.nan]
        else:
            self.x += [self.x[-1] + 1]
            self.y["Reward"] += [reward.item()]
            for i in range(self.action_dim):
                self.y[f"Action{i}"] += [action[0, i].item()]
            if self.wrapper_names:
                for i in range(self.action_dim):
                    self.y[f"Safe Action{i}"] += [
                        self.eval_env.safe_actions[0, i].item()]
                self.y["Intervention"] += [torch.isclose(
                    self.eval_env.safe_actions[0, 0], action[0, 0]).item()]
            if self.algo_name == "SAC":
                with torch.no_grad():
                    self.y["Value1"] += [
                        self.model.value_function1(observation, action).item()]
                    self.y["Value2"] += [
                        self.model.value_function2(observation, action).item()]

        if "Pendulum" in self.env_name:
            self.y["Angle"] += [
                torch.arctan2(observation[0, 0], observation[0, 1]).item()]
            self.y["Angular Velocity"] += [observation[0, 2].item()]
        elif "CartPole" in self.env_name:
            self.y["Velocity"] += [observation[0, 1].item()]
            self.y["Angle"] += [
                torch.arctan2(observation[0, 2], observation[0, 3]).item()]
            self.y["Angular Velocity"] += [observation[0, 4].item()]
        elif "Quadrotor" in self.env_name:
            self.y["Horizontal Position"] += [observation[0, 0].item()]
            self.y["Vertical Position"] += [observation[0, 1].item()]
            self.y["Roll"] += [observation[0, 2].item()]
            self.y["Horizontal Velocity"] += [observation[0, 3].item()]
            self.y["Vertical Velocity"] += [observation[0, 4].item()]
            self.y["Roll Velocity"] += [observation[0, 5].item()]
        if self.algo_name == "SHAC":
            with torch.no_grad():
                self.y["Target Value"] += [
                    self.model.target_value_function(observation).item()]
        elif self.algo_name == "PPO":
            with torch.no_grad():
                self.y["Value"] += [self.model.value_function(observation).item()]

    def evaluate_policy(self, eps: int, num_learn_episodes: int) -> float:
        """
        Evaluates the current policy on the evaluation environment.
        """
        if self.log_eval:
            self.fig = self.initialize_plot()

        eval_reward = 0
        record = not self.fast_eval or eps == num_learn_episodes - 1
        frames = []

        for eps in range(self.eval_env.eval_eps):
            observation, info = self.eval_env.eval_reset(eps)
            if record:
                frame = self.eval_env.render()
                if self.log_eval:
                    frame = self.augment_frame(frame, observation)
                frames += [frame]

            terminal = False
            while not terminal:
                action = self.model.policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                terminal = (terminated | truncated).item()
                if record:
                    frame = self.eval_env.render()
                    if self.log_eval:
                        frame = self.augment_frame(frame, observation, reward, action)
                    frames += [frame]

                eval_reward += reward.item()

        self.log_data["eval/Episodic Reward"] = eval_reward

        if record:
            frames = np.array(frames).transpose([0, 3, 1, 2])
            self.log_data["eval/Video"] = wandb.Video(frames, fps=60, format="mp4")

        if eval_reward > self.best_reward:
            self.best_reward = eval_reward

        return eval_reward

    def evaluate_value_function(self):
        """
        Evaluates the value function and logs it.
        """
        if "CartPole" in self.env_name or "Pendulum" in self.env_name:
            theta = torch.linspace(-torch.pi, torch.pi, 360, device=self.model.device)
            thetadot = torch.linspace(-5, 5, 10, device=self.model.device)

            theta_grid, thetadot_grid = torch.meshgrid(theta, thetadot, indexing='ij')
            if "Pendulum" in self.env_name:
                obs = torch.stack([
                    torch.sin(theta_grid),
                    torch.cos(theta_grid),
                    thetadot_grid
                ], dim=-1).reshape(-1, 3)
            elif "CartPole" in self.env_name:
                obs = torch.stack([
                    torch.zeros_like(theta_grid),
                    torch.zeros_like(theta_grid),
                    torch.sin(theta_grid),
                    torch.cos(theta_grid),
                    thetadot_grid
                ], dim=-1).reshape(-1, 5)

            X = thetadot_grid.cpu()
            Y = theta_grid.cpu() / np.pi * 180
            X_title = "Angular Velocity"
            Y_title = "Angle (Degrees)"
        elif "Quadrotor" in self.env_name:
            x = torch.linspace(-1.7, 1.7, 60, device=self.model.device)
            y = torch.linspace(0.3, 2.0, 60, device=self.model.device)

            x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
            obs = torch.stack([
                x_grid,
                y_grid,
                torch.zeros_like(x_grid),
                torch.zeros_like(x_grid),
                torch.zeros_like(x_grid),
                torch.zeros_like(x_grid)]
                , dim=-1).reshape(-1, 6)
            X = x_grid.cpu()
            Y = y_grid.cpu()

            X_title = "Horizontal Position"
            Y_title = "Vertical Position"
        else:
            raise NotImplementedError(
                f"Value function evaluation not implemented for {self.eval_env.unwrapped}.")

        if self.algo_name == "SHAC":
            with torch.no_grad():
                Z = {
                    "Value Function": self.model.value_function(obs).view(-1).cpu(),
                    "Target Value Function": self.model.target_value_function(obs).view(
                        -1).cpu()
                }
        elif self.algo_name == "SAC":
            with torch.no_grad():
                actions = self.model.policy.predict(obs, deterministic=True)
                Z = {
                    "Value Function1": self.model.value_function1(obs, actions).view(
                        -1).cpu(),
                    "Value Function2": self.model.value_function2(obs, actions).view(
                        -1).cpu(),
                    "Target Value Function1": self.model.value_function1(obs,
                                                                         actions).view(
                        -1).cpu(),
                    "Target Value Function2": self.model.value_function2(obs,
                                                                         actions).view(
                        -1).cpu()
                }
        elif self.algo_name == "PPO":
            with torch.no_grad():
                Z = {
                    "Value Function": self.model.value_function(obs).view(-1).cpu(),
                }
        else:
            raise NotImplementedError(
                f"Value function evaluation not implemented for this {self.algo_name}.")

        z_min = min([value.min().item() for value in Z.values()])
        z_max = max([value.max().item() for value in Z.values()])

        fig = go.Figure()

        buttons = [
            {
                "label": "All",
                "method": "update",
                "args": [{"visible": [True for _ in Z.keys()]}]
            }
        ]
        for key, value in Z.items():
            value_norm = (value - z_min) / (z_max - z_min)
            fig.add_trace(
                go.Surface(x=X, y=Y, z=value_norm.reshape(X.shape),
                           colorscale='Viridis', name=key,
                           opacity=0.7,
                           showscale=True if key == "Value Function" else False)
            )
            buttons += [
                {
                    "label": key,
                    "method": "update",
                    "args": [{"visible": [key == k for k in Z.keys()]}]
                }
            ]

        fig.update_layout(
            scene=dict(
                xaxis_title=X_title,
                yaxis_title=Y_title,
                zaxis_title='Value',
                aspectratio=dict(x=2, y=2, z=1)
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.1, xanchor="left",
                    y=1.15, yanchor="top"
                )
            ]
        )

        self.log_data["eval/Value Functions"] = wandb.Plotly(fig)
