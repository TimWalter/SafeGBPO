from abc import ABC, abstractmethod
from typing import Callable

import cvxpy as cp
import numpy as np
import torch
from gymnasium.vector import VectorActionWrapper
from jaxtyping import Float
from torch import Tensor

from tasks.envs.interfaces.torch_vector_env import TorchVectorEnv


class SafetyWrapper(VectorActionWrapper, ABC):
    """
    Ensuring safety of actions according to some RCI set.
    """
    env: TorchVectorEnv

    def __init__(self,
                 env: TorchVectorEnv,
                 lin_state: list[float],
                 lin_action: list[float],
                 lin_noise: list[float]):
        """
        Args:
            env(SecuredEnv): A custom secured, pytorch-based environment.
            lin_state(list[float]): State point to linearise around.
            lin_action(list[float]): Action point to linearise around.
            lin_noise(list[float]): Noise point to linearise around.
        """
        super().__init__(env)

        self.state_dim = self.env.state.shape[1]
        self.action_dim = self.env.action_space.shape[1]

        self.lin_state = torch.tensor(lin_state, device=self.env.device, dtype=torch.float64)
        self.lin_action = torch.tensor(lin_action, device=self.env.device, dtype=torch.float64)
        self.lin_noise = torch.tensor(lin_noise, device=self.env.device, dtype=torch.float64)
        self.constant_mat, self.state_mat, self.action_mat, self.noise_mat = (
            self.env.linear_dynamics(self.lin_state, self.lin_action, self.lin_noise))

        self.safety_layer = self.construct_safety_func()
        self.safe_actions = None
        self.interventions = 0

        self.lower_clip = torch.from_numpy(self.env.single_action_space.low).to(
            self.env.device)
        self.upper_clip = torch.from_numpy(self.env.single_action_space.high).to(
            self.env.device)

    @abstractmethod
    def construct_safety_func(self) -> Callable[
        [
            Float[Tensor, "num_envs action_dim"]
        ],
        Float[Tensor, "num_envs action_dim"]]:
        """
        Construct the function which ensures safety.

        Returns:
            Callable[
            [
                Float[Tensor, "num_envs action_dim"]
            ],
            Float[Tensor, "num_envs action_dim"]]: The safety function.
        """
        pass

    def actions(
            self, actions: Float[Tensor, "num_envs action_dim"]
    ) -> Float[Tensor, "num_envs action_dim"]:
        safe_actions = self.safety_layer(actions)

        safe_actions = torch.clamp(safe_actions, self.lower_clip, self.upper_clip)

        self.safe_actions = safe_actions

        self.interventions += ((~torch.isclose(safe_actions, actions)).count_nonzero(
            dim=1) == self.action_dim).sum().item()
        return safe_actions

    def linear_step(self,
                    action: cp.Expression | np.ndarray,
                    state: cp.Expression | np.ndarray) \
            -> tuple[cp.Expression | np.ndarray, np.ndarray]:
        """
        Propagate the system through its linearised dynamics.

        Args:
            action (cp.Expression | np.ndarray): The action to take.
            state (cp.Expression | np.ndarray): The current state.

        Returns:
            tuple[cp.Expression | np.ndarray, np.ndarray]: The next state center and generator.
        """
        constant_mat = self.constant_mat.cpu().numpy()
        state_mat = self.state_mat.cpu().numpy()
        action_mat = self.action_mat.cpu().numpy()
        noise_mat = self.noise_mat.cpu().numpy()

        lin_state = self.lin_state.cpu().numpy()
        lin_action = self.lin_action.cpu().numpy()
        lin_noise = self.lin_noise.cpu().numpy()

        noise_center = self.env.noise.center[0].cpu().numpy()
        noise_generator = self.env.noise.generator[0].cpu().numpy()

        next_state_center = constant_mat \
                            + state_mat @ (state - lin_state) \
                            + action_mat @ (action - lin_action) \
                            + noise_mat @ (noise_center - lin_noise)
        next_state_generator = noise_mat @ noise_generator

        return next_state_center, next_state_generator

    def clear_computation_graph(self):
        self.env.clear_computation_graph()

    @property
    def steps(self):
        return self.env.steps

    @property
    def eval_eps(self):
        return self.env.eval_eps

    def eval_reset(self, eps: int):
        return self.env.eval_reset(eps)

    @property
    def render_mode(self) -> str:
        """Returns the `render_mode` from the base environment."""
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, render_mode: str):
        self.env.render_mode = render_mode
