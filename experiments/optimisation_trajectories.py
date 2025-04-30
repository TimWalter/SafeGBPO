import cvxpy as cp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from scipy.spatial import ConvexHull

from tasks.balance_quadrotor import BalanceQuadrotorTask
import src.sets as sets
from src.utils import PassthroughCvxpyLayer

class PassthroughTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.tanh(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def plot_shape(ax, vertices, label, color):
    hull = ConvexHull(vertices.T)

    for i, simplex in enumerate(hull.simplices):
        ax.plot(vertices[0, simplex], vertices[1, simplex], lw=1,
                label=label if i == 0 else None, color=color)

    ax.scatter(vertices[0], vertices[1], color=color, s=5)


def plot_trajectory(ax, action, proj_action,
                    title1='Action',
                    title2='Projected Action',
                    color1="black",
                    color2="steelblue",
                    ):
    trajectory_x = [a[0].numpy() for a in action]
    trajectory_y = [a[1].numpy() for a in action]

    executed_trajectory_x = [a[0].numpy() for a in proj_action]
    executed_trajectory_y = [a[1].numpy() for a in proj_action]

    for j in range(len(trajectory_x)):
        ax.scatter(
            trajectory_x[j],
            trajectory_y[j],
            c=color1,
            marker=f"${j}$",
            s=200,
            label=title1 if j == 0 else None
        )

    ax.scatter(
        executed_trajectory_x,
        executed_trajectory_y,
        c=color2,
        marker=".",
        s=50,
        label=title2
    )

    for i in range(len(trajectory_x) - 1):
        ax.annotate(
            '',  # No text
            xy=(executed_trajectory_x[i], executed_trajectory_y[i]),
            xytext=(trajectory_x[i], trajectory_y[i]),
            arrowprops=dict(
                arrowstyle='->',
                color=color2,
                lw=0.4
            )
        )


def plot_reward(ax, goal, cm=plt.cm.Reds):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x = np.linspace(x_min, x_max, 2400)
    y = np.linspace(y_min, y_max, 2400)
    X, Y = np.meshgrid(x, y)

    grid_points = np.stack([X.flatten(), Y.flatten()], axis=-1)
    rewards = np.linalg.norm(grid_points - goal.numpy(), axis=1) ** 2
    rewards = rewards.reshape(X.shape)

    norm = mcolors.Normalize(vmin=rewards.min(), vmax=rewards.max())
    colors = cm((norm(rewards) * -1 + 1) ** 4)

    # Plot the background
    ax.imshow(colors, extent=[x_min, x_max, y_min, y_max], origin='lower')


def plot(title, z, goal, action, proj_action, feasible_vertices, losses, gradient_norms,
         reward_gradient_norms):

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, height_ratios= [7, 1])
    ax = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    # Adjust font sizes globally
    plt.rcParams.update({
        'font.size': 24,  # Increase font size
        'legend.fontsize': 22,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    })

    plot_shape(ax[0], feasible_vertices, "Feasible Set", color="silver")
    plot_trajectory(ax[0], action, proj_action)
    plot_shape(ax[0], z.vertices(), "Safe Set", color="black")
    plot_reward(ax[0], goal)

    ax[0].scatter(
        goal[0].numpy(),
        goal[1].numpy(),
        marker='1',
        color="red",
        s=300,
        label='Optimal Action'
    )

    ax[0].scatter(
        z.center[0][0].numpy(),
        z.center[0][1].numpy(),
        marker='1',
        color="blue",
        s=300,
        label='Center'
    )

    ax[0].legend(loc="lower right")
    ax[0].grid(False)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title(title, fontsize=28)
    ax[0].tick_params(axis='both', which='major', labelsize=22)
    ax[0].tick_params(axis='both', which='minor', labelsize=22)

    ax[1].plot(losses.numpy(), color="black")
    ax[1].set_title("Loss", fontsize=24)
    ax[1].set_xlabel("Iteration", fontsize=22)
    ax[1].set_ylabel("Loss", fontsize=22)
    ax[1].set_xticks(np.arange(0, len(losses), step=5))
    ax[1].grid(True)

    ax[2].plot(gradient_norms, color="black", label="Gradient Norm")
    ax[2].plot(reward_gradient_norms, color="blue", label="Reward Gradient Norm")
    ax[2].set_xlabel("Iteration", fontsize=22)
    ax[2].set_ylabel("Value", fontsize=22)
    ax[2].set_xticks(np.arange(0, len(gradient_norms), step=5))
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(title)
    plt.show()


def construct_projection_layer(z, passthrough):
    action = cp.Parameter(2)

    safe_action = cp.Variable(2)

    objective = cp.Minimize(cp.sum_squares(action - safe_action))

    constraints = sets.Zonotope.point_containment_constraints(safe_action,
                                                         z.center[0].numpy(),
                                                         z.generator[0].numpy())

    problem = cp.Problem(objective, constraints)

    if passthrough:
        return PassthroughCvxpyLayer(problem, parameters=[action], variables=[safe_action])
    else:
        return CvxpyLayer(problem, parameters=[action], variables=[safe_action])


def construct_safety_layer():
    action = cp.Parameter(2)

    weights = cp.Variable(z.generator.shape[2])

    constraints = [
        z.generator[0].numpy() @ weights == action - z.center[0].numpy(),
    ]

    objective = cp.Minimize(cp.norm(weights, "inf"))
    problem = cp.Problem(objective, constraints)

    if passthrough:
        return PassthroughCvxpyLayer(problem, parameters=[action], variables=[weights])
    else:
        return CvxpyLayer(problem, parameters=[action], variables=[weights])


def construct_safe_boundary_layer():
    action = cp.Parameter(2)
    direction = cp.Parameter(2)

    alpha = cp.Variable(1, nonneg=True)
    scaling = cp.Variable(z.generator.shape[2])

    safe_boundary = action + alpha * direction

    objective = cp.Maximize(alpha)

    constraints = [
        z.generator[0].numpy() @ scaling == safe_boundary - z.center[0].numpy(),
        cp.norm(scaling, "inf") <= 1
    ]

    problem = cp.Problem(objective, constraints)
    if passthrough:
        return PassthroughCvxpyLayer(problem, parameters=[action, direction],
                                     variables=[alpha, scaling])
    else:
        return CvxpyLayer(problem, parameters=[action, direction],
                      variables=[alpha, scaling])


def get_boundary(action, direction, low, high):
    shift_high = torch.full_like(action, torch.inf)
    shift_low = torch.full_like(action, torch.inf)
    mask = direction != 0
    shift_high[mask] = (torch.from_numpy(high)[mask] - action[mask]) / direction[mask]
    shift_low[mask] = (torch.from_numpy(low)[mask] - action[mask]) / direction[mask]

    shifts = torch.cat([shift_high, shift_low], dim=0)
    shift = torch.min(torch.where(shifts < 0, torch.inf, shifts))
    return action + shift * direction

def create_smooth_zonotope():
    env = BalanceQuadrotorTask(stochastic=False, rci_size=5)
    env.reset()
    env.goal_pos[0] = env.rci.center[:2]
    env.state[0] = torch.tensor([0.3, 0.9, -0.2, 0.4, 0.1, -0.5])

    num_generators = 100
    direction = np.random.rand(2, num_generators) * 2 - 1
    direction = direction / np.linalg.norm(direction, axis=0, keepdims=True)
    state = env.state[0].numpy()

    center = cp.Variable(2)
    length = cp.Variable(num_generators, nonneg=True)

    generator = direction @ cp.diag(length)

    lin_state = torch.tensor([0.0, 1.15, 0.0, 0.0, 0.0, 0.0])
    lin_action = torch.tensor([0.0, 0.0])
    lin_noise = torch.tensor([0.0, 0.0])

    constant_mat, state_mat, action_mat, noise_mat = env.linear_dynamics(lin_state,
                                                                         lin_action,
                                                                         lin_noise)

    constant_mat = constant_mat.numpy()
    state_mat = state_mat.numpy()
    action_mat = action_mat.numpy()
    noise_mat = noise_mat.numpy()

    lin_state = lin_state.numpy()
    lin_action = lin_action.numpy()
    lin_noise = lin_noise.numpy()

    noise_center = env.noise.center.numpy()
    noise_generator = env.noise.generator.numpy()

    next_state_center = constant_mat \
                        + state_mat @ (state - lin_state) \
                        + action_mat @ (center - lin_action) \
                        + noise_mat @ (noise_center - lin_noise)
    next_state_generator = cp.hstack([action_mat @ generator,
                                      noise_mat @ noise_generator])

    objective = cp.Maximize(cp.geo_mean(length))

    state_safety = sets.Zonotope.zonotope_containment_constraints(next_state_center,
                                                             next_state_generator,
                                                             env.rci.center.numpy(),
                                                             env.rci.generator.numpy())

    action_safety = sets.Zonotope.zonotope_containment_constraints(center,
                                                              generator,
                                                              env.ctrl.center.numpy(),
                                                              env.ctrl.generator.numpy())

    constraints = state_safety + action_safety

    problem = cp.Problem(objective, constraints)

    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise RuntimeError("Optimization failed.")

    z = sets.Zonotope(torch.from_numpy(center.value).type(torch.float32).unsqueeze(0),
                 torch.from_numpy(generator.value).type(torch.float32).unsqueeze(0) * 0.8)


    return z

def create_edgy_zonotope():
    z = sets.Zonotope(torch.zeros(1, 2, dtype=torch.float32),
                 torch.tensor([[0.1, 0.1],
                               [0.1, 0.3],
                               [0.3, 0.2],
                               [0.2, 0.1]],
                              dtype=torch.float32).T.unsqueeze(0))

    return z

def reward(action, goal):
    return -torch.linalg.vector_norm(action - goal) ** 2

if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    solver_args = {"solve_method": "Clarabel"}
    low = torch.tensor([[-1.0, -1.0]]).numpy()
    high = torch.tensor([[1.0, 1.0]]).numpy()
    feasible_vertices = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]]).T
    lr = 0.2

    for add_loss in [True]:
        for smooth in [False]:
            for passthrough in [False]:
                for safe_optimal in [False]:
                    for method in ["ZRPR-Exp"]:
                        #smooth = True
                        #passthrough = True
                        #safe_optimal = True
                        #method = "ORP"

                        policy = torch.tensor([-0.8, 1.0], requires_grad=True)

                        if safe_optimal:
                            goal = torch.tensor([0.25 ,0.25])
                        else:
                            goal = torch.tensor([0.9, 0.0])

                        if smooth:
                            z = create_smooth_zonotope()
                        else:
                            z = create_edgy_zonotope()

                        projection_layer = construct_projection_layer(z, passthrough)
                        safety_layer = construct_safety_layer()
                        safe_boundary_layer = construct_safe_boundary_layer()

                        opt = torch.optim.SGD([policy], lr=lr)
                        actions = torch.zeros((100, 2), dtype=torch.float32)
                        proj_actions = torch.zeros((100, 2), dtype=torch.float32)
                        losses = torch.zeros(100, dtype=torch.float32)
                        reward_gradient_norms = []
                        gradient_norms = []

                        for i in range(100):
                            if passthrough:
                                action = PassthroughTanh.apply(policy)
                            else:
                                action = torch.tanh(policy)
                            action.retain_grad()
                            actions[i, :] = action.detach().clone()
                            opt.zero_grad()
                            if method == "NP":
                                proj_action = action
                            elif method == "P":
                                proj_action = projection_layer(action, solver_args=solver_args)[0]
                                proj_action.retain_grad()
                            elif method == "ORP":
                                scaling = safety_layer(action, solver_args=solver_args)[0]
                                if torch.linalg.norm(scaling, float("inf")) > 1:
                                    safe_boundary = projection_layer(action, solver_args=solver_args)[0]
                                    sf2_move = \
                                    safe_boundary_layer(safe_boundary, safe_boundary - action,
                                                        solver_args=solver_args)[0]
                                    safe_boundary2 = safe_boundary + sf2_move * (safe_boundary - action)
                                    center = (safe_boundary + safe_boundary2) / 2
                                    boundary = get_boundary(action, action - safe_boundary, low, high)

                                    full_dist = torch.linalg.vector_norm(center - boundary)
                                    safe_dist = torch.linalg.vector_norm(center - safe_boundary)

                                    if passthrough:
                                        proj_action = center.detach() + safe_dist.detach() / full_dist.detach() * (action - center.detach())
                                    else:
                                        proj_action = center + safe_dist / full_dist * (action - center)
                                else:
                                    proj_action = action
                                proj_action.retain_grad()
                            elif "ZRPR" in method:
                                alpha = safe_boundary_layer(z.center, action - z.center, solver_args=solver_args)[0]

                                if not passthrough:
                                    safe_boundary = z.center + alpha * (action - z.center)
                                    boundary = get_boundary(safe_boundary, safe_boundary - z.center, low, high)
                                else:
                                    with torch.no_grad():
                                        safe_boundary = z.center + alpha * (action - z.center)
                                        boundary = get_boundary(safe_boundary, safe_boundary - z.center, low, high)

                                full_dist = torch.linalg.vector_norm(z.center - boundary)
                                safe_dist = torch.linalg.vector_norm(z.center - safe_boundary)
                                if not "Exp" in method:
                                    proj_action = z.center + safe_dist / full_dist * (action - z.center)
                                else:
                                    action_dist = (action - z.center).norm()
                                    ray = (action - z.center) / action_dist
                                    proj_action= z.center + ray * safe_dist * torch.tanh(action_dist/safe_dist) / torch.tanh(full_dist/safe_dist)
                            else:
                                raise NotImplementedError(f"Method {method} not implemented.")
                            proj_actions[i, :] = proj_action.detach().clone()
                            loss = -reward(proj_action, goal)
                            if add_loss:
                                loss = (-reward(proj_action, goal)) * (1+torch.linalg.norm(proj_action - action))

                            losses[i] = loss.detach().clone()

                            loss.backward()

                            g = ((proj_action - goal) * 2).detach()
                            reward_gradient_norms.append(g.norm())
                            if passthrough and "RP" in method and not (action == proj_action).all():
                                policy.grad = policy.grad * full_dist / safe_dist

                            gradient_norms.append(policy.grad.norm().detach())
                            opt.step()
                            for param_group in opt.param_groups:
                                param_group["lr"] *= 1.0
                        title = "l_" if add_loss else ""
                        title += "s_" if smooth else "e_"
                        title += "p_" if passthrough else "d_"
                        title += "s_" if safe_optimal else "u_"
                        title += method.replace(" ", "_") + ".png"
                        plot(title, z, goal, actions, proj_actions, feasible_vertices, losses,
                             gradient_norms, reward_gradient_norms)
