from timeit import default_timer as timer

import torch

from tasks.balance_pendulum import BalancePendulumTask
from tasks.balance_quadrotor import BalanceQuadrotorTask
from tasks.wrapper.orthogonal_ray_map import OrthogonalRayMapWrapper
from tasks.wrapper.boundary_projection import BoundaryProjectionWrapper
from tasks.wrapper.zonotope_ray_map import ZonotopeRayMapWrapper
from src.sets.zonotope import Zonotope
from tabulate import tabulate

quad = BalanceQuadrotorTask(device="cuda", num_envs=16)
pen = BalancePendulumTask(device="cuda", num_envs=16)

times = {
    "Quadrotor": {
        "BoundaryProjection": {},
        "OrthogonalRayMap": {},
        "ZonotopeRayMap": {},
        "ZonotopeRayMapReuse": {}
    },
    "Pendulum": {
        "BoundaryProjection": {},
        "OrthogonalRayMap": {},
        "ZonotopeRayMap": {},
        "ZonotopeRayMapReuse": {}
    }
}


def run_test(env, size):
    time = 0
    for i in range(3):
        _ = env.reset()
        action = torch.zeros((16, size), device="cuda", dtype=torch.float64,
                             requires_grad=True)
        start = timer()
        for i in range(1000):
            _ = env.step(action)
        end = timer()
        time += end - start
    return time / 10


times["Quadrotor"]["NoProjection"] = run_test(quad, 2)
times["Pendulum"]["NoProjection"] = run_test(pen, 1)

for order in [5, 4, 3, 2]:
    pen.rci = Zonotope(
        torch.load(f"../src/assets/pendulum_rci_center.pt", weights_only=True).unsqueeze(0),
        torch.load(f"../src/assets/pendulum_rci_generators_{order}.pt",
                   weights_only=True).unsqueeze(0),
    )
    pen.rci.center.to("cuda")
    pen.rci.generator.to("cuda")

    lin = ([0.0, 0.0], [0.0], [0.0, 0.0])
    projection = BoundaryProjectionWrapper(pen, *lin)
    orp = OrthogonalRayMapWrapper(pen, *lin, exponential_projection=False)
    zrp = ZonotopeRayMapWrapper(pen, *lin, num_generators=4,
                                       reuse_safe_set=False, exponential_projection=False, passthrough=False)
    zrpr = ZonotopeRayMapWrapper(pen, *lin, num_generators=4,
                                        reuse_safe_set=True, exponential_BoundaryProjection=False, passthrough=False)

    num_gens = pen.rci.generator.shape[1]
    times["Pendulum"]["BoundaryProjection"][num_gens] = run_test(projection, 1)
    times["Pendulum"]["OrthogonalRayMap"][num_gens] = run_test(orp, 1)
    times["Pendulum"]["ZonotopeRayMap"][num_gens] = run_test(zrp, 1)
    times["Pendulum"]["ZonotopeRayMapReuse"][num_gens] = run_test(zrpr, 1)

for order in [5, 4]:
    quad.rci = Zonotope(
        torch.load(f"../src/assets/quadrotor_rci_center.pt", weights_only=True).unsqueeze(0),
        torch.load(f"../src/assets/quadrotor_rci_generators_{order}.pt",
                   weights_only=True).unsqueeze(0),
    )
    quad.rci.center.to("cuda")
    quad.rci.generator.to("cuda")
    lin = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0],
           [0.0, 0.0,0.0,0.0,0.0,0.0])
    projection = BoundaryProjectionWrapper(quad, *lin)
    orp = OrthogonalRayMapWrapper(quad, *lin, linear_projection=True)
    zrp = ZonotopeRayMapWrapper(quad, *lin, num_generators=12,
                                       reuse_safe_set=False, linear_projection=True, passthrough=False)
    zrpr = ZonotopeRayMapWrapper(quad, *lin, num_generators=12,
                                        reuse_safe_set=True, linear_projection=True, passthrough=False)

    num_gens = quad.rci.generator.shape[1]
    times["Quadrotor"]["BoundaryProjection"][num_gens] = run_test(projection, 2)
    times["Quadrotor"]["OrthogonalRayMap"][num_gens] = run_test(orp, 2)
    times["Quadrotor"]["ZonotopeRayMap"][num_gens] = run_test(zrp, 2)
    times["Quadrotor"]["ZonotopeRayMapReuse"][num_gens] = run_test(zrpr, 2)


rows = []
for system, projections in times.items():
    for projection, values in projections.items():
        if isinstance(values, dict):
            for order, time in values.items():
                rows.append([system, projection, order, time])
        else:
            rows.append([system, projection, "-", values])

headers = ["System", "Projection Type", "Order", "Time (s)"]
table = tabulate(rows, headers, tablefmt="pipe")

print(table)

with open("generator_scaling_results.txt", "w") as f:
    f.write(table)