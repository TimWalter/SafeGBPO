from timeit import default_timer as timer

import torch
from tabulate import tabulate

from tasks.balance_pendulum import BalancePendulumTask
from tasks.balance_quadrotor import BalanceQuadrotorTask
from tasks.wrapper.zonotope_ray_map import ZonotopeRayMapWrapper

quad = BalanceQuadrotorTask(device="cuda", num_envs=16)
pen = BalancePendulumTask(device="cuda", num_envs=16)

times = {
    "Quadrotor": {
        "ZonotopeRayMap": {},
        "ZonotopeRayMapReuse": {}
    },
    "Pendulum": {
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
        for i in range(100):
            _ = env.step(action)
        end = timer()
        time += end - start
    return time / 10

for reuse in [True, False]:
    for env in [quad, pen]:
        for num_generators in [2, 4, 6, 8, 10]:
            if env == quad:
                lin = ([0.0, 1.15, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                name = "Quadrotor"
                size = 2
            else:
                lin = ([0.0, 0.0], [0.0], [0.0, 0.0])
                name = "Pendulum"
                size = 1

            projection = ZonotopeRayMapWrapper(env, *lin,
                                                      num_generators=num_generators,
                                                      reuse_safe_set=reuse,
                                                      linear_projection=True,
                                                      passthrough=False
                                                      )
            algo = "ZonotopeRayMap" + ("Reuse" if reuse else "")
            times[name][algo][num_generators] = run_test(projection, size)

rows = []
for system, projections in times.items():
    for projection, values in projections.items():
        for num_generators, time in values.items():
            rows.append([system, projection, num_generators, time])

headers = ["System", "Projection Type", "Num Generators", "Time (s)"]
table = tabulate(rows, headers, tablefmt="pipe")

print(table)

# Write the table to a file
with open("zonotopic_approximation_scaling_results.txt", "w") as f:
    f.write(table)