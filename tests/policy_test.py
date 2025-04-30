import torch
import matplotlib.pyplot as plt

from algorithms.components.policy import Policy
from conf.algorithm.shac import SHACConfig


def test_policy():
    config = SHACConfig()
    input_dim = 5
    output_dim = 1
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.arange(0, batch_size * input_dim, device=device,
                     dtype=torch.float64).reshape(batch_size,
                                                  input_dim)
    y = (x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())
    y += (x[:, 1] - x[:, 1].min()) / (x[:, 1].max() - x[:, 1].min())
    y /= 2
    y = y.reshape(batch_size, output_dim)

    policy = Policy(input_dim, output_dim, **config.policy_kwargs, stochastic=True)
    optim = torch.optim.Adam(policy.parameters(), **config.policy_optim_kwargs)

    loss_list = []
    log_std_list = []
    for epoch in range(5000):
        optim.zero_grad()
        y_hat = policy(x)
        loss = ((y_hat - y) ** 2).mean()
        loss.backward()
        #for name, param in policy.named_parameters():
            #print(f"{name}: {param.grad.mean()}")
        optim.step()
        loss_list.append(loss.item())
        log_std_list.append(policy.log_std.item())


    print("Final loss:", loss.item())
    print("LogStd:", policy.log_std.cpu())
    print("Target:", y[:5].squeeze().tolist())
    print("Prediction:", y_hat[:5].squeeze().tolist())

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(loss_list)
    ax[1].plot(log_std_list)
    plt.show()

    assert loss.item() < 1e-2
