import torch
import matplotlib.pyplot as plt

from algorithms.components.value_function import ValueFunction
from conf.algorithm.shac import SHACConfig


def test_value_function():
    config = SHACConfig()
    input_dim = 5
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.arange(0, batch_size * input_dim, device=device,
                     dtype=torch.float64).reshape(batch_size,
                                                  input_dim)
    y = (x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())
    y += (x[:, 1] - x[:, 1].min()) / (x[:, 1].max() - x[:, 1].min())
    y /= 2
    y = y.reshape(batch_size, 1)

    value_function = ValueFunction(input_dim, **config.vf_kwargs)
    optim = torch.optim.Adam(value_function.parameters(), **config.vf_optim_kwargs)

    loss_list = []
    for epoch in range(5000):
        optim.zero_grad()
        y_hat = value_function(x)
        loss = ((y_hat - y) ** 2).mean()
        loss.backward()
        #for name, param in policy.named_parameters():
            #print(f"{name}: {param.grad.mean()}")
        optim.step()
        loss_list.append(loss.item())


    print("Final loss:", loss.item())
    print("Target:", y[:5].squeeze().tolist())
    print("Prediction:", y_hat[:5].squeeze().tolist())
    plt.plot(loss_list)
    plt.show()

    assert loss.item() < 1e-2
