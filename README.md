# Leveraging Analytic Gradients in Provably Safe Reinforcement Learning

---
This repository contains the implementation for paper [Leveraging Analytic Gradients in Provably Safe Reinforcement Learning](https://arxiv.org/abs/2506.01665)

In this paper, we develop the first effective safeguard for analytic gradient-based reinforcement learning, 
addressing the lack of safety guarantees in this high-performance paradigm. By adapting existing differentiable
safeguards and integrating them with a state-of-the-art algorithm and differentiable simulation, we enable provably 
safe training. Numerical experiments on three control tasks show that safeguarded training 
can be achieved without compromising performance.

## Installation

1. Clone the repository  
```bash
git clone git@github.com:TimWalter/SafeGBPO.git
```
2. In the project folder create and sync a virtual environment using uv 
```bash
uv venv
uv sync
```
3. Add src to your PYTHONPATH 
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Unit Tests

To check your installation activate the virtual environment and run the unit tests
```bash
source .venv/bin/activate
pytest tests/
```

## Training

---
To train a policy define an experiment in [main.py](https://github.com/TimWalter/SafeGBPO/blob/c8a806a7626c7b36bffd1e44249e55c9cd1fbbdc/src/main.py#L65) and run
```bash
python main.py
```
to properly see results ensure you are logged into weights and biases.

## Citation

---

If you consider our paper or code useful, please consider citing:

```kvk
@article{11154003,
  author={Walter, Tim and Markgraf, Hannah and Külz, Jonathan and Althoff, Matthias},
  journal={IEEE Open Journal of Control Systems}, 
  title={Leveraging Analytic Gradients in Provably Safe Reinforcement Learning}, 
  year={2025},
  volume={4},
  number={},
  pages={463-481},
  doi={10.1109/OJCSYS.2025.3607845}
}
```
