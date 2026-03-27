from typing import Type
from pathlib import Path
from itertools import combinations_with_replacement

import torch
import pytest
import matplotlib.pyplot as plt

import sets

torch.manual_seed(0)

COMPACT_SETS = [
    sets.Ball,
    sets.Capsule,
    sets.AxisAlignedBox,
    sets.Box,
    sets.Zonotope,
    sets.Polytope
]


def save_plot(fig, test_name, node_name, file_name):
    node_name = node_name[len(test_name):]
    directory = Path(f"plots/{test_name}/{node_name}")
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{file_name}.png"
    fig.savefig(directory / filename)
    plt.close(fig)


@pytest.mark.parametrize("set_type", COMPACT_SETS)
def test_draw_bounds_sample(set_type: Type[sets.CompactSet], request):
    batch_dim = 10
    for i, test_set in enumerate(set_type.random(batch_dim, 2)):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        test_set.draw(ax, color="black", label="Test Set")
        lower, upper = test_set.bounds()
        sets.AxisAlignedBox(center=(lower + upper) / 2, generator=torch.diag_embed(upper - lower) / 2).draw(ax,
                                                                                                            color="red",
                                                                                                            label="BoundingBox")
        samples = test_set.sample(10000)[:, 0, :]
        ax.scatter(samples[:, 0], samples[:, 1], s=1, color="green", label="Samples")
        ax.legend()
        save_plot(fig, request.function.__name__, request.node.name, i)


@pytest.mark.parametrize("set_type", COMPACT_SETS)
def test_contain_points(set_type: Type[sets.CompactSet], request):
    batch_dim = 10
    for i, test_set in enumerate(set_type.random(batch_dim, 2)):
        assert test_set.contains(test_set.sample(100)).all()

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        test_set.draw(ax, color="black", label="Test Set")
        lower, upper = test_set.bounds()
        bounding_box = sets.AxisAlignedBox(center=(lower + upper) / 2,
                                           generator=torch.diag_embed(upper - lower) / 2)
        samples = bounding_box.sample(1000)
        contains = test_set.contains(samples)[:, 0]
        ax.scatter(samples[contains, 0, 0], samples[contains, 0, 1], s=1, color="green", label="Contained Samples")
        ax.scatter(samples[~contains, 0, 0], samples[~contains, 0, 1], s=1, color="red",
                   label="Not Contained Samples")
        ax.legend()
        save_plot(fig, request.function.__name__, request.node.name, i)


@pytest.mark.parametrize("container_type", COMPACT_SETS)
@pytest.mark.parametrize("probe_type", COMPACT_SETS)
def test_contain_compact_set(container_type: Type[sets.CompactSet], probe_type: Type[sets.CompactSet], request):
    for container_set in container_type.random(1, 2):
        for i, probe_set in enumerate(probe_type.random(25, 2)):
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            container_set.draw(ax, color="black", label="Container Set")

            contained = container_set.contains(probe_set)
            color = "green" if contained else "red"
            probe_set.draw(ax, color=color, label="Probe Set")
            ax.legend()
            save_plot(fig, request.function.__name__, request.node.name, i)


@pytest.mark.parametrize("set_a_type, set_b_type", list(combinations_with_replacement(COMPACT_SETS, 2)))
def test_intersect_compact_set(set_a_type, set_b_type, request):
    batch_dim = 10
    for i, set_a in enumerate(set_a_type.random(batch_dim, 2)):
        for j, set_b in enumerate(set_b_type.random(batch_dim, 2)):
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            intersect = set_a.intersects(set_b)
            color = "green" if not intersect else "red"
            set_a.draw(ax, color=color, label="Set A")
            set_b.draw(ax, color=color, label="Set B")
            ax.legend()
            save_plot(fig, request.function.__name__, request.node.name, i * batch_dim + j)
