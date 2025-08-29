import matplotlib.pyplot as plt
import torch

import src.sets as sets


def basic_test(cvx_set: sets.ConvexSet):
    """
    Basic tests for convex sets, including sampling, point containment and drawing.
    Args:
        cvx_set: The convex set to test.
    """
    contained_samples = torch.zeros(1000, 2)
    uncontained_samples = torch.zeros(1000, 2)
    for i in range(1000):
        contained_samples[i] = cvx_set.sample()
        uncontained_samples[i] = torch.randn(2) * 2 - 1 + contained_samples[i]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    cvx_set.draw(ax, color="black")
    ax.scatter(contained_samples[:, 0], contained_samples[:, 1], s=5, color="green")

    bad_contained = torch.tensor(
        [cvx_set.contains(sample.unsqueeze(0)) for sample in uncontained_samples])
    ax.scatter(uncontained_samples[~bad_contained][:, 0],
               uncontained_samples[~bad_contained][:, 1],
               s=5, color="red")

    plt.show()

    contained = torch.tensor(
        [cvx_set.contains(sample.unsqueeze(0)) for sample in contained_samples])
    assert contained.all()


def containment_test(cvx_set: sets.Ball | sets.Box | sets.Capsule | sets.Zonotope,
                     contained_samples: list[sets.ConvexSet],
                     uncontained_samples: list[sets.ConvexSet]):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    cvx_set.draw(ax, color="black")
    try:
        for sample in contained_samples:
            sample.draw(ax, color="green")
            assert cvx_set.contains(sample).all()

        for sample in uncontained_samples:
            sample.draw(ax, color="red")
            assert not cvx_set.contains(sample).all()
    except AssertionError:
        plt.show()
        assert False
    plt.show()


def intersection_test(cvx_set: sets.Ball | sets.Box | sets.Capsule | sets.Zonotope,
                      non_intersecting_samples: list[sets.ConvexSet],
                      intersecting_samples: list[sets.ConvexSet]):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    cvx_set.draw(ax, color="black")
    try:
        for sample in non_intersecting_samples:
            sample.draw(ax, color="green")
            assert not cvx_set.intersects(sample).all()
        for sample in intersecting_samples:
            sample.draw(ax, color="red")
            assert cvx_set.intersects(sample).all()
    except AssertionError:
        plt.show()
        assert False
    plt.show()


def test_basic_ball():
    ball = sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([3.0]))
    basic_test(ball)


def test_basic_capsule():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([5.0]))
    basic_test(capsule)


def test_basic_box():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))
    basic_test(box)


def test_basic_zonotope():
    zonotope = sets.Zonotope(torch.tensor([[1.0, 2.0]]), torch.tensor([[
        [1.0, 1.3, 2.4, -1.0],
        [1.0, -1.0, 0.0, 1.5]
    ]]))
    basic_test(zonotope)


def test_ball_ball_containment():
    ball = sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([3.0]))

    contained_samples = [
        sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[1.5, 2.5]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[-1.0, 2.5]]), torch.tensor([0.1])),
        sets.Ball(torch.tensor([[1.5, -0.5]]), torch.tensor([0.4])),
    ]

    uncontained_samples = [
        sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([3.5])),
        sets.Ball(torch.tensor([[3.0, 5.0]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[-4.0, -5.0]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[1.5, -0.5]]), torch.tensor([0.5])),
    ]

    containment_test(ball, contained_samples, uncontained_samples)


def test_ball_capsule_containment():
    ball = sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([3.0]))

    contained_samples = [
        sets.Capsule(torch.tensor([[0.0, 2.0]]), torch.tensor([[2.0, 2.0]]),
                     torch.tensor([2.0])),
        sets.Capsule(torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 3.0]]),
                     torch.tensor([1.0])),
        sets.Capsule(torch.tensor([[-1.0, 2.0]]), torch.tensor([[3.0, 2.0]]),
                     torch.tensor([1.0])),
        sets.Capsule(torch.tensor([[0.0, 3.0]]), torch.tensor([[0.0, 1.0]]),
                     torch.tensor([0.5])),
    ]

    uncontained_samples = [
        sets.Capsule(torch.tensor([[-1.0, 2.0]]), torch.tensor([[4.0, 2.0]]),
                     torch.tensor([3.0])),
        sets.Capsule(torch.tensor([[-1.0, 0.0]]), torch.tensor([[3.0, 4.0]]),
                     torch.tensor([3.0])),
        sets.Capsule(torch.tensor([[1.0, 4.0]]), torch.tensor([[-1.0, -1.0]]),
                     torch.tensor([0.1])),
        sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[4.1, 2.0]]),
                     torch.tensor([0.2])),
    ]

    containment_test(ball, contained_samples, uncontained_samples)


def test_ball_box_containment():
    ball = sets.Ball(torch.tensor([[-1.0, -2.0]]), torch.tensor([3.0]))

    contained_samples = [
        sets.Box(torch.tensor([[-1.0, -2.0]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
        sets.Box(torch.tensor([[-1.0, -2.0]]), torch.tensor([[[0.5, 0.5], [-0.5, 0.5]]])),
        sets.Box(torch.tensor([[-1.0, -2.0]]), torch.tensor([[[0.5, -0.5], [0.5, 0.5]]])),
        sets.Box(torch.tensor([[-1.0, -2.0]]), torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])),
    ]

    uncontained_samples = [
        sets.Box(torch.tensor([[0.0, 0.0]]), torch.tensor([[[0.5, 1.0],
                                                            [0.5, -1.0]]])),
        sets.Box(torch.tensor([[0.0, 0.0]]), torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])),
        sets.Box(torch.tensor([[-4.0, -5.0]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
        sets.Box(torch.tensor([[2.0, 2.0]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),

    ]
    containment_test(ball, contained_samples, uncontained_samples)


def test_ball_zonotope_containment():
    ball = sets.Ball(torch.tensor([[-1.0, -2.0]]), torch.tensor([3.0]))

    contained_samples = [
        sets.Zonotope(torch.tensor([[-1.0, -2.0]]),
                      torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
        sets.Zonotope(torch.tensor([[-1.0, -2.0]]),
                      torch.tensor([[[0.5, 0.5, 0.1], [0.5, 0.5, 0.1]]])),
        sets.Zonotope(torch.tensor([[-1.0, -2.0]]),
                      torch.tensor([[[0.5, -0.5, 0.2], [-0.5, 0.5, 0.2]]])),

    ]

    uncontained_samples = [
        sets.Zonotope(torch.tensor([[-1.0, -2.0]]),
                      torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5]]])),
        sets.Zonotope(torch.tensor([[0.0, 0.0]]),
                      torch.tensor([[[0.5, 1.0, 0.3], [0.5, -1.0, 0.3]]])),
        sets.Zonotope(torch.tensor([[0.0, 0.0]]),
                      torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5]]])),
        sets.Zonotope(torch.tensor([[-4.0, -5.0]]),
                      torch.tensor([[[1.0, 0.0, 0.2], [0.0, 1.0, 0.2]]])),
        sets.Zonotope(torch.tensor([[2.0, 2.0]]),
                      torch.tensor([[[1.0, 0.0, 0.3], [0.0, 1.0, 0.3]]])),
    ]

    containment_test(ball, contained_samples, uncontained_samples)


def test_capsule_ball_containment():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([5.0]))

    contained_samples = [
        sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([3.0])),
        sets.Ball(torch.tensor([[1.5, 2.5]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[-1.0, 2.5]]), torch.tensor([0.1])),
        sets.Ball(torch.tensor([[1.5, -0.5]]), torch.tensor([0.4])),
    ]

    uncontained_samples = [
        sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([5.5])),
        sets.Ball(torch.tensor([[4.0, 8.0]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[-4.0, -5.0]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[-1.5, -2.0]]), torch.tensor([0.5])),
    ]

    containment_test(capsule, contained_samples, uncontained_samples)


def test_capsule_capsule_containment():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([5.0]))

    contained_samples = [
        sets.Capsule(torch.tensor([[0.0, 2.0]]), torch.tensor([[2.0, 2.0]]),
                     torch.tensor([2.0])),
        sets.Capsule(torch.tensor([[-2.0, 3.0]]), torch.tensor([[6.0, 5.0]]),
                     torch.tensor([1.0]))
    ]

    uncontained_samples = [
        sets.Capsule(torch.tensor([[-3.0, -1.0]]), torch.tensor([[2.0, 2.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[-3.0, 5.0]]), torch.tensor([[0.0, 0.0]]),
                     torch.tensor([0.8]))
    ]

    containment_test(capsule, contained_samples, uncontained_samples)


def test_capsule_box_containment():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([5.0]))

    contained_samples = [
        sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
        sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[1.5, 1.5], [-1.0, 1.0]]])),
        sets.Box(torch.tensor([[2.0, 6.0]]), torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])),
    ]

    uncontained_samples = [
        sets.Box(torch.tensor([[2.0, 6.0]]), torch.tensor([[[2.0, 0.0], [0.0, 2.1]]])),
        sets.Box(torch.tensor([[6.0, 4.0]]), torch.tensor([[[1.5, 1.5], [-1.0, 1.0]]])),
    ]

    containment_test(capsule, contained_samples, uncontained_samples)


def test_capsule_zonotope_containment():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([2.5]))

    contained_samples = [
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      torch.tensor([[[0.5, 0.5, 0.1], [0.5, 0.5, 0.1]]])),
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      torch.tensor([[[0.5, -0.5, 0.2], [-0.5, 0.5, 0.2]]])),
    ]

    uncontained_samples = [
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5]]])),
        sets.Zonotope(torch.tensor([[0.0, 0.0]]),
                      torch.tensor([[[0.5, 1.0, 0.3], [0.5, -1.0, 0.3]]])),
        sets.Zonotope(torch.tensor([[0.0, 0.0]]),
                      torch.tensor([[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5]]])),
        sets.Zonotope(torch.tensor([[-4.0, -5.0]]),
                      torch.tensor([[[1.0, 0.0, 0.2], [0.0, 1.0, 0.2]]])),
        sets.Zonotope(torch.tensor([[2.0, 2.0]]),
                      torch.tensor([[[1.0, 0.0, 0.3], [0.0, 1.0, 0.3]]])),
    ]

    containment_test(capsule, contained_samples, uncontained_samples)


def test_box_ball_containment():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))

    contained_samples = [
        sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([1.0])),
        sets.Ball(torch.tensor([[3.0, 2.5]]), torch.tensor([0.6])),
        sets.Ball(torch.tensor([[-1.5, 0.5]]), torch.tensor([0.2])),
    ]

    uncontained_samples = [
        sets.Ball(torch.tensor([[-1.5, 0.5]]), torch.tensor([0.4])),
        sets.Ball(torch.tensor([[1.0, 1.0]]), torch.tensor([0.4])),
        sets.Ball(torch.tensor([[1.0, 3.0]]), torch.tensor([0.4])),
        sets.Ball(torch.tensor([[3.8, 3.0]]), torch.tensor([0.5])),
        sets.Ball(torch.tensor([[6.0, 6.0]]), torch.tensor([1.0])),
    ]

    containment_test(box, contained_samples, uncontained_samples)


def test_box_capsule_containment():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))

    contained_samples = [
        sets.Capsule(torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 2.0]]),
                     torch.tensor([0.3])),
        sets.Capsule(torch.tensor([[-1.0, 0.5]]), torch.tensor([[0.0, 2.0]]),
                     torch.tensor([0.2])),
        sets.Capsule(torch.tensor([[-1.0, 1.0]]), torch.tensor([[3.0, 2.3]]),
                     torch.tensor([0.7])),
    ]

    uncontained_samples = [
        sets.Capsule(torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 2.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[-1.0, 0.5]]), torch.tensor([[0.0, 2.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[-1.0, 1.0]]), torch.tensor([[3.0, 2.3]]),
                     torch.tensor([1.5])),
    ]

    containment_test(box, contained_samples, uncontained_samples)


def test_box_box_containment():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))

    contained_samples = [
        sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[0.8, 0.0],
                                                            [0.0, 0.8]]])),
        sets.Box(torch.tensor([[3.2, 2.2]]), torch.tensor([[[0.5, 0.1],
                                                            [0.5, -0.1]]])),
        sets.Box(torch.tensor([[-1.0, 1.0]]), torch.tensor([[[-0.2, 0.06],
                                                             [0.6, 0.02]]])),
    ]

    uncontained_samples = [
        sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[0.9, 0.0], [0.0, 0.9]]])),
        sets.Box(torch.tensor([[3.0, 2.0]]), torch.tensor([[[0.5, 0.1],
                                                            [0.5, -0.1]]])),
        sets.Box(torch.tensor([[-1.0, 1.0]]), torch.tensor([[[-0.3, 0.06],
                                                             [0.9, 0.02]]])),
        sets.Box(torch.tensor([[-1.2, 1.0]]), torch.tensor([[[-0.2, 0.9],
                                                             [0.6, 0.3]]])),
        sets.Box(torch.tensor([[-1.0, 5.0]]), torch.tensor([[[0.0, 1.0],
                                                             [1.0, 0.0]]])),
    ]

    containment_test(box, contained_samples, uncontained_samples)


def test_box_zonotope_containment():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))

    contained_samples = [
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      0.9 * torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
    ]

    uncontained_samples = [
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
    ]

    containment_test(box, contained_samples, uncontained_samples)


def test_zonotope_zonotope_containment():
    zonotope = sets.Zonotope(torch.tensor([[1.0, 2.0]]), torch.tensor([[
        [1.0, 1.3, 2.4, -1.0],
        [1.0, -1.0, 0.0, 1.5]
    ]]))

    contained_samples = [
        sets.Zonotope(torch.tensor([[1.0, 2.0]]),
                      0.9 * torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
        sets.Zonotope(torch.tensor([[3.0, 1.0]]),
                      0.9 * torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),

    ]

    uncontained_samples = [
        sets.Zonotope(torch.tensor([[-3.0, 3.0]]),
                      torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
        sets.Zonotope(torch.tensor([[4.5, 0.5]]),
                      0.9 * torch.tensor([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]]])),
    ]

    containment_test(zonotope, contained_samples, uncontained_samples)


def test_ball_ball_intersection():
    ball = sets.Ball(torch.tensor([[-1.0, -2.0]]), torch.tensor([3.0]))

    non_intersecting_samples = [
        sets.Ball(torch.tensor([[1.0, 2.0]]), torch.tensor([1.0])),
    ]

    intersecting_samples = [
        sets.Ball(torch.tensor([[-1.0, 2.0]]), torch.tensor([1.0])),
    ]

    intersection_test(ball, non_intersecting_samples, intersecting_samples)


def test_ball_capsule_intersection():
    ball = sets.Ball(torch.tensor([[-1.0, -2.0]]), torch.tensor([3.0]))

    non_intersecting_samples = [
        sets.Capsule(torch.tensor([[3.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                     torch.tensor([1.0])),
        sets.Capsule(torch.tensor([[-4.0, -8.0]]), torch.tensor([[10.0, -4.0]]),
                     torch.tensor([1.0])),
    ]

    intersecting_samples = [
        sets.Capsule(torch.tensor([[1.0, 1.0]]), torch.tensor([[3.0, 4.0]]),
                     torch.tensor([1.0])),
        sets.Capsule(torch.tensor([[-10.0, -10.0]]), torch.tensor([[-3.0, -4.0]]),
                     torch.tensor([1.0])),
        sets.Capsule(torch.tensor([[-8.0, -4.0]]), torch.tensor([[3.0, 4.0]]),
                     torch.tensor([1.0])),
    ]

    intersection_test(ball, non_intersecting_samples, intersecting_samples)


def test_ball_box_intersection():
    ball = sets.Ball(torch.tensor([[-1.0, -2.0]]), torch.tensor([3.0]))

    non_intersecting_samples = [
        sets.Box(torch.tensor([[4.0, 2.0]]), torch.tensor([[[1.0, 0.0],
                                                            [0.0, 1.0]]])),
    ]

    intersecting_samples = [
        sets.Box(torch.tensor([[2.5, -0.5]]), torch.tensor([[[1.0, 0.0],
                                                             [0.0, 1.0]]])),
        sets.Box(torch.tensor([[-4.0, -4.0]]), torch.tensor([[[1.0, 0.0],
                                                              [0.0, 1.0]]])),
        sets.Box(torch.tensor([[-1.0, 2.0]]), torch.tensor([[[1.0, 1.0],
                                                             [1.0, -1.0]]])),
    ]

    intersection_test(ball, non_intersecting_samples, intersecting_samples)


def test_ball_zonotope_intersection():
    ball = sets.Ball(torch.tensor([[-1.0, -2.0]]), torch.tensor([3.0]))

    non_intersecting_samples = [
        sets.Zonotope(torch.tensor([[-3.0, 3.0]]), torch.tensor([[[1.0, 0.5, 0.2],
                                                                  [0.5, 1.0, 0.3]]])),
    ]

    intersecting_samples = [
        sets.Zonotope(torch.tensor([[-1.0, 1.0]]), torch.tensor([[[1.0, 0.5, 0.2],
                                                                  [0.5, 1.0, 0.3]]])),
        # Problem of both overapproximations
        sets.Zonotope(torch.tensor([[-1.5, 2.2]]), torch.tensor([[[1.0, 0.5, 0.2],
                                                                  [0.5, 1.0, 0.3]]])),
    ]

    intersection_test(ball, non_intersecting_samples, intersecting_samples)


def test_capsule_capsule_intersection():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([1.0]))

    non_intersecting_samples = [
        sets.Capsule(torch.tensor([[4.0, 1.0]]), torch.tensor([[7.0, 4.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[0.0, -3.0]]), torch.tensor([[8.0, 5.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[0.0, -3.0]]), torch.tensor([[3.0, 0.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[4.0, 6.0]]), torch.tensor([[2.0, 6.0]]),
                     torch.tensor([0.5])),
    ]

    intersecting_samples = [
        sets.Capsule(torch.tensor([[-0.5, 1.0]]), torch.tensor([[-0.5, 3.0]]),
                     torch.tensor([0.5])),
        sets.Capsule(torch.tensor([[2.0, 5.0]]), torch.tensor([[1.0, 5.0]]),
                     torch.tensor([0.5])),
    ]

    intersection_test(capsule, non_intersecting_samples, intersecting_samples)


def test_capsule_box_intersection():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([1.0]))

    non_intersecting_samples = [
        sets.Box(torch.tensor([[0.0, 4.5]]), torch.tensor([[[0.5, 0.0],
                                                            [0.0, 0.5]]])),
    ]

    intersecting_samples = [
        sets.Box(torch.tensor([[4.0, 2.0]]), torch.tensor([[[1.0, 0.0],
                                                            [0.0, 1.0]]])),
        sets.Box(torch.tensor([[0.0, 1.0]]), torch.tensor([[[0.5, 0.5],
                                                            [0.5, -0.5]]])),
    ]

    intersection_test(capsule, non_intersecting_samples, intersecting_samples)


def test_capsule_zonotope_intersection():
    capsule = sets.Capsule(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]),
                           torch.tensor([1.0]))

    non_intersecting_samples = [
        sets.Zonotope(torch.tensor([[3.0, 0.0]]), 0.5 * torch.tensor([[[1.0, 0.5, 0.2],
                                                                       [0.5, 1.0, 0.3]]])),
    ]

    intersecting_samples = [
        sets.Zonotope(torch.tensor([[1.0, 2.0]]), torch.tensor([[[1.0, 0.5, 0.2],
                                                                 [0.5, 1.0, 0.3]]])),
    ]

    intersection_test(capsule, non_intersecting_samples, intersecting_samples)


def test_box_box_intersection():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))

    non_intersecting_samples = [
        sets.Box(torch.tensor([[5.0, 0.0]]), torch.tensor([[[1.0, 1.0],
                                                            [1.0, -1.0]]])),
        sets.Box(torch.tensor([[5.0, 0.5]]), torch.tensor([[[1.0, 1.0],
                                                            [1.0, -1.0]]])),
    ]

    intersecting_samples = [
        sets.Box(torch.tensor([[4.0, 2.0]]), torch.tensor([[[1.0, 0.0],
                                                            [0.0, 1.0]]])),
        sets.Box(torch.tensor([[4.0, 2.5]]), torch.tensor([[[0.15, 0.25 / 3],
                                                            [0.05, -0.25]]])),
        sets.Box(torch.tensor([[3.0, 3.0]]), torch.tensor([[[0.15, 0.25 / 3],
                                                            [0.05, -0.25]]])),
    ]

    intersection_test(box, non_intersecting_samples, intersecting_samples)


def test_box_zonotope_intersection():
    box = sets.Box(torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 1.0 / 3],
                                                              [1.0, -1.0]]]))

    non_intersecting_samples = [
        sets.Zonotope(torch.tensor([[3.0, 0.0]]), 0.5 * torch.tensor([[[1.0, 0.5, 0.2],
                                                                       [0.5, 1.0, 0.3]]])),
    ]

    intersecting_samples = [
        sets.Zonotope(torch.tensor([[0.0, 0.0]]), 0.5 * torch.tensor([[[1.0, 0.5, 0.2],
                                                                       [0.5, 1.0, 0.3]]])),
    ]

    intersection_test(box, non_intersecting_samples, intersecting_samples)


def test_zonotope_zonotope_intersection():
    zonotope = sets.Zonotope(torch.tensor([[1.0, 2.0]]), torch.tensor([[
        [1.0, 1.3, 2.4, -1.0],
        [1.0, -1.0, 0.0, 1.5]
    ]]))

    non_intersecting_samples = [
        sets.Zonotope(torch.tensor([[-3.0, -4.0]]), 0.5 * torch.tensor([[[1.0, 0.5, 0.2],
                                                                         [0.5, 1.0, 0.3]]])),
    ]

    intersecting_samples = [
        sets.Zonotope(torch.tensor([[0.0, 0.0]]), 0.5 * torch.tensor([[[1.0, 0.5, 0.2],
                                                                       [0.5, 1.0, 0.3]]])),
    ]

    intersection_test(zonotope, non_intersecting_samples, intersecting_samples)
