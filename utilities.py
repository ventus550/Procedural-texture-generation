import numpy as np


def cells(matrix):
    """More compact way of traversing a matrix.

        >> for y, x in cells(matrix): ...
    """
    n, m, *_ = matrix.shape
    for y in range(n):
        for x in range(m):
            yield y, x


def softmax(matrix):
    e = np.exp(matrix - np.max(matrix))
    return e/e.sum()


def norm(matrix, axis=None):
    """Normalize matrix into a [-1.0, 1.0] range of values."""
    return matrix / np.max(np.abs(matrix), axis=axis)


def vectorize(matrix, intensity=0.01, time=40, alpha=1.0, seed=0):
    """Create vector field from matrix."""
    assert(len(matrix.shape) == 2)
    assert(intensity >= 0)
    assert(time >= 0)

    # Create vector field
    n, m = matrix.shape
    pimatrix = np.pi * matrix
    field = np.dstack((np.cos(pimatrix), np.sin(pimatrix)))

    # Create particles
    arn, arm = np.arange(n), np.arange(m)
    coords = np.array(np.meshgrid(arn, arm), dtype=np.float64).T.reshape(-1, 2)
    np.random.seed(seed)
    particles = coords[np.random.choice(coords.shape[0], int(
        len(coords) * intensity), replace=False), :]

    # Simulate particle movements
    trail = np.zeros_like(matrix)
    for _ in range(time):
        for p in range(len(particles)):
            pos = tuple(np.int64(np.floor(particles[p])))
            trail[pos] += 1
            particles[p] += field[pos]
            particles %= matrix.shape
    return trail**alpha / np.max(trail)


def weight_sum(weights, noises):
    """Weight sum of noises."""
    assert len(weights) == len(noises)
    return sum(weight * noise for weight, noise in zip(weights, noises))


def combine_noises(weights, noises):
    return weight_sum(weights, noises)/sum(weights)
