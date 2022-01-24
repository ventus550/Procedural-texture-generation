import numpy as np
from noise import pnoise2, snoise2
from random import randint
import opensimplex


def __noise(noise_funtion, resolution, octaves=8, persistence=5.0, lacunarity=0.5, repeat=None, seed=None):
    "wrapper for library noise functions"
    assert (resolution >= 1)
    assert (octaves >= 1)

    # Settle some technicalities
    persistence += 1e-5
    lacunarity += 1e-5
    if seed is None:
        seed = randint(1, 4389192)
    if repeat is None:
        repeat = resolution

    # Parameterize the noise function
    def noise(x, y):
        return noise_funtion(x, y,
                             octaves=octaves,
                             persistence=persistence,
                             lacunarity=lacunarity,
                             repeatx=repeat,
                             repeaty=repeat,
                             base=seed)

    return np.array([[noise(x, y) for x in range(resolution)]
                     for y in range(resolution)])


def simplex(resolution,
            octaves=8,
            persistence=5.0,
            lacunarity=0.5,
            repeat=None,
            seed=None):
    return __noise(snoise2,
                   resolution,
                   octaves=octaves,
                   persistence=persistence,
                   lacunarity=lacunarity,
                   repeat=repeat,
                   seed=seed)


def perlin(resolution,
           octaves=8,
           persistence=5.0,
           lacunarity=0.5,
           repeat=None,
           seed=None):
    return __noise(pnoise2,
                   resolution,
                   octaves=octaves,
                   persistence=persistence,
                   lacunarity=lacunarity,
                   repeat=repeat,
                   seed=seed)


def generate_perlin_noise_2d(shape, res=(1, 1)):
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    shape, res = np.array(shape), np.array(res)
    delta = res / shape
    d = shape // res
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2,
                                                                    0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def smooth(distance, resolution):
    return generate_perlin_noise_2d(
        (resolution * distance, resolution * distance), (distance, distance))


def distances(A, B):
    # Smoking hot distance matrix :D
    assert A.shape[1] == B.shape[1]
    A_dots = np.sum(A**2, axis=1).reshape((-1, 1)) * np.ones(len(B))
    B_dots = np.sum(B**2, axis=1) * np.ones((len(A), 1))
    D_squared = A_dots + B_dots - 2 * (A @ B.T)
    return np.sqrt(D_squared)


def worley(resolution, points=2, n=1):
    assert resolution**2 > points
    assert points > n and points > 0
    assert n >= 1

    ar = np.arange(resolution)
    region = np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)
    points = np.random.randint(resolution, size=(points, 2))
    dists = distances(region, points)
    return np.partition(dists, n)[:, n - 1:n].flatten().reshape(
        (resolution, resolution)) / resolution


def heat(resolution, temperature, x, y):
    assert 30 >= temperature >= 0
    assert resolution >= x >= 0
    assert resolution >= y >= 0

    ar = np.arange(resolution)
    rx, ry = np.array(np.meshgrid(ar, ar))
    rx = np.abs(rx - x)
    ry = np.abs(ry - y)
    dists = np.sqrt(rx**2 + ry**2)

    region = 2 * dists / np.max(dists) - 1
    region[region < 1] -= temperature - 20
    region /= np.max(np.abs(region))
    return -region


def pure_simplex(size, seed=1234):
    opensimplex.seed(seed)
    value = np.zeros((size, size))
    for y in range(size):
        for x in range(size):
            nx = x / size - 0.5
            ny = y / size - 0.5
            value[y][x] = opensimplex.noise2(x=nx, y=ny)
    return value
