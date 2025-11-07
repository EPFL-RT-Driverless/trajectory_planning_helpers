#  Copyright (c) 2022. Tudor Oancea, EPFL Racing Team Driverless

import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad as quadrature
from scipy.interpolate import CubicSpline

from trajectory_planning_helpers import uniform_spline_from_points


def main(**kwargs):
    data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "example_files/fs_track.csv"),
        delimiter=",",
        skiprows=1,
    )
    left_cones = data[:, :2]
    right_cones = data[:, 2:4]
    center_line_points = data[:, 4:6]

    # declare uniform spline
    new_points, sigma = uniform_spline_from_points(
        ref_points=center_line_points,
        nbr_interpolation_points=2 * center_line_points.shape[0],
    )
    M = sigma.size - 1
    L = sigma[-1]
    new_x_cl = CubicSpline(
        sigma,
        new_points[:, 0],
        bc_type="periodic",
    )
    new_y_cl = CubicSpline(
        sigma,
        new_points[:, 1],
        bc_type="periodic",
    )
    new_length = lambda new_t1, new_t2: quadrature(
        lambda u: np.sqrt(new_x_cl(u, 1) ** 2 + new_y_cl(u, 1) ** 2),
        new_t1,
        new_t2,
    )[0]
    lam = np.zeros(M)
    for j in range(M):
        lam[j] = new_length(sigma[j], sigma[j + 1])
    computed_sigma = np.insert(np.cumsum(lam), 0, 0.0)

    # declare naive spline
    N = center_line_points.shape[0] - 1
    t = np.linspace(0.0, L, N + 1)
    x_cl = CubicSpline(
        t,
        center_line_points[:, 0],
        bc_type="periodic",
    )
    y_cl = CubicSpline(
        t,
        center_line_points[:, 1],
        bc_type="periodic",
    )
    length = lambda t1, t2: quadrature(
        lambda u: np.sqrt(x_cl(u, 1) ** 2 + y_cl(u, 1) ** 2),
        t1,
        t2,
    )[0]
    l = np.zeros(N)
    for i in range(N):
        l[i] = length(t[i], t[i + 1])
    s = np.insert(np.cumsum(l), 0, 0.0)

    if kwargs.get("plot", False):
        # plot the two reference path to make sure they are the same
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(
            x_cl(np.linspace(0.0, N, 4 * N, dtype=float)),
            y_cl(np.linspace(0.0, N, 4 * N, dtype=float)),
            "r-",
        )
        plt.plot(
            new_x_cl(np.linspace(0.0, L, 4 * M, dtype=float)),
            new_y_cl(np.linspace(0.0, L, 4 * M, dtype=float)),
            "g-",
        )
        plt.legend(["cl", "new_cl"])
        plt.plot(left_cones[:, 0], left_cones[:, 1], "y+")
        plt.plot(right_cones[:, 0], right_cones[:, 1], "b+")
        plt.axis("equal")
        plt.grid("on")
        plt.title("map of original and uniformized curve")

        # plot the new arc length vs the old one
        plt.subplot(1, 2, 2)
        plt.plot(t, s, "r-")
        plt.plot(sigma, computed_sigma, "g-")
        # plt.plot([0.0, L], [0.0, L], "b-")
        plt.legend(["s", "sigma", "ref"])
        plt.axis("equal")
        plt.grid("on")

        plt.show()


if __name__ == "__main__":
    main(plot=True)
