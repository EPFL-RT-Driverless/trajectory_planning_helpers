#  Copyright (c) 2022. Tudor Oancea, EPFL Racing Team Driverless
# splunif : A very simple package that constructs 2D uniform splines, whose continuous
# parameters correspond to the arc length of the spline.
#
# This code implements the algorithm described in
# Wang, Hongling & Kearney, Joseph & Atkinson, Kendall. (2002). Arc-length parameterized spline curves for real-time simulation.

from typing import Tuple, List, Union

import numpy as np
from scipy.integrate import quadrature
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect


def uniform_spline_from_points(
    ref_points: np.ndarray,
    nbr_interpolation_points: int = None,
    additional_ref_points: List[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[np.ndarray]]
]:
    """
    Fits a spline using the give reference points and computes new breaks and
    interpolation points on it that will yield a uniform spline (i.e. whose continuous
    parameter corresponds to the arc length).

    The total length of the curve corresponds to the last returned break point.

    :param ref_points: array of initial reference points, shape=(N,2)
    :type ref_points: np.ndarray
    :param nbr_interpolation_points: the number of generated breaks / interpolation points.
    If none specified, will default to 2*N.
    :type nbr_interpolation_points: int
    :param additional_ref_points: list of additional values to be interpolated to the spline.
    :type: list of np.ndarray

    :return: new interpolation points (np.ndarray of shape (nbr_interpolation_points, 2))
     and the break points to use to fit the uniform spline.
    """
    assert (
        len(ref_points.shape) == 2 and ref_points.shape[1] == 2
    ), "Parameter ref_points has wrong shape : {}".format(ref_points.shape)
    N = ref_points.shape[0] - 1
    if nbr_interpolation_points is None:
        nbr_interpolation_points = 2 * N

    t = np.linspace(0.0, N, N + 1, dtype=float)
    x_ref = CubicSpline(
        t,
        ref_points[:, 0],
    )
    y_ref = CubicSpline(
        t,
        ref_points[:, 1],
    )
    length = lambda t1, t2: quadrature(
        lambda u: np.sqrt(x_ref(u, 1) ** 2 + y_ref(u, 1) ** 2),
        t1,
        t2,
    )[0]

    # Step 1 : find the lengths of each segment of the original curve =================
    l = np.zeros(N)
    for i in range(N):
        l[i] = length(t[i], t[i + 1])

    s = np.insert(np.cumsum(l), 0, 0.0)
    L = s[-1]

    # Step 2 : find the uniformization points ========================================
    M = nbr_interpolation_points
    t_tilde = np.zeros(M + 1)
    t_tilde[-1] = N
    lam = L / M * np.ones(M)
    sigma = np.insert(np.cumsum(lam), 0, 0.0)
    assert (
        np.max(np.abs(sigma - L / M * np.arange(M + 1))) < 1e-10
    ), "sigma is not well computed"

    for j in range(M - 1):
        i = np.searchsorted(s, sigma[j + 1], side="right")
        obj = (
            lambda upper_bound: length(t[i - 1], upper_bound) - sigma[j + 1] + s[i - 1]
        )
        t_tilde[j + 1] = bisect(
            obj,
            t[i - 1],
            t[i],
        )

    # step 3 : construct the new re-parametrized spline ==============================
    new_points = np.zeros((M + 1, 2))
    for j in range(M + 1):
        new_points[j, 0] = x_ref(t_tilde[j])
        new_points[j, 1] = y_ref(t_tilde[j])

    if additional_ref_points is None:
        return new_points, sigma
    else:
        additional_results = []
        for points in additional_ref_points:
            if len(points.shape) == 1:
                spline = CubicSpline(t, points)
                additional_results.append(
                    np.apply_along_axis(spline, 0, t_tilde).ravel()
                )
            elif len(points.shape) == 2 and points.shape[1] == 2:
                spline1 = CubicSpline(t, points[:, 0])
                spline2 = CubicSpline(t, points[:, 1])
                additional_results.append(
                    np.transpose(
                        np.array(
                            [
                                np.apply_along_axis(spline1, 0, t_tilde),
                                np.apply_along_axis(spline2, 0, t_tilde),
                            ]
                        )
                    )
                )
            else:
                raise ValueError(
                    "additional points have wrong shape :{}".format(points.shape)
                )

        return new_points, sigma, additional_results


# TODO: add new function to construct a uniform spline from coeffs_x, coeffs_y
