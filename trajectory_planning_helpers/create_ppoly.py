import numpy as np
from scipy.interpolate import PPoly
from typing import Tuple


def create_ppoly(
    coeffs_x: np.ndarray,
    coeffs_y: np.ndarray,
    breaks_x: np.ndarray = None,
    breaks_y: np.ndarray = None,
    periodic: bool = False,
) -> Tuple[PPoly, PPoly]:
    """Create a piecewise polynomial from the coefficients of the splines.

    :param coeffs_x: coefficients of the splines in x-direction
    :type coeffs_x:  np.ndarray
    :param coeffs_y: coefficients of the splines in y-direction
    :type coeffs_y:  np.ndarray
    :param breaks_x: breaks of the splines in x-direction, optional
    :type breaks_x:  np.ndarray
    :param breaks_y: breaks of the splines in y-direction, optional. If not specified and breaks_x are specified, the breaks in x-direction are used.
    :type breaks_y:  np.ndarray
    :param periodic: true if the spline is closed, false otherwise.

    :return: x_spline
    :rtype: PPoly
    :return: y_spline
    :rtype: PPoly
    """
    if breaks_y is None and breaks_x is not None:
        breaks_y = breaks_x

    x_spline = PPoly(
        coeffs_x.T,
        breaks_x
        if breaks_x is not None
        else np.arange(coeffs_x.shape[0] + 1, dtype=np.float64),
        extrapolate="periodic" if periodic else False,
    )
    y_spline = PPoly(
        coeffs_y.T,
        breaks_y
        if breaks_y is not None
        else np.arange(coeffs_y.shape[0] + 1, dtype=np.float64),
        extrapolate="periodic" if periodic else False,
    )

    return x_spline, y_spline
