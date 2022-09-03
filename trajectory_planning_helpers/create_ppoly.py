import numpy as np
from scipy.interpolate import PPoly
from typing import Tuple


def create_ppoly(coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> Tuple[PPoly, PPoly]:
    """Create a piecewise polynomial from the coefficients of the splines.

    :param coeffs_x: coefficients of the splines in x-direction
    :type coeffs_x:  np.ndarray
    :param coeffs_y: coefficients of the splines in y-direction
    :type coeffs_y:  np.ndarray
    :return:         piecewise polynomial
    :rtype:          CubicSpline
    """
    x_spline = PPoly(coeffs_x.T, np.arange(coeffs_x.shape[0] + 1, dtype=np.float64))
    y_spline = PPoly(coeffs_y.T, np.arange(coeffs_y.shape[0] + 1, dtype=np.float64))

    return x_spline, y_spline
