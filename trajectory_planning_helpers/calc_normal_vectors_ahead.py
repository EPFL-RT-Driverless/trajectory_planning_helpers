import numpy as np
from .calc_tangent_vectors import calc_tangent_vectors
import math


def calc_normal_vectors_ahead(psi: np.ndarray) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    Use heading to calculate normalized (i.e. unit length) normal vectors. Normal vectors point in direction psi + pi/2.

    .. inputs::
    :param psi:                     array containing the heading of every point (north up, range [-pi,pi[).
    :type psi:                      np.ndarray

    .. outputs::
    :return normvec_normalized:     unit length normal vectors for every point [x, y].
    :rtype normvec_normalized:      np.ndarray

    .. notes::
    len(psi) = len(normvec_normalized)
    """

    # calculate tangent vectors
    tangvec_normalized = calc_tangent_vectors(psi=psi)

    # find normal vectors
    normvec_normalized = np.stack(
        (-tangvec_normalized[:, 1], tangvec_normalized[:, 0]), axis=1
    )

    return normvec_normalized
