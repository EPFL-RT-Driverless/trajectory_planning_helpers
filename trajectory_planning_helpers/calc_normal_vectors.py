import numpy as np
from .calc_normal_vectors_ahead import calc_normal_vectors_ahead
import math


def calc_normal_vectors(psi: np.ndarray) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    Use heading to calculate normalized (i.e. unit length) normal vectors. Normal vectors point in direction psi - pi/2.

    .. inputs::
    :param psi:                     array containing the heading of every point (north up, range [-pi,pi[).
    :type psi:                      np.ndarray

    .. outputs::
    :return normvec_normalized:     unit length normal vectors for every point [x, y].
    :rtype normvec_normalized:      np.ndarray

    .. notes::
    len(psi) = len(normvec_normalized)
    """

    # calculate normal vectors
    normvec_normalized = -calc_normal_vectors_ahead(psi=psi)

    return normvec_normalized
