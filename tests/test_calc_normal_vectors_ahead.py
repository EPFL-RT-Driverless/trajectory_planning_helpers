import numpy as np
from trajectory_planning_helpers import calc_normal_vectors_ahead

if __name__ == "__main__":
    psi_test = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi, -np.pi / 2])
    print("Result:\n", calc_normal_vectors_ahead(psi=psi_test))
