from numba import cuda
import numpy as np
import math
from time import time
import QubitSim2_bak.function as fun


@cuda.jit
def subspace_Hamiltonian_generator_GPU(qubit_number, subspace_list, Hamiltonian_list, Hamiltonian_subspace):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < len(subspace_list)**2:
        col_index = idx % len(subspace_list)
        row_index = int(round((idx-col_index)/len(subspace_list)))
    temp = 1
    for i in range(qubit_number):
        temp = temp * \
            Hamiltonian_list[i][subspace_list[row_index]
                                [i]][subspace_list[col_index][i]]
    Hamiltonian_subspace[row_index][col_index] = Hamiltonian_subspace[row_index][col_index] + temp


def main():
    qubit_number = 3
    Hamiltonian_list = []
    Hamiltonian_list.append(np.array([[1, 2], [2, 1]]))
    Hamiltonian_list.append(np.array([[3, 4], [1, 2]]))
    Hamiltonian_list.append(np.array([[1, 0], [0, 1]]))
    subspace_list = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [
        0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    Hamiltonian_subspace = np.zeros([len(subspace_list), len(subspace_list)])
    n = len(subspace_list)*len(subspace_list)
    threads_per_block = 16
    blocks_per_grid = math.ceil(n / threads_per_block)

    subspace_Hamiltonian_generator_GPU[[blocks_per_grid, threads_per_block]](
        qubit_number, subspace_list, Hamiltonian_list, Hamiltonian_subspace)
    cuda.synchronize()
    print(Hamiltonian_subspace)


# if __name__ == "__main__":
#     main()

Hamiltonian_list=[np.eye(4)]*3
print(Hamiltonian_list)