from numba import cuda
import numpy as np
import math
from time import time
import QubitSim2_bak.function as fun


@cuda.jit
def subspace_Hamiltonian_generator_GPU(subspace_list, Hamiltonian_list, Hamiltonian_subspace):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < len(subspace_list)**2:
        col_index = idx % len(subspace_list)
        row_index = int(round((idx-col_index)/len(subspace_list)))
    temp = 1
    for i in range(len(Hamiltonian_list)):
        temp = temp * \
            Hamiltonian_list[i][int(subspace_list[row_index]
                                [i])][int(subspace_list[col_index][i])]
    Hamiltonian_subspace[row_index][col_index] = Hamiltonian_subspace[row_index][col_index] + temp


def main():
    Hamiltonian_list = []
    Hamiltonian_list.append(np.array([[1, 2], [2, 1]], dtype=complex))
    Hamiltonian_list.append(np.array([[3, 4], [1, 2]], dtype=complex))
    Hamiltonian_list.append(np.array([[1, 0], [0, 1]], dtype=complex))
    subspace_list = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [
        0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    Hamiltonian_subspace = cuda.device_array(
        [len(subspace_list), len(subspace_list)], dtype=complex)
    Hamiltonian_list_device=cuda.to_device(Hamiltonian_list)
    subspace_list_device=cuda.to_device(subspace_list)

    n = len(subspace_list)*len(subspace_list)
    threads_per_block = 16
    blocks_per_grid = math.ceil(n / threads_per_block)

    subspace_Hamiltonian_generator_GPU[[blocks_per_grid, threads_per_block]](
        subspace_list_device, Hamiltonian_list_device, Hamiltonian_subspace)
    cuda.synchronize()
    print(Hamiltonian_subspace.copy_to_host())


if __name__ == "__main__":
    main()
