import numpy as np
from matplotlib import pyplot as plt
import QubitSim2_bak.model

circuit = QubitSim2_bak.model.Circuit()

# 创建比特
C = 88.1E-15+1E-20
phi_r = 0.0*np.pi
I_c_1 = 2.8E-4/18000
I_c_2 = 2.8E-4/18000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 88.1E-15+1E-20
phi_r = 0.12*np.pi
I_c_1 = 2.8E-4/18000
I_c_2 = 2.8E-4/18000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 88.1E-15+1E-20
phi_r = 0.08*np.pi
I_c_1 = 2.8E-4/18000
I_c_2 = 2.8E-4/18000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 88.1E-15+1E-20
phi_r = 0.13*np.pi
I_c_1 = 2.8E-4/18000
I_c_2 = 2.8E-4/18000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 125.4E-15+1E-20
phi_r = 0.39*np.pi
I_c_1 = 2.8E-4/3000
I_c_2 = 2.8E-4/2000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 125.4E-15+1E-20
phi_r = 0.39*np.pi
I_c_1 = 2.8E-4/3000
I_c_2 = 2.8E-4/2000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 125.4E-15+1E-20
phi_r = 0.39*np.pi
I_c_1 = 2.8E-4/3000
I_c_2 = 2.8E-4/2000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

C = 125.4E-15+1E-20
phi_r = 0.39*np.pi
I_c_1 = 2.8E-4/3000
I_c_2 = 2.8E-4/2000
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

# 创建连接
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(0, 4, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(1, 4, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(1, 5, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(2, 5, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(2, 6, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(3, 6, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(3, 7, C, L)
C = 10.11E-15+1E-20
L = 1
circuit.add_connect(0, 7, C, L)

# 设置仿真参数
t_start = 0
t_end = 20E-11
t_piece = 1E-11
operator_order_num = 4
trigonometric_function_expand_order_num = 8
low_energy_tag = 1
high_energylevel_num = 1
circuit.set_simulation_parameter(t_start, t_end, t_piece, operator_order_num,
                                 trigonometric_function_expand_order_num, low_energy_tag, high_energylevel_num)

# 设置子空间
circuit.subspace = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]

# 设置信号函数


def signal_fun(t):
    Amplitude = 0.00365
    f01_Q1 = 4.7035E9
    phi = np.pi-0.3356
    Envolope = 1-np.cos(2*np.pi*(t)/(20E-9))
    return Amplitude*Envolope*np.cos(2*np.pi*f01_Q1*t+phi)


# 将信号加入量子线路
qubit_index = 0
circuit.add_signal(qubit_index, 'x', signal_fun)

circuit.run()

X2PQ1_idleQ2_matrix = circuit.time_evolution_operator_dressed_sub
print(X2PQ1_idleQ2_matrix)
