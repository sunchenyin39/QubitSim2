# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:18:21 2023

@author: 97832
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
import QubitSim2.model
#%%
# 创建一个量子线路
circuit = QubitSim2.model.Circuit()
#%% 创建结点
Cq = 85E-15
PhiQ = 0
Cc = 100E-15
PhiC = 0.35
Rq = 11000*2
Rc = 1500*2
Cqc = 8E-15
Cqq = 0.5E-15

# 创建qubit
C = Cq+Cqc
phi_r = PhiQ*np.pi
I_c_1 = 2.8E-4/Rq
I_c_2 = 2.8E-4/Rq
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

# 创建coupler
C = Cc
phi_r = PhiC*np.pi
I_c_1 = 2.8E-4/Rc 
I_c_2 = 2.8E-4/Rc 
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

# 创建qubit
C = Cq
phi_r = PhiQ*np.pi
I_c_1 = 2.8E-4/Rq
I_c_2 = 2.8E-4/Rq
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

# 创建coupler
C = Cc
phi_r = PhiC*np.pi
I_c_1 = 2.8E-4/Rc 
I_c_2 = 2.8E-4/Rc 
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

# 创建qubit
C = Cq+Cqc
phi_r = PhiQ*np.pi
I_c_1 = 2.8E-4/Rq
I_c_2 = 2.8E-4/Rq
circuit.add_qubit(C, phi_r, I_c_1, I_c_2)

#%% 创建连接

C_default = 1E-20
L_default = 1

circuit.add_connect(0, 1, Cqc, L_default)
circuit.add_connect(1, 2, Cqc, L_default)
circuit.add_connect(2, 3, Cqc, L_default)
circuit.add_connect(3, 4, Cqc, L_default)
circuit.add_connect(0, 2, Cqq, L_default)
circuit.add_connect(2, 4, Cqq, L_default)

#%%
# 设置仿真参数
t_start = 0
t_end = 0.5
t_piece = 0.01
operator_order_num = 4
trigonometric_function_expand_order_num = 8
exponent_function_expand_order_num = 15
circuit.set_simulation_parameter(t_start, t_end, t_piece, operator_order_num)
#%%
circuit.subspace = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
#%%
def signal_fun(t):
    return t

# 将信号加入量子线路
qubit_index = 1
circuit.add_signal(qubit_index,'z',signal_fun)
#%%
for ii in range(50):
    print(ii)
    Hamiltonian = circuit.Hamiltonian_generator(0,'z')
    # [eigenvalue, featurevector] = transformational_matrix_generator(Hamiltonian)
    

# %%
