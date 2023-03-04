import numpy as np
import progressbar
from matplotlib import pyplot as plt
from matplotlib import lines
import QubitSim2.constant as ct
import QubitSim2.function as fun


class Circuit():
    def __init__(self):
        # ====================================================================
        self.qubit_number = 0
        self.qubit_list = []
        self.connect_list = []
        self.simulator = None
        self.subspace = []
        self.dressed_eigenvalue = None
        self.dressed_featurevector = None
        self.time_evolution_operator = None
        self.time_evolution_operator_path = []
        self.time_evolution_operator_dressed = None
        self.time_evolution_operator_dressed_sub = None
        # ====================================================================

    def add_qubit(self, C, phi_r, I_c_1, I_c_2):
        """Qubit adding function.

        Args:
            C (float, optional): Self capacitor of qubit. Defaults to 0.
            phi_r (float, optional): Residual magnetic flux phase in qubit's DCSQUID. Defaults to 0.
            I_c_1 (float, optional): The critical current of first junction of qubit's DCSQUID. Defaults to 0.
            I_c_2 (float, optional): The critical current of second junction of qubit's DCSQUID. Defaults to 0.
        """
        self.qubit_list.append(Qubit(C, phi_r, I_c_1, I_c_2))
        self.qubit_number += 1

    def add_connect(self, left_qubit_index, right_qubit_index, connect_capacitor, connect_inductance):
        """Connect adding function.

        Args:
            left_qubit_index (int, optional): The left qubit's index.
            right_qubit_index (int, optional): The right qubit's index.
            connect_capacitor (float, optional): Connect capacitor.
            connect_inductance (float, optional): Connect inductance.
        """
        if (left_qubit_index < 0 or left_qubit_index > self.qubit_number-1) or (right_qubit_index < 0 or right_qubit_index > self.qubit_number-1):
            print("ERROR: Wrong Connect, Node Index Out Of Range!")
            exit()
        for i in range(len(self.connect_list)):
            if (min(left_qubit_index, right_qubit_index), max(left_qubit_index, right_qubit_index)) == (self.connect_list[i].left_qubit_index, self.connect_list[i].right_qubit_index):
                print("ERROR: Repetitive Connect!")
                exit()
        self.connect_list.append(Connect(min(left_qubit_index, right_qubit_index), max(
            left_qubit_index, right_qubit_index), connect_capacitor, connect_inductance))

    def add_signal(self, qubit_index, channel, signal_fun):
        """Signal adding function.

        Args:
            qubit_index (int): Qubit index which this signal would be added on.
            channel (str): Channel.
            signal_fun (function): Signal function.
        """
        if (qubit_index < 0 or qubit_index > self.qubit_number-1):
            print("ERROR: Node Index Out Of Range!")
            exit()
        if channel != 'x' and channel != 'z':
            print("ERROR: Unkonw channel!")
            exit()
        if channel == 'x':
            self.qubit_list[qubit_index].signal_x = signal_fun
        if channel == 'z':
            self.qubit_list[qubit_index].signal_z = signal_fun

    def set_simulation_parameter(self, t_start=0, t_end=20E-9, t_piece=1E-11, operator_order_num=4, trigonometric_function_expand_order_num=8, exponent_function_expand_order_num=15):
        """_summary_

        Args:
            t_start (float, optional): Starting time point. Defaults to 0.
            t_end (float, optional): Ending time point. Defaults to 20E-9.
            t_piece (float, optional): Piece time. Defaults to 1E-11.
            operator_order_num (int, optional): Operator_order_num. Defaults to 4.
            trigonometric_function_expand_order_num (int, optional): Trigonometric_function_expand_order_num. Defaults to 8.
            exponent_function_expand_order_num (int, optional): Exponent_function_expand_order_num. Defaults to 15.
        """
        self.simulator = Simulator(t_start, t_end, t_piece, operator_order_num, trigonometric_function_expand_order_num,
                                   exponent_function_expand_order_num)

    def run(self):
        # 1.Getting transformational matrix converting bare bases to dressed bases.
        # dressed_eigenvalue: Dressed states' energy eigenvalue.
        # dressed_featurevector: Transformational matrix converting bare bases to dressed bases
        self.dressed_eigenvalue, self.dressed_featurevector = self.transformational_matrix_generator(
            self.Hamiltonian_generator(0, 'z'))

        # 2.Simulation calculating the whole time evolution operator.
        p = progressbar.ProgressBar()
        self.time_evolution_operator = np.eye(
            self.simulator.operator_order_num**self.qubit_number)
        self.time_evolution_operator_path = []
        self.time_evolution_operator_path.append(np.matmul(np.linalg.inv(
            self.dressed_featurevector), np.matmul(self.time_evolution_operator, self.dressed_featurevector)))
        print("Calculating the whole time evolution operator:")
        for i in p(range(int(self.simulator.t_piece_num/2))):
            self.time_evolution_operator = np.matmul(
                self.time_evolution_operator_generator(i+1), self.time_evolution_operator)
            self.time_evolution_operator_path.append(np.matmul(np.linalg.inv(
                self.dressed_featurevector), np.matmul(self.time_evolution_operator, self.dressed_featurevector)))

        # 3.Dressed state process, subspace process, phase process.
        self.time_evolution_operator_dressed, self.time_evolution_operator_dressed_sub = self.dressed_state_subspace_phase_process(
            self.subspace)

    def energy_levels_show_process(self, H, subspace):
        """The function which is used to draw enrgy level with subspace

        Args:
            H (np.array): Hamiltonian.
            subspace (list[list[int]]): Subspace.
        """
        dressed_eigenvalue_temp, dressed_featurevector_temp = self.transformational_matrix_generator(
            H)
        dressed_eigenvalue_temp = dressed_eigenvalue_temp/ct.H/1E9
        dressed_eigenvalue_temp = dressed_eigenvalue_temp - \
            dressed_eigenvalue_temp[0]
        dressed_eigenvalue = np.zeros(len(subspace))

        for i in range(len(subspace)):
            dressed_eigenvalue[i] = dressed_eigenvalue_temp[self.dressed_state_index_find(
                subspace[i], dressed_featurevector_temp)]

        padding = dressed_eigenvalue.ptp() / 10
        figure = plt.figure(figsize=(2, 6))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_label_text('Energy')
        ax.set_ylim(dressed_eigenvalue.min() - padding,
                    dressed_eigenvalue.max() + padding)
        for eval_ in dressed_eigenvalue:
            line = lines.Line2D((0.3, 0.7), (eval_, eval_),
                                c='black')
            ax.add_line(line)

        for idx in range(len(subspace)):
            string = "|"
            for i in range(len(subspace[idx])):
                string = string+str(subspace[idx][i])
            string = string+">"
            ax.text(
                x=0.72, y=dressed_eigenvalue[idx], s=string, fontsize=10)
        plt.show()

    def M_C_generator(self):
        """The function calculationg capacitor matrix.

        Returns:
            np.array: Capacitor matrix.
        """
        M_C = np.zeros([self.qubit_number, self.qubit_number])
        for i in range(len(self.qubit_list)):
            M_C[i][i] = self.qubit_list[i].C
        for i in range(len(self.connect_list)):
            M_C[self.connect_list[i].left_qubit_index][self.connect_list[i]
                                                       .right_qubit_index] = self.connect_list[i].connect_capacitor
            M_C[self.connect_list[i].right_qubit_index][self.connect_list[i]
                                                        .left_qubit_index] = self.connect_list[i].connect_capacitor
        return M_C

    def M_Ec_generator(self):
        """The function calculationg capacitor energy matrix.

        Returns:
            np.array: Capacitor energy matrix.
        """
        M_Ec = self.M_C_generator()
        for i in range(len(self.qubit_list)):
            M_Ec[i][i] = np.sum(M_Ec[:, i])
        for i in range(len(self.qubit_list)):
            for j in range(len(self.qubit_list)):
                if i != j:
                    M_Ec[i][j] = -M_Ec[i][j]
        M_Ec = 0.5*ct.E**2*np.linalg.pinv(M_Ec)
        return M_Ec

    def M_L_generator(self, time=None):
        """The function calculationg inductance matrix at time.

        Args:
            time (float, optional): Time. Defaults to None. If time=None, this function would calculate the inductance matrix with no signal.

        Returns:
            np.array: Inductance matrix.
        """
        phi_list = []
        Ic_1 = []
        Ic_2 = []
        M_L = np.zeros([self.qubit_number, self.qubit_number])
        if (time != None):
            if (time < self.simulator.t_start or time > self.simulator.t_end):
                print("ERROR: Time out og simulator's range!")
                exit()
            for i in range(len(self.qubit_list)):
                phi_list.append(
                    self.qubit_list[i].phi_r+self.qubit_list[i].signal_z(time))
                Ic_1.append(self.qubit_list[i].I_c_1)
                Ic_2.append(self.qubit_list[i].I_c_2)
        else:
            for i in range(len(self.qubit_list)):
                phi_list.append(self.qubit_list[i].phi_r)
                Ic_1.append(self.qubit_list[i].I_c_1)
                Ic_2.append(self.qubit_list[i].I_c_2)

        for i in range(len(self.connect_list)):
            M_L[self.connect_list[i].left_qubit_index][self.connect_list[i]
                                                       .right_qubit_index] = self.connect_list[i].connect_inductance
            M_L[self.connect_list[i].right_qubit_index][self.connect_list[i]
                                                        .left_qubit_index] = self.connect_list[i].connect_inductance

        for i in range(self.qubit_number):
            M_L[i][i] = ct.PHI_ZERO/2/np.pi / \
                np.sqrt(Ic_1[i]**2+Ic_2[i]**2+2*Ic_1[i]
                        * Ic_2[i]*np.cos(2*phi_list[i]))

        return M_L

    def M_Ej_generator(self, time=None):
        """The function calculationg Josephson energy at time.

        Args:
            time (float, optional): Time. Defaults to None. If time=None, this function would calculate the Josephson energy matrix with no signal.

        Returns:
            np.array: Josephson energy matrix.
        """
        M_Ej_0 = ct.H**2/(4*np.pi**2*4*ct.E**2) / \
            self.M_L_generator(time)
        M_Ej = np.zeros([self.qubit_number, self.qubit_number])
        for i in range(self.qubit_number):
            for j in range(self.qubit_number):
                if (i == j):
                    M_Ej[i][j] = np.sum(M_Ej_0[:, i])
                else:
                    M_Ej[i][j] = -M_Ej_0[i][j]
        return M_Ej

    def operator_phi_generator(self, E_c, E_j, operator_order_num):
        """The function generating phase operator with order of operator_order_num.

        Args:
            E_c (float): Electric energy.
            E_j (float): Josephson energy.
            operator_order_num (int): Expanding order of operator. 

        Returns:
            np.array: Returned phase operator.
        """
        return np.power(2*E_c/E_j, 0.25)*(fun.creation_operator_n(operator_order_num)+fun.annihilation_operator_n(operator_order_num))

    def operator_n_generator(self, E_c, E_j, operator_order_num):
        """The function generating phase operator with order of operator_order_num.

        Args:
            E_c (float): Electric energy.
            E_j (float): Josephson energy.
            operator_order_num (int): Expanding order of operator. 

        Returns:
            np.array: Returned phase operator.
        """
        return complex(0, 0.5)*np.power(0.5*E_j/E_c, 0.25)*(fun.creation_operator_n(operator_order_num)-fun.annihilation_operator_n(operator_order_num))

    def tensor_identity_expand_generator(self, matrix, index):
        """The function to expand matrix with identity matrix.

        Args:
            matrix (np.array): The matrix to be expanded.
            index (int): Qubit index.

        Returns:
            np.array: The matrix expanded.
        """
        matrix_expand = 1
        for i in range(self.qubit_number):
            if i == index:
                matrix_expand = np.kron(matrix_expand, matrix)
            else:
                matrix_expand = np.kron(matrix_expand, np.eye(
                    self.simulator.operator_order_num))
        return matrix_expand

    def Hamiltonian_generator(self, n, mode='m'):
        """The function calculating the n'st time piece's Hamiltonian operator.

        Args:
            n (int): The n'st time piece.
            mode (str, optional): Calculating mode. Defaults to 'm'.

        Returns:
            np.array: The n'st time piece's Hamiltonian.
        """
        if mode == 'm':
            Hamiltonian = 0
            M_Ec = self.M_Ec_generator()
            M_Ej = self.M_Ej_generator(self.simulator.t_list[2*n-1])
            Y = complex(0, 1)*(fun.annihilation_operator_n(self.simulator.operator_order_num_change) -
                               fun.creation_operator_n(self.simulator.operator_order_num_change))/np.sqrt(2)

            for i in range(self.qubit_number):
                Hamiltonian_temp = 0.5*np.sqrt(8*M_Ec[i][i]*M_Ej[i][i])*np.matmul((Y-0*np.eye(self.simulator.operator_order_num_change)), (Y-0*np.eye(self.simulator.operator_order_num_change)))-M_Ej[i][i]*fun.cos_matrix_n(self.operator_phi_generator(M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num_change)-np.power(
                    8*M_Ec[i][i]/M_Ej[i][i], 0.25)*self.qubit_list[i].signal_x(self.simulator.t_list[2*n-1])*np.eye(self.simulator.operator_order_num_change), self.simulator.trigonometric_function_expand_order_num)+(M_Ej[i][i]-0.5*np.sqrt(8*M_Ej[i][i]*M_Ec[i][i]))*np.eye(self.simulator.operator_order_num_change)
                Hamiltonian_temp = Hamiltonian_temp[0:self.simulator.operator_order_num,
                                                    0:self.simulator.operator_order_num]
                Hamiltonian = Hamiltonian+self.tensor_identity_expand_generator(
                    Hamiltonian_temp, i)

            for i in range(self.qubit_number):
                for j in range(self.qubit_number):
                    if i != j:
                        Hamiltonian = Hamiltonian+4*M_Ec[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_n_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_n_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))
                        Hamiltonian = Hamiltonian+0.5*M_Ej[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_phi_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_phi_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))

            return Hamiltonian

        if mode == 'l':
            Hamiltonian = 0
            M_Ec = self.M_Ec_generator()
            M_Ej = self.M_Ej_generator(self.simulator.t_list[2*n-2])
            Y = complex(0, 1)*(fun.annihilation_operator_n(self.simulator.operator_order_num_change) -
                               fun.creation_operator_n(self.simulator.operator_order_num_change))/np.sqrt(2)

            for i in range(self.qubit_number):
                Hamiltonian_temp = 0.5*np.sqrt(8*M_Ec[i][i]*M_Ej[i][i])*np.matmul((Y-self.qubit_list[i].signal_x(self.simulator.t_list[2*n-2])*np.eye(self.simulator.operator_order_num_change)), (Y-self.qubit_list[i].signal_x(self.simulator.t_list[2*n-2])*np.eye(self.simulator.operator_order_num_change)))-M_Ej[i][i]*fun.cos_matrix_n(self.operator_phi_generator(M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num_change)-np.power(
                    8*M_Ec[i][i]/M_Ej[i][i], 0.25)*self.qubit_list[i].signal_x(self.simulator.t_list[2*n-1])*np.eye(self.simulator.operator_order_num_change), self.simulator.trigonometric_function_expand_order_num)+(M_Ej[i][i]-0.5*np.sqrt(8*M_Ej[i][i]*M_Ec[i][i]))*np.eye(self.simulator.operator_order_num_change)
                Hamiltonian_temp = Hamiltonian_temp[0:self.simulator.operator_order_num,
                                                    0:self.simulator.operator_order_num]
                Hamiltonian = Hamiltonian+self.tensor_identity_expand_generator(
                    Hamiltonian_temp, i)

            for i in range(self.qubit_number):
                for j in range(self.qubit_number):
                    if i != j:
                        Hamiltonian = Hamiltonian+4*M_Ec[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_n_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_n_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))
                        Hamiltonian = Hamiltonian+0.5*M_Ej[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_phi_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_phi_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))

            return Hamiltonian

        if mode == 'r':
            Hamiltonian = 0
            M_Ec = self.M_Ec_generator()
            M_Ej = self.M_Ej_generator(self.simulator.t_list[2*n])
            Y = complex(0, 1)*(fun.annihilation_operator_n(self.simulator.operator_order_num_change) -
                               fun.creation_operator_n(self.simulator.operator_order_num_change))/np.sqrt(2)

            for i in range(self.qubit_number):
                Hamiltonian_temp = 0.5*np.sqrt(8*M_Ec[i][i]*M_Ej[i][i])*np.matmul((Y-self.qubit_list[i].signal_x(self.simulator.t_list[2*n])*np.eye(self.simulator.operator_order_num_change)), (Y-self.qubit_list[i].signal_x(self.simulator.t_list[2*n])*np.eye(self.simulator.operator_order_num_change)))-M_Ej[i][i]*fun.cos_matrix_n(self.operator_phi_generator(M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num_change)-np.power(
                    8*M_Ec[i][i]/M_Ej[i][i], 0.25)*self.qubit_list[i].signal_x(self.simulator.t_list[2*n-1])*np.eye(self.simulator.operator_order_num_change), self.simulator.trigonometric_function_expand_order_num)+(M_Ej[i][i]-0.5*np.sqrt(8*M_Ej[i][i]*M_Ec[i][i]))*np.eye(self.simulator.operator_order_num_change)
                Hamiltonian_temp = Hamiltonian_temp[0:self.simulator.operator_order_num,
                                                    0:self.simulator.operator_order_num]
                Hamiltonian = Hamiltonian+self.tensor_identity_expand_generator(
                    Hamiltonian_temp, i)

            for i in range(self.qubit_number):
                for j in range(self.qubit_number):
                    if i != j:
                        Hamiltonian = Hamiltonian+4*M_Ec[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_n_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_n_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))
                        Hamiltonian = Hamiltonian+0.5*M_Ej[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_phi_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_phi_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))

            return Hamiltonian

        if mode == 'z':
            Hamiltonian = 0
            M_Ec = self.M_Ec_generator()
            M_Ej = self.M_Ej_generator()
            Y = complex(0, 1)*(fun.annihilation_operator_n(self.simulator.operator_order_num_change) -
                               fun.creation_operator_n(self.simulator.operator_order_num_change))/np.sqrt(2)

            for i in range(self.qubit_number):
                Hamiltonian_temp = 0.5*np.sqrt(8*M_Ec[i][i]*M_Ej[i][i])*np.matmul((Y-0*np.eye(self.simulator.operator_order_num_change)), (Y-0*np.eye(self.simulator.operator_order_num_change)))-M_Ej[i][i]*fun.cos_matrix_n(self.operator_phi_generator(M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num_change)-np.power(
                    8*M_Ec[i][i]/M_Ej[i][i], 0.25)*0*np.eye(self.simulator.operator_order_num_change), self.simulator.trigonometric_function_expand_order_num)+(M_Ej[i][i]-0.5*np.sqrt(8*M_Ej[i][i]*M_Ec[i][i]))*np.eye(self.simulator.operator_order_num_change)
                Hamiltonian_temp = Hamiltonian_temp[0:self.simulator.operator_order_num,
                                                    0:self.simulator.operator_order_num]
                Hamiltonian = Hamiltonian+self.tensor_identity_expand_generator(
                    Hamiltonian_temp, i)

            for i in range(self.qubit_number):
                for j in range(self.qubit_number):
                    if i != j:
                        Hamiltonian = Hamiltonian+4*M_Ec[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_n_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_n_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))
                        Hamiltonian = Hamiltonian+0.5*M_Ej[i][j]*np.matmul(self.tensor_identity_expand_generator(self.operator_phi_generator(
                            M_Ec[i][i], M_Ej[i][i], self.simulator.operator_order_num), i), self.tensor_identity_expand_generator(self.operator_phi_generator(M_Ec[j][j], M_Ej[j][j], self.simulator.operator_order_num), j))

            return Hamiltonian

    def transformational_matrix_generator(self, H_0):
        """The function generating transformational matrix converting bare bases to dressed bases.

        Returns:
            (np.array,np.array): The first return is eigenvalue list and the second return is featurevector matrix.
        """
        eigenvalue, featurevector_temp = np.linalg.eig(H_0)
        eigenvalue = np.real(eigenvalue)
        featurevector = np.zeros(
            [featurevector_temp.shape[0], featurevector_temp.shape[1]], dtype=complex)
        sort_index_list = np.argsort(eigenvalue)
        for i in range(len(sort_index_list)):
            featurevector[:, i] = featurevector_temp[:, sort_index_list[i]]
        eigenvalue = np.sort(eigenvalue)
        return (eigenvalue, featurevector)

    def time_evolution_operator_generator(self, n):
        """The function calculating the n'st time piece's time evolution operator.

        Args:
            n (int): The n'st time piece.

        Returns:
            np.array: The n'st time piece's time evolution operator.
        """
        t_piece = self.simulator.t_piece*1E9
        Hamiltonian_middle = self.Hamiltonian_generator(n, 'm')/ct.H/1E9
        Hamiltonian_left = self.Hamiltonian_generator(n, 'l')/ct.H/1E9
        Hamiltonian_right = self.Hamiltonian_generator(n, 'r')/ct.H/1E9
        Hamiltonian_I = (Hamiltonian_right-Hamiltonian_left)/t_piece
        Hamiltonian_II = 4*(Hamiltonian_right+Hamiltonian_left -
                            2*Hamiltonian_middle)/(t_piece**2)
        Hamiltonian_I0 = np.matmul(
            Hamiltonian_middle, Hamiltonian_I)-np.matmul(Hamiltonian_I, Hamiltonian_middle)

        time_evolution_operator = fun.exp_matrix_n(-2*np.pi*complex(0, 1)*(Hamiltonian_middle*t_piece+1/24*Hamiltonian_II *
                                                   t_piece**3)+4*np.pi**2/12*Hamiltonian_I0*t_piece**3, self.simulator.exponent_function_expand_order_num)

        return time_evolution_operator

    def dressed_state_subspace_phase_process(self, subspace):
        """The function converting time evolution operator from bare bases to dressed bases, subspace processing and phase reset processing.

        Args:
            subspace (list[list[int]]): Subspace.

        Returns:
            (np.array,np.array): The first return is the time evolution operator in dressed bases and the 
            second return is the sub time evolution operator in subsapce. 
        """
        time_evolution_operator_dressed = np.matmul(np.linalg.inv(
            self.dressed_featurevector), np.matmul(self.time_evolution_operator, self.dressed_featurevector))
        phase_gate = np.zeros([time_evolution_operator_dressed.shape[0],
                              time_evolution_operator_dressed.shape[1]], dtype=complex)
        for i in range(time_evolution_operator_dressed.shape[0]):
            phase_gate[i][i] = np.exp(
                2*np.pi*complex(0, 1)/ct.H*self.dressed_eigenvalue[i]*self.simulator.t_end)
        time_evolution_operator_dressed = np.matmul(phase_gate,
                                                    time_evolution_operator_dressed)

        index_list = []
        for i in range(len(subspace)):
            index_list.append(self.dressed_state_index_find(
                subspace[i], self.dressed_featurevector))
        time_evolution_operator_dressed_sub = np.zeros(
            [len(index_list), len(index_list)], dtype=complex)
        for i in range(len(index_list)):
            for j in range(len(index_list)):
                time_evolution_operator_dressed_sub[i][j] = time_evolution_operator_dressed[index_list[i]][index_list[j]]

        return (time_evolution_operator_dressed, time_evolution_operator_dressed_sub)

    def dressed_state_index_find(self, bare_state_list, dressed_featurevector):
        """The function finding the corresponding dress state's index according to the bare state's tag.

        Args:
            dressed_featurevector (np.array): Dressed featurevector.
            bare_state_list (list[int]): Bare state tag.

        Returns:
            int: The index of dressed state in dressed_featurevector.
        """
        bare_state_index = 0
        for i in range(self.qubit_number):
            bare_state_index = bare_state_index+bare_state_list[i] * \
                self.simulator.operator_order_num**(self.qubit_number-1-i)
        return np.argmax(np.abs(dressed_featurevector[bare_state_index, :]))


class Qubit():
    def __init__(self, C=0, phi_r=0, I_c_1=0, I_c_2=0):
        """Qubit class's initial function.

        Args:
            C (float, optional): Self capacitor of qubit. Defaults to 0.
            phi_r (float, optional): Residual magnetic flux phase in qubit's DCSQUID. Defaults to 0.
            I_c_1 (float, optional): The critical current of first junction of qubit's DCSQUID. Defaults to 0.
            I_c_2 (float, optional): The critical current of second junction of qubit's DCSQUID. Defaults to 0.
        """
        self.C = C
        self.phi_r = phi_r
        self.I_c_1 = I_c_1
        self.I_c_2 = I_c_2
        # signal_x: Signal function which would be added on the main loop.
        # signal_z: Signal function which would be added on the DCSQUID loop.
        self.signal_x = fun.const_zero
        self.signal_z = fun.const_zero


class Connect():
    def __init__(self, left_qubit_index=0, right_qubit_index=0, connect_capacitor=0, connect_inductance=0):
        """Connect class's initial function.

        Args:
            left_qubit_index (int, optional): The left qubit's index. Defaults to 0.
            right_qubit_index (int, optional): The right qubit's index. Defaults to 0.
            connect_capacitor (float, optional): Connect capacitor. Defaults to 0.
            connect_inductance (float, optional): Connect inductance. Defaults to 0.
        """
        self.left_qubit_index = left_qubit_index
        self.right_qubit_index = right_qubit_index
        self.connect_capacitor = connect_capacitor
        self.connect_inductance = connect_inductance


class Simulator():
    def __init__(self, t_start=0, t_end=20E-9, t_piece=1E-11, operator_order_num=4, trigonometric_function_expand_order_num=8, exponent_function_expand_order_num=15):
        """Simulator class's initial function.

        Args:
            t_start (float, optional): Starting time point. Defaults to 0.
            t_end (float, optional): Ending time point. Defaults to 20E-9.
            t_piece (float, optional): Piece time. Defaults to 1E-11.
            operator_order_num (int, optional): Operator_order_num. Defaults to 4.
            trigonometric_function_expand_order_num (int, optional): Trigonometric_function_expand_order_num. Defaults to 8.
            exponent_function_expand_order_num (int, optional): Exponent_function_expand_order_num. Defaults to 15.
        """
        self.t_start = t_start
        self.t_end = t_end
        self.t_piece = t_piece
        self.operator_order_num = operator_order_num
        self.trigonometric_function_expand_order_num = trigonometric_function_expand_order_num
        self.exponent_function_expand_order_num = exponent_function_expand_order_num

        # operator_order_num_change: Operator expanding order using to calculating H0.
        # t_piece_num: 2*Number of piece time.
        # t_list: Time list.
        self.operator_order_num_change = self.operator_order_num+5
        self.t_piece_num = int(
            np.round(2*(self.t_end-self.t_start)/self.t_piece))
        self.t_list = np.linspace(self.t_start, self.t_end, self.t_piece_num+1)
