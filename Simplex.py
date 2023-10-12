import numpy as np

class Simplex:
    """
        Inicializa el Simplex

        Parameters:
            self - Problema (PL)
            restriccions - m restricciones de (PL)
            b - La parte de la derecha de las restricciones
            c - La funcion objetivo
            fase - Valor 1 o no 1 dependiendo si es fase 1 o no
            ineq_s - lista que traiga -1 0 1 en el caso que todavia no tengamos el problema estandar
                     (-1 <= / 0 == / 1 >=)
        Returns:
            None
	"""
    def __init__(self, restriccions, b, c, fase, ineq_s = None):
        self.A = restriccions
        self.num_restriccions, self.num_variables =  np.shape(self.A)
        self.b = b
        self.c = c
        self.cur_SBF = None
        self.x = None
        self.cur_N = None
        self.N = None
        self.cur_matriu_B = None
        self.cur_z = None
        self.fase = fase

    """
        Obtiene la variable no basica que entrara para la proxima SBF (con Bland)

        Parameters:
            self - Problema (PL)
            r_q - Los costes reducidos correspondientes a las VNB
        Returns:
            El primer indice de la variable que tenga coste reducido negativo o -1 si ya hemos encontrado sol óptima
	"""
    def _get_next_q(self, r_q):
        for i in range(len(r_q)):
            if r_q[i] < 0:
                return i
        return -1                   #En este caso r_q >= [0]
    
    """
        Evalua el valor de la función objetivo en la base actual

        Parameters:
            self - Problema (PL)
        Returns:
            z = c'x_B
	"""
    def _get_z(self):
        return self.get_indexes(self.cur_SBF, self.c)@self.x
    
    """
        Obtiene la direccion basica de descenso

        Parameters:
            self - Problema (PL)
            index - indice de la VNB entrante
        Returns:
            Direccion basica de descenso d_B = -B^-1 A_q
	"""
    def _get_DB(self, index):
        return -self.cur_matriu_B @ (self.A.T[index].T)
    
    """
        Obtiene la longitud de paso maxima

        Parameters:
            self - Problema (PL)
            d_B - direccion basica de descenso para la que calculamos theta
        Returns:
            Theta* y el indice de la VB que sale
	"""
    def _get_theta(self, d_B):
        f_index = -1
        theta = 1e9
        for i in range(len(d_B)):
            if d_B[i] < 0:
                index = self.cur_SBF[i]
                if f_index == -1:
                    f_index = index
                if theta > -self.x[i]/d_B[i]:
                    theta = -self.x[i]/d_B[i]
                
        return theta, f_index
    
    """
        Obtiene matriz eta de cambio de base

        Parameters:
            self - Problema (PL)
            d_B - direccion basica de descenso
            index - indice de la VNB entrante
        Returns:
            Retorna la matriz H de cambio de base
	"""
    def compute_eta(self, d_B, index):
        eta = np.eye(self.num_restriccions)
        for i in range(self.num_restriccions):
            if i == index:
                eta[i][index] = -1/d_B[index]
            else:
                eta[i][index] = -d_B[i]/d_B[index]
        return eta
    
    """
        Obtiene las filas/columnas/elementos de una matriz o lista

        Parameters:
            self - Problema (PL)
            index_list - lista de indices
            element - el elemento del que queremos los elementos
            axis - 0 si se quiere de seguido o 1 en filas
        Returns:
            Los elementos respectivos a los indices
	"""
    def get_indexes(self, index_list, element, c_axis=0):
        answer = np.array([element[index_list[0]]])
        for index in index_list:
            if index == index_list[0]:
                continue
            answer = np.concatenate((answer, [element[index]]),axis=c_axis)
        return answer

    """
        Obtiene los costes reducidos de una SB

        Parameters:
            self - Problema (PL)
        Returns:
            Retorna los costes reducidos de las VNB
	"""
    def compute_r(self):
        chosen_c = self.get_indexes(self.cur_SBF,self.c,0)
        lambda_v = chosen_c @ self.cur_matriu_B
        c_N = self.get_indexes(self.cur_N, self.c, 0)
        a_N = self.get_indexes(self.cur_N, self.A.T, 0)
        return c_N - (lambda_v @ a_N.T)

    """
        Actualiza los valores y cambia de base

        Parameters:
            self - Problema (PL)
            VNB - Variable no basica entrante
            VB - Variable basica saliente
            theta - longitud de paso maxima
            d_B - direccion de descenso
        Returns:
            None
	"""
    def _cambiar_variables(self, VNB, VB, theta, d_B):
        index_1, index_2 = 0,0
        for i in range(len(self.cur_SBF)):
            if self.cur_SBF[i] == VB:
                index_1 = i
        for i in range(len(self.cur_N)):
            if self.cur_N[i] == VNB:
                index_2 = i
        eta = self.compute_eta(d_B, index_1)
        self.cur_matriu_B = eta @ self.cur_matriu_B
        self.x -= theta*d_B
        self.x[index_1] = theta                 #x_q = theta*
        self.cur_N[index_2], self.cur_SBF[index_1] = self.cur_SBF[index_1], self.cur_N[index_2]

    """
        Resuelve el problema de Programacion Lineal

        Parameters:
            self - Problema (PL)
        Returns:
            Factible - Un booleano que indica si (PL) es factible
            Ilimitado - Un booleano que indica si (PL) es ilimitado
            z - El valor optimo de la funcion objetivo (si lo tiene, None de lo contrario)
            base - Una base optima (si la tiene, None de lo contrario)
            B - La B^-1 de la base óptima, None de lo contrario
	"""
    def solve(self):
        #Inicializamos                                                      Algo STEP1
        if self.fase != 1:
            total_variables = self.num_variables+self.num_restriccions
            new_c = np.array([1 if i >= self.num_variables else 0 for i in range(total_variables)])
            new_A = np.concatenate((np.copy(self.A), np.eye(self.num_restriccions)), axis=1)
            _, _, z_0, self.cur_SBF, self.cur_matriu_B, self.x = Simplex(new_A, self.b, new_c, 1).solve()
            self.cur_N = [index for index in range(self.num_variables) if index not in self.cur_SBF]
            if (z_0 > 0):       #No hemos encontrado ninguna SBF
                return False, False, None, None, None, None
        else:
            self.cur_matriu_B = np.eye(self.num_restriccions)
            self.cur_SBF = [index for index in range(self.num_variables-self.num_restriccions, self.num_variables)]
            self.cur_N = [index for index in range(self.num_variables) if index not in self.cur_SBF]
            self.x = self.b
        Factible = True
        Ilimitado = False
        self.cur_z = self._get_z()
        while (True):
            r_q = self.compute_r()                                          #Algo STEP2
            VNB_entrant = self._get_next_q(r_q)
            if VNB_entrant == -1:
                if self.fase == 1:      #Modificamos las variables
                    pass
                break
            d_B = self._get_DB(VNB_entrant)                                 #Algo STEP3
            theta, VB_sortida = self._get_theta(d_B)                        #Algo STEP4

            if VB_sortida == -1:
                Ilimitado = False
                break
            self._cambiar_variables(VNB_entrant, VB_sortida, theta, d_B)    #Algo STEP5
        return Factible, Ilimitado, self._get_z(), self.cur_SBF, self.cur_matriu_B, self.x
    
a = np.array([[1,0,1],[0,1,0]], dtype=float)
b = np.array([1, 2], dtype=float)
c = np.array([1, 1, 0], dtype=float)
ans = Simplex(a, b, c, 2).solve()