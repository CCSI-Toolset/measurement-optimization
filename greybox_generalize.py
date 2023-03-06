import numpy as np
import pyomo.environ as pyo
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

class LogDetModel(ExternalGreyBoxModel):
    def __init__(self, num_para=2, use_exact_derivatives=True,verbose=True):
        self._use_exact_derivatives = use_exact_derivatives
        self.verbose = verbose
        self.num_para = num_para
        self.num_input = int(num_para + (num_para*num_para-num_para)//2)

        # For use with exact Hessian
        self._output_con_mult_values = np.zeros(1)

        if not use_exact_derivatives:
            raise NotImplementedError("use_exact_derivatives == False not supported")
        
    def input_names(self):
        input_name_list = []
        for i in range(self.num_para):
            for j in range(i,self.num_para):
                input_name_list.append("ele_"+str(i)+"_"+str(j))
                
        return input_name_list

    def equality_constraint_names(self):
        # no equality constraints
        return [ ]
    
    def output_names(self):
        return ['log_det']

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 1
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def finalize_block_construction(self, pyomo_block):
        ele_to_order = {}
        count  = 0
        # initialize, set up LB and UB
        
        for i in range(self.num_para):
            for j in range(i, self.num_para):
                # get rid of j,i 
                ele_to_order[(i,j)], ele_to_order[(j,i)] = count, count 
                str_name = 'ele_'+str(i)+"_"+str(j)
                # identity matrix 
                if i==j:
                    pyomo_block.inputs[str_name].value = 1
                else:
                    pyomo_block.inputs[str_name].value = 0
                    
                count += 1 
                
        self.ele_to_order = ele_to_order

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):

        # Not sure what this function should return with no equality constraints
        return None
    
    def evaluate_outputs(self):
        # form matrix
        M = []
        for i in range(self.num_para):
            M.append([])
            for k in range(self.num_para):                
                M[-1].append(self._input_values[self.ele_to_order[(i,k)]])

        M = np.asarray(M)

        # compute log determinant
        (sign, logdet) = np.linalg.slogdet(M)

        if self.verbose:
            print("\n Consider M =\n",M)
            print("   logdet = ",logdet,"\n")

        return np.asarray([logdet], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        return None

    def evaluate_jacobian_outputs(self):


        if self._use_exact_derivatives:
            # make a private function 
            # make a np array here 
            M = []
            count = 0 
            # use symmetry
            for i in range(self.num_para):
                M.append([])
                for k in range(self.num_para):
                    M[-1].append(self._input_values[self.ele_to_order[(i,k)]])

            M = np.asarray(M)

            # compute inverse
            Minv = np.linalg.pinv(M)

            row = np.zeros(self.num_input)
            col = np.zeros(self.num_input)
            data = np.zeros(self.num_input)
            
            for i in range(self.num_para):
                for j in range(i, self.num_para):
                    order = self.ele_to_order[i,j]
                    if i==j: # factor = 2 
                        row[order], col[order], data[order] = (0,order, Minv[i,j])
                    
                    else:
                        row[order], col[order], data[order] = (0,order, 2*Minv[i,j])
                    
            return coo_matrix((data, (row, col)), shape=(1, self.num_input))
  