from POD_Lib import Calculation as calc





if __name__=='__main__':
    sol_mat= calc.sol_matrix(names='CD')
    print(sol_mat.shape)