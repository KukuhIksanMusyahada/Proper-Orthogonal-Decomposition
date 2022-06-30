from POD_Lib import Calculation as calc





if __name__=='__main__':
    sol_mat= calc.sol_matrix(names='CD')
    U,s,V = calc.perform_SVD()
    print(U.shape,'\n', s.shape, '\n', V.shape)