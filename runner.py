from POD_Lib import Calculation as calc





if __name__=='__main__':
    sol_mat= calc.sol_matrix(names='CD')
    print(f'Size of solution matrix is {sol_mat.shape}')
    U,s,V = calc.perform_SVD()
    print(f'Size of U matrix from SVD is {U.shape}')
    k = calc.calc_K(s)
    U_hat = calc.calc_U_hat(U,k)
    delta_hat = calc.calc_delta_hat(U_hat)
    print(f'Size of delta hat matrix is {delta_hat.shape}')