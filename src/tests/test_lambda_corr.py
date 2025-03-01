import numpy as np
import unittest
import sys
import lDGA.lambda_corr as lcorr
import lDGA.utilities as util

class TestLambdaCorr(unittest.TestCase):
    def test_uniform(self):
        Nq = 40; Nw=100; beta=10.0; dim=2
        chi_w_q = np.zeros( (2*Nw-1,Nq**dim), dtype=np.complex128 )
        chi_imp = np.zeros( 2*Nw-1, dtype=np.complex128 )
        lambda_imp = 0.43
        w_arr = (2.0*np.pi/beta)*np.linspace(-Nw,Nw,2*Nw-1,endpoint=True)
        
        for iq in range(Nq**dim):
            q=util.ik2k(iq,Nq,dim)
            chi_w_q[:,iq] = (0.5+np.sum(q*q) + w_arr**2)**(-1)
            
        chi_imp[:] = chi_w_q.shape[1]/np.sum( 1/chi_w_q + lambda_imp ,axis=1)
        root = lcorr.get_lambda_uniform(chi_w_q, chi_imp, beta,lambda_0 = 0.35)

        print("Root: ", root)
        print("Error: ", np.abs(root-lambda_imp))
        self.assertEqual( np.abs(root-lambda_imp) < 1e-5, True)

    def test_wdep(self):
        Nq = 40; Nw=100; beta=10.0; dim=2
        chi_w_q = np.zeros( (2*Nw-1,Nq**dim), dtype=np.complex128 )
        chi_imp = np.zeros( 2*Nw-1, dtype=np.complex128 )
        lambda_sol = np.linspace(0.02,0.4,2*Nw-1,dtype=np.complex128)
        lambda_sol = lambda_sol/(0.1+lambda_sol**2)
        lambda_imp = np.zeros( 2*Nw-1, dtype=np.complex128 )
        lambda_imp[:] = np.sum(lambda_sol)/lambda_sol.shape[0]
        w_arr = (2.0*np.pi/beta)*np.linspace(-Nw,Nw,2*Nw-1,endpoint=True)
        
        for iq in range(Nq**dim):
            q=util.ik2k(iq,Nq,dim)
            chi_w_q[:,iq] = (0.5+np.sum(q*q) + w_arr**2)**(-1)
            
        chi_imp[:] = chi_w_q.shape[1]/np.sum( 1/chi_w_q + lambda_sol.reshape(2*Nw-1,1) ,axis=1)
        root = lcorr.get_lambda_wdep(chi_w_q, chi_imp, beta, lambda_imp)

        print("Error: ", np.sum(np.abs(root-lambda_sol))/len(root) )
        self.assertEqual( np.all(np.abs(root-lambda_sol) < 1e-5), True)
    

if __name__ == '__main__':
    unittest.main()

