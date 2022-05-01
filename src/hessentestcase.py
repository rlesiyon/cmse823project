from hessenberg import *
from scipy.linalg import hessenberg as hessen
import numpy as np


class hessenTestCase: 

    def __init__(self, testMatrix = None):

        self.testMatrix = testMatrix

    def compare_inbuilt(self):

       impl_output =  hessenberg_form(self.testMatrix)
       inb_output = hessen(self.testMatrix)
       
       diffl2norm = np.linalg.norm(impl_output - inb_output, 2)

       return diffl2norm

    def compare_bound(self, tol):

       l2diff =  self.compare_inbuilt()

       if l2diff < tol:
            print(f'Calculated hessenberg form is of order {tol} lower')
       else:
           print(f'Comparision difference {l2diff}')
