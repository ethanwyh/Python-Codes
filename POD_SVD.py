''' 
A simple tutorial on Proper Orthogonal Decomposition
This tutorial is essentially a Jupyterlab version of the MATLAB code presented in 'A Tutorial on the Proper Othorgonal Decomposition' 
Weiss J. A tutorial on the proper orthogonal decomposition. InAIAA aviation 2019 forum 2019 (p. 3333), https://arc.aiaa.org/doi/10.2514/6.2019-3333
Data can be found from: https://depositonce.tu-berlin.de/items/b4597a76-b5a0-4698-951c-7872f3435572
This is a code for Singular Value Decomposition POD
''' 
import tensorflow as tf
import scipy.io as scp
from scipy.linalg import svd
import numpy as np
import numpy.matlib as matl
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

##  Load in data                                  
# matlab =  scp.loadmat('D:/StarCCM+/Ryan Tunstall pipe/Conjugate Heat Transfer/tunstall_comparison/IDDES results/data_tsb.mat')
matlab =  scp.loadmat('<file directory>/data_tsb.mat')


## Reshape everything 
nt = matlab['Nt']                          # Number of time-slices
axeX_Z0 = matlab['axeX_Z0']                # X coordinates 
axeY_Z0 = matlab['axeY_Z0']                # Y coordinates
U_Z0_VER_Run1 = matlab['U_Z0_VER_Run1']    # U component of velocity
V_Z0_VER_Run1 = matlab['V_Z0_VER_Run1']    # V component of velocity

## Convert floats to numpy arrays
a = np.array(axeX_Z0) 
b = np.array(axeY_Z0)
c = np.array(U_Z0_VER_Run1)
d = np.array(V_Z0_VER_Run1)
e = np.array(nt)

X_raw = a.T
yr = np.flipud(b-b[0,0])
Y_raw = yr.T
U_raw = np.transpose(c, (1, 0, 2))
V_raw = np.transpose(d, (1, 0, 2))

# # resample every 3 points
axis_row =  len(X_raw)
axis_col =  len(X_raw[0])
vel_row  =  len(U_raw)
vel_col  =  len(U_raw[0])

# # resample every 3 points
# # NOTE the different syntax used for both MATLAB and Python!!
# # Python: [Pages x Rows x Columns]
# # MATLAB: [Rows x Columns x Pages]
X  = X_raw[0:axis_row:3,0:axis_col:3]
Y  = Y_raw[0:axis_row:3,0:axis_col:3]
SU = U_raw[0:vel_row:3,0:vel_col:3]  #[45 pages, 129 rows, 3580 cols]
SV = V_raw[0:vel_row:3,0:vel_col:3]
SU_corrected = SU.transpose(2, 0, 1) #[3580 pages, 45 rows, 129 cols]



## Create snapshot matrix
Nt = np.size(SU, 2)

## Reshape data into a matrix S with Nt rows
S = SU_corrected.reshape(Nt, -1) #[3580 rows, 5805 cols]
U = S - matl.repmat(np.mean(S, axis = 0), Nt, 1)


# Solve eigenvalue problem via SVD
# L = temporal modes
# diag(SIG)^2 = eigenvalues
# R = spatial modes / eigenvectors
denom = math.sqrt(Nt-1)
main_U = U/denom
L,SIG,R = svd(main_U) 
PHI = R                                 # PHI is spatial      

A = np.dot(main_U,PHI)                  # A is temporal

sigma = np.square(SIG)
eigenvalue = sigma.reshape(Nt, 1)       # eigenvalues


# sort eigenvalues and vectors
# sort eigenvale and PHI (where PHI = R)
LAM_s, A_s = zip(*sorted(zip(eigenvalue,A), reverse=True))
l_sort = np.array(LAM_s)
l_sort = np.reshape(l_sort, (l_sort.size, 1))
a_sort = np.array(A_s)
a_sort = np.reshape(a_sort, (Nt, Nt))



# Reconstruct mode k in a loop

for k in range(4):
    phi_extract = PHI[:,[k]]
    A_extract   = a_sort[:,[k]]
    phi_trans = np.transpose(phi_extract)
    utilde_k = np.dot(A_extract,phi_trans)
    ut_permute = utilde_k.reshape(Nt, len(X), len(X[0])) 
    
    
    plt.subplot(5, 1, k+1)
    plt.figure(figsize=(8, 4), dpi=80)
    for i in range(Nt-1):
        plt.contourf(X, Y, ut_permute[i, :, :], cmap = cm.jet)
        break
        
    plt.title('Mode %i' %(k+1))
    ax =  plt.gca()
    ax.invert_yaxis()
    plt.xlabel('x mm ',fontsize=14)
    plt.ylabel('y mm ',fontsize=14)
    plt.colorbar()
    plt.show()


