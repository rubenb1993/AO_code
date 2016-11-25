import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin

def wrap_function(phase):
    number = np.floor((np.abs(phase) - np.pi)/ (2* np.pi)) + 1 #number of times 2pi has to be substracted/added in order to be in [-pi, pi)
    phase_wrap = (phase - (np.sign(phase) * number * 2 * np.pi))
    return phase_wrap

def delta_x(wrapped_phase):
    "Equation 2 from Ghiglia and Romero"
    delta_x = (wrapped_phase[:,1:] - wrapped_phase[:,:-1])
    zero_padding = np.zeros(wrapped_phase.shape) #bigger than deltax to ensure boundary conditions
    zero_padding[:,:-1] = delta_x
    delta_x_wrapped = wrap_function(zero_padding)
    return delta_x_wrapped

def delta_y(wrapped_phase):
    "Equation 3 from Ghiglia and Romero"
    delta_y = (wrapped_phase[1:,:] - wrapped_phase[:-1,:])
    zero_padding = np.zeros(wrapped_phase.shape) #bigger to ensure boundary conditions
    zero_padding[:-1,:] = delta_y
    delta_y_wrapped = wrap_function(zero_padding)
    return delta_y_wrapped

def rho_ij(wrapped_phase, **kwargs):
    if 'weighted' in kwargs:
        #initiate matrices to efficiently calculate rho and min(w_(i,j), w_(i-1,j)) etc
        shapes = list(Weight.shape)
        zero_weight_x = np.ones((shapes[0], shapes[1]+2))
        zero_weight_y = np.ones((shapes[0] + 2, shapes[1]))
        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))

        zero_weight_x[:, 1:shapes[1]+1] = Weight
        zero_weight_y[1:shapes[0]+1] = Weight
        
        min_mat_1 = np.minimum(zero_weight_x[:, 2:]**2, zero_weight_x[:, 1:-1]**2) ## min(w^2_(i+1, j), w^2_(i,j))
        min_mat_2 = np.minimum(zero_weight_x[:, 1:-1]**2, zero_weight_x[:, :-2]**2)## min(w^2_(i,j), w^2_(i-1, j)
        min_mat_3 = np.minimum(zero_weight_y[2:, :]**2, zero_weight_y[1:-1, :]**2) ##min(w^2_(i,j+1), w^2_(i,j))
        min_mat_4 = np.minimum(zero_weight_y[1:-1, :]**2, zero_weight_y[:-2, :]**2) ##min(w^2_(i,j), w^2_(i,j-1))

        zero_yy[1:, :] = delta_y(wrapped_phase) 
        zero_xx[:, 1:] = delta_x(wrapped_phase)
        
        rho_ij_x = min_mat_1 * zero_xx[:, 1:] - min_mat_2 * zero_xx[:, :-1]
        rho_ij_y = min_mat_3 * zero_yy[1:, :] - min_mat_4 * zero_yy[:-1, :]
        rho_ij = rho_ij_x + rho_ij_y
    else:
        shapes = list(wrapped_phase.shape)
        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))
        zero_xx[:, 1:] = delta_x(wrapped_phase)
        zero_yy[1:, :] = delta_y(wrapped_phase)
        rho_ij = (zero_xx[:, 1:] - zero_xx[:, :-1]) + (zero_yy[1:,:] - zero_yy[:-1, :])
    return rho_ij

def unwrap_phase_dct(wrapped_phase, xx, yy, ii, jj, M, N, **kwargs):
    "Direct phase unwrapping using Discrete Cosine Transform"
    if 'rho' in kwargs: #meaning it is already rho rather than a wrapped phase
        rho_ij_hat = fftp.dct(fftp.dct(wrapped_phase, axis = 0), axis = 1)
    else:
        rho_ij_hat = fftp.dct(fftp.dct(rho_ij(wrapped_phase, **kwargs), axis = 0), axis = 1)
    denom = 2 * (np.cos(np.pi * ii / M) + np.cos(np.pi * jj / N) - 2)
    denom[0,0] = 1.0 ## to avoid dividing by 0, therefore 0,0 is undefined and phi 0,0 is set to rho 0,0
    phi_ij_hat = rho_ij_hat/denom
    unwr_phase = 1.0/((4 * M*N)) * fftp.dct(fftp.dct(phi_ij_hat, axis = 0, type = 3), axis = 1, type = 3) #type 3 == inverse dct
    return unwr_phase

def ravel_indices(shape, *args):
    """given a shape and arbitrary number of vecotrs containing indices which are unraveled, return an arbitrary number of vectors which are raveled"""
    new_positions = []
    for arg in args:
        new_positions.append(np.ravel_multi_index(arg, shape))
    return new_positions

def least_squares_matrix(wrapped_phase, Weight):
    #### make matrix for solving 2D weighted wrapped phase
    N = list(Weight.shape)[0]
    Bdiag = -4.0 * sparse.eye(N**2)
    Bupper = sparse.eye(N**2, k = 1)
    Blower = sparse.eye(N**2, k = -1)
    Buppupp = sparse.eye(N**2, k = N)
    Blowlow = sparse.eye(N**2, k = -N)
    A = Bdiag + Bupper + Blower + Buppupp + Blowlow

    ### make unraveled indices for special nodes
    indices_left = (np.arange(1,N-1), [0]*len(np.arange(1,N-1)))
    indices_right = (np.arange(1,N-1), [N-1]*len(np.arange(1, N-1)))
    indices_top = ([0]*len(np.arange(1, N-1)) , np.arange(1, N-1))
    indices_bot = ([N-1]*len(np.arange(1, N-1)), np.arange(1, N-1))    

    index_top_left = ([0],[0])
    index_top_right = ([0], [N-1])
    index_bot_left = ([N-1], [0])
    index_bot_right = ([N-1], [N-1])

    ## find vectors for making weighted matrix
    shapes = list(Weight.shape)
    zero_weight_x = np.ones((shapes[0], shapes[1]+2))
    zero_weight_x[:, 1:shapes[1]+1] = Weight
    zero_weight_y = np.ones((shapes[0] + 2, shapes[1]))
    zero_weight_y[1:shapes[0]+1] = Weight

    #make minimum matrices
    min_mat_1 = np.minimum(zero_weight_x[:, 2:]**2, zero_weight_x[:, 1:shapes[1]+1]**2) ## min(w^2_(i+1, j), w^2_(i,j))
    min_mat_2 = np.minimum(zero_weight_x[:, 1:shapes[1]+1]**2, zero_weight_x[:, :shapes[1]]**2)## min(w^2_(i,j), w^2_(i-1, j)
    min_mat_3 = np.minimum(zero_weight_y[2:, :]**2, zero_weight_y[1:shapes[0]+1, :]**2) ##min(w^2_(i,j+1), w^2_(i,j))
    min_mat_4 = np.minimum(zero_weight_y[1:shapes[0]+1, :]**2, zero_weight_y[:-2, :]**2) ##min(w^2_(i,j), w^2_(i,j-1))

    #create diagonals & offsets form weighted A matrix
    diag_0_weighted = -min_mat_1.flatten() - min_mat_2.flatten() - min_mat_3.flatten() - min_mat_4.flatten()
    diag_1_weighted = min_mat_1.flatten()[:-1]
    diag_min1_weighted = min_mat_2.flatten()[1:]
    diag_N_weighted = min_mat_3.flatten()[:-N]
    diag_minN_weighted = min_mat_4.flatten()[N:]
    diagonals = np.array([diag_0_weighted, diag_1_weighted, diag_min1_weighted, diag_N_weighted, diag_minN_weighted])
    offsets = np.array([0, 1, -1, N, -N])

    #create sparse weighted A matrix
    A_weighted = sparse.diags(diagonals, offsets)

    ## create (un)weightened forcing vectors
    rho_test_phase = rho_ij(wrapped_phase)
    weighted_test_phase = rho_ij(wrapped_phase, weighted = True, W = Weight)
    c = rho_test_phase.flatten()
    c_weighted = weighted_test_phase.flatten()

    ## ravel indices of special nodes
    indices_left_ravel, indices_right_ravel, indices_top_ravel, indices_bot_ravel = ravel_indices((N, N), indices_left, indices_right, indices_top, indices_bot)
    top_left_ravel, top_right_ravel, bot_left_ravel, bot_right_ravel = ravel_indices((N, N), index_top_left, index_top_right, index_bot_left, index_bot_right)

    ## Adjust values of boundary nodes according to boundary conditions to ensure positive defniteness and symmetry## 
    A = A.tolil()
    A_weighted = A_weighted.tolil()
    
    #left boundary indices
    A[indices_left_ravel, indices_left_ravel] *= 0.5
    A[indices_left_ravel, indices_left_ravel + N] *= 0.5
    A[indices_left_ravel, indices_left_ravel - N] *= 0.5
    A[indices_left_ravel, indices_left_ravel-1] = 0.0
    
    A_weighted[indices_left_ravel, indices_left_ravel] *= 0.5
    A_weighted[indices_left_ravel, indices_left_ravel + N] *= 0.5
    A_weighted[indices_left_ravel, indices_left_ravel - N] *= 0.5
    A_weighted[indices_left_ravel, indices_left_ravel-1] = 0.0
    #right boundary indices
    A[indices_right_ravel, indices_right_ravel] *= 0.5
    A[indices_right_ravel, indices_right_ravel + 1] = 0.0
    A[indices_right_ravel, indices_right_ravel + N] *= 0.5
    A[indices_right_ravel, indices_right_ravel - N] *= 0.5

    A_weighted[indices_right_ravel, indices_right_ravel] *= 0.5
    A_weighted[indices_right_ravel, indices_right_ravel + 1] = 0.0
    A_weighted[indices_right_ravel, indices_right_ravel + N] *= 0.5
    A_weighted[indices_right_ravel, indices_right_ravel - N] *= 0.5
    #top boundary indices
    A[indices_top_ravel, indices_top_ravel] *= 0.5
    A[indices_top_ravel, indices_top_ravel + 1] *= 0.5
    A[indices_top_ravel, indices_top_ravel - 1] *= 0.5

    A_weighted[indices_top_ravel, indices_top_ravel] *= 0.5
    A_weighted[indices_top_ravel, indices_top_ravel + 1] *= 0.5
    A_weighted[indices_top_ravel, indices_top_ravel - 1] *= 0.5
    #bottom boundary indices
    A[indices_bot_ravel, indices_bot_ravel] *= 0.5
    A[indices_bot_ravel, indices_bot_ravel + 1] *= 0.5
    A[indices_bot_ravel, indices_bot_ravel - 1] *= 0.5

    A_weighted[indices_bot_ravel, indices_bot_ravel] *= 0.5
    A_weighted[indices_bot_ravel, indices_bot_ravel + 1] *= 0.5
    A_weighted[indices_bot_ravel, indices_bot_ravel - 1] *= 0.5
    #top left corner
    A[top_left_ravel, top_left_ravel] *= 0.25
    A[top_left_ravel, top_left_ravel +1] *= 0.5
    A[top_left_ravel, top_left_ravel +N] *= 0.5

    A_weighted[top_left_ravel, top_left_ravel] *= 0.25
    A_weighted[top_left_ravel, top_left_ravel +1] *= 0.5
    A_weighted[top_left_ravel, top_left_ravel +N] *= 0.5
    #top right corner
    A[top_right_ravel, top_right_ravel] *= 0.25
    A[top_right_ravel, top_right_ravel - 1] *= 0.5
    A[top_right_ravel, top_right_ravel + N] *= 0.5
    A[top_right_ravel, top_right_ravel + 1] *= 0.0

    A_weighted[top_right_ravel, top_right_ravel] *= 0.25
    A_weighted[top_right_ravel, top_right_ravel - 1] *= 0.5
    A_weighted[top_right_ravel, top_right_ravel + N] *= 0.5
    A_weighted[top_right_ravel, top_right_ravel + 1] *= 0.0
    #bottom left corner
    A[bot_left_ravel, bot_left_ravel] *= 0.25
    A[bot_left_ravel, bot_left_ravel + 1] *= 0.5
    A[bot_left_ravel, bot_left_ravel - 1] *= 0.0
    A[bot_left_ravel, bot_left_ravel - N] *= 0.5

    A_weighted[bot_left_ravel, bot_left_ravel] *= 0.25
    A_weighted[bot_left_ravel, bot_left_ravel + 1] *= 0.5
    A_weighted[bot_left_ravel, bot_left_ravel - 1] *= 0.0
    A_weighted[bot_left_ravel, bot_left_ravel - N] *= 0.5
    #bottom right corner
    A[bot_right_ravel, bot_right_ravel - 1] *= 0.5
    A[bot_right_ravel, bot_right_ravel - N] *= 0.5
    A[bot_right_ravel, bot_right_ravel] *= 0.25

    A_weighted[bot_right_ravel, bot_right_ravel - 1] *= 0.5
    A_weighted[bot_right_ravel, bot_right_ravel - N] *= 0.5
    A_weighted[bot_right_ravel, bot_right_ravel] *= 0.25

    #adjust for the fact that orignially it is negative definite and defined on the 2x2 square
    A *= -1 ## to ensure positivie semi-definiteness
    A_weighted *= -1
    c = -c ## to ensure that phi does not change sign due to the previous alteration
    c_weighted = -c_weighted   ## the boundary conditions are already accounted for in c by the way it is built up (i.e. eq.6 Ghiglia and Romero)
    
    A = A.tocsr()
    A_weighted = A_weighted.tocsr()    
    return A, c, A_weighted, c_weighted

def pcg_algorithm(Q, c, N, tol, xx, yy, ii, jj):
    "PCG algorithm (3) from Ghiglia and Romero"
    k = 0
    phi = np.zeros(N**2)
    r_new = c
    r_old = c
    while k <= N**2:
        r_new_ravel = r_new.reshape((N, N))
        z_new = unwrap_phase_dct(r_new_ravel, xx, yy, ii, jj, N, N, rho = True).flatten()
        k += 1
        if k == 1:
            p = z_new
            rz_dot_new = (r_new.T).dot(z_new)
            beta = 1
        else:
            rz_dot_new = (r_new.T).dot(z_new)
            rz_dot_old = (r_old.T).dot(z_old)
            beta =  rz_dot_new/rz_dot_old
            p = z_new + beta * p
        Qp = Q.dot(p)
        alpha = (rz_dot_new /((p.T).dot(Qp)))
        phi = phi + alpha * p
        r_old = r_new
        r_new = r_old -  alpha * Qp
        print "i = " + str(k) + "  res = " + str(np.linalg.norm(r_new))
        if np.linalg.norm(r_new) <= tol:
            print("converged with residuals")
            break
        z_old = z_new
    return phi

def picard_it(c, c_weighted, k_max, Q, A, N):
    "algorithm 2 from Ghiglia and Romero. DOES NOT WORK WELL (YET?)"
    D = (Q - A)
    k = 0
    phi = c_weighted
    for k in range(k_max):
        res = D.dot(phi)
        rho = c_weighted - res
        phi = sparse.linalg.cg(A, rho)[0] /(2*N)
    return phi

### set up physical constants and a test phase
x, y = np.linspace(-1, 1, 512), np.linspace(-1, 1, 512)
xx, yy = np.meshgrid(x, y)
phase = 23 * xx + 25 * yy
phase[10:100, 10:100] += 10 * (np.random.rand(90,90) - 0.5)
wr_phase = wrap_function(phase)
wr_phase += 0.6 * (np.random.rand(xx.shape[0], yy.shape[1]) - 0.5)

yshape, xshape = list(wr_phase.shape)
x, y = np.linspace(-1, 1, xshape), np.linspace(-1, 1, yshape)
xx, yy = np.meshgrid(x,y)
i, j = np.linspace(0, len(x)-1, len(x)), np.linspace(0, len(y)-1, len(y))
ii, jj = np.meshgrid(i, j)
M, N = len(x), len(y)

## DCT unwrapping
unwr_phase = unwrap_phase_dct(wr_phase, xx, yy, ii, jj, M, N)

#start plotting results
f, axarr = plt.subplots(2,3)
axarr[0,1].imshow(unwr_phase, cmap = 'bone', vmin = np.floor(np.min(phase)/10)*10, vmax = np.ceil(np.max(phase)/10)*10)
axarr[0,1].set_title('DCT retrieved phase')
axarr[0,0].imshow(phase, cmap = 'bone', vmin = np.floor(np.min(phase)/10)*10, vmax = np.ceil(np.max(phase)/10)*10)
axarr[0,0].set_title('Original phase')
axarr[1,0].imshow(wr_phase, cmap = 'bone', vmin = -4, vmax = 4)
axarr[1,0].set_title('Original wr phase')
axarr[1,1].imshow(wrap_function(unwr_phase), cmap = 'bone', vmin = -4, vmax = 4)
axarr[1,1].set_title('DCT rewrapped phase')

## Set up weight where there is noise
Weight_zeros = np.ones((N,N))
Weight_zeros[10:100, 10:100] = 0
Weight = Weight_zeros
indices_0 = np.where(Weight_zeros==0)
weight_factor = 0.0
Weight[indices_0] = weight_factor

A, c, A_weighted, c_weighted = least_squares_matrix(wr_phase, Weight)
phi = pcg_algorithm(A_weighted, c_weighted, N, 1e-13, xx, yy, ii, jj).reshape(phase.shape)

#plot rest of results
im2 = axarr[1,2].imshow(wrap_function(phi), cmap='bone', vmin = -4, vmax = 4)
axarr[1,2].set_title('PCG wrapped phase')
im =axarr[0,2].imshow(phi, cmap='bone', vmin = np.floor(np.min(phase)/10)*10, vmax = np.ceil(np.max(phase)/10)*10)
axarr[0,2].set_title('PCG retrieved phase')
f.subplots_adjust(right=0.8)
cbar_ax1 = f.add_axes([0.85, 0.55, 0.05, 0.35])
cbar_ax2 = f.add_axes([0.85, 0.10, 0.05, 0.35])
f.colorbar(im, cax=cbar_ax1)
f.colorbar(im2, cax=cbar_ax2)
for i in range(6):
    j = np.unravel_index(i, axarr.shape)
    axarr[j].get_xaxis().set_ticks([])
    axarr[j].get_yaxis().set_ticks([])

plt.show()
