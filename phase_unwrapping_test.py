import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftp
import scipy.sparse as sparse
import scipy.io as io
import scipy.sparse.linalg as lin
#import scikits-sparse.cholmod as cholmod

def wrap_function(phase):
    number = np.floor((np.abs(phase) - np.pi)/ (2* np.pi)) + 1 #number of times 2pi has to be substracted/added in order to be in [-pi, pi)
    phase_wrap = (phase - (np.sign(phase) * number * 2 * np.pi))
    return phase_wrap

def delta_x(wrapped_phase, **kwargs):
##    if 'weighted' in kwargs:
##        W_min = np.minimum(W[:, 1:]**2, W[:, :wrapped_phase.shape[1]-1]**2)
##        delta_x = W_min * (wrapped_phase[:,1:] - wrapped_phase[:,:wrapped_phase.shape[1]-1])
##        zero_padding = np.zeros(wrapped_phase.shape)
##        zero_padding[:,:-1] = delta_x
##        delta_x_wrapped = wrap_function(zero_padding)
##    else:
    delta_x = (wrapped_phase[:,1:] - wrapped_phase[:,:wrapped_phase.shape[1]-1])
    zero_padding = np.zeros(wrapped_phase.shape)
    zero_padding[:,:-1] = delta_x
    delta_x_wrapped = wrap_function(zero_padding)
    return delta_x_wrapped

def delta_y(wrapped_phase, **kwargs):
##    if 'weighted' in kwargs:
##        W_min_1= np.minimum(W[1:, :]**2, W[:wrapped_phase.shape[0]-1, :])
##        delta_y = W_min * (wrapped_phase[1:,:] - wrapped_phase[:wrapped_phase.shape[0]-1,:])
##        zero_padding = np.zeros(wrapped_phase.shape)
##        zero_padding[:-1,:] = delta_y
##        delta_y_wrapped = wrap_function(zero_padding)
##    else:
    delta_y = (wrapped_phase[1:,:] - wrapped_phase[:wrapped_phase.shape[0]-1,:])
    zero_padding = np.zeros(wrapped_phase.shape)
    zero_padding[:-1,:] = delta_y
    delta_y_wrapped = wrap_function(zero_padding)
    return delta_y_wrapped

def rho_ij(wrapped_phase, **kwargs):
##    if 'test' in kwargs:
##        shapes = list(wrapped_phase.shape)
##        zero_xx = np.zeros((shapes[0],shapes[1]+1))
##        zero_yy = np.zeros((shapes[0]+1, shapes[1]))
##        zero_xx[:, 1:] = delta_x(wrapped_phase, **kwargs)
##        zero_xx[:, -1] = 0
##        zero_yy[1:, :] = delta_y(wrapped_phase, **kwargs)
##        zero_yy[-1, :] = 0
##        rho_ij = (zero_xx[:, 1:] - zero_xx[:, :-1]) + (zero_yy[1:,:] - zero_yy[:-1, :])
    if 'weighted' in kwargs:
        ## weighted x part
##        shapes = list(Weight.shape)
##        zero_weight_x = np.ones((shapes[0], shapes[1]+2))
##        zero_weight_x[:, 1:shapes[1]+1] = Weight
##        min_mat_1 = np.minimum(zero_weight_x[:, 2:]**2, zero_weight_x[:, 1:shapes[1]+1]**2)
##        min_mat_2 = np.minimum(zero_weight_x[:, 1:shapes[1]+1]**2, zero_weight_x[:, :shapes[1]]**2)
        shapes = list(Weight.shape)
        zero_weight_x = np.ones((shapes[0], shapes[1]+2))
        zero_weight_x[:, 1:shapes[1]+1] = Weight
        min_mat_1 = np.minimum(zero_weight_x[:, 2:]**2, zero_weight_x[:, 1:shapes[1]+1]**2) ## min(w^2_(i+1, j), w^2_(i,j))
        min_mat_2 = np.minimum(zero_weight_x[:, 1:shapes[1]+1]**2, zero_weight_x[:, :shapes[1]]**2)## min(w^2_(i,j), w^2_(i-1, j)
        zero_weight_y = np.ones((shapes[0] + 2, shapes[1]))
        zero_weight_y[1:shapes[0]+1] = Weight
        min_mat_3 = np.minimum(zero_weight_y[2:, :]**2, zero_weight_y[1:shapes[0]+1, :]**2) ##min(w^2_(i,j+1), w^2_(i,j))
        min_mat_4 = np.minimum(zero_weight_y[1:shapes[0]+1, :]**2, zero_weight_y[:-2, :]**2) ##min(w^2_(i,j), w^2_(i,j-1))

        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))
        zero_yy[1:, :] = delta_y(wrapped_phase, **kwargs) 
        zero_xx[:, 1:] = delta_x(wrapped_phase, **kwargs)
        rho_ij_x = min_mat_1 * zero_xx[:, 1:] - min_mat_2 * zero_xx[:, :-1]
        ## weighted y part
##        zero_weight_y = np.ones((shapes[0] + 2, shapes[1]))
##        zero_weight_y[1:shapes[0]+1] = Weight
##        min_mat_3 = np.minimum(zero_weight_y[2:, :]**2, zero_weight_y[1:shapes[0]+1, :]**2)
##        min_mat_4 = np.minimum(zero_weight_y[1:shapes[0]+1, :]**2, zero_weight_y[:-2, :]**2)
        rho_ij_y = min_mat_3 * zero_yy[1:, :] - min_mat_4 * zero_yy[:-1, :]
        rho_ij = rho_ij_x + rho_ij_y
    else:
        shapes = list(wrapped_phase.shape)
        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))
        zero_xx[:, 1:] = delta_x(wrapped_phase, **kwargs)
        zero_yy[1:, :] = delta_y(wrapped_phase, **kwargs)
        rho_ij = (zero_xx[:, 1:] - zero_xx[:, :-1]) + (zero_yy[1:,:] - zero_yy[:-1, :])
    return rho_ij

def unwrap_phase_dct(wrapped_phase, xx, yy, ii, jj, M, N, **kwargs):
    if 'rho' in kwargs:
        rho_ij_hat = fftp.dct(fftp.dct(wrapped_phase, axis = 0), axis = 1)
    else:
        rho_ij_hat = fftp.dct(fftp.dct(rho_ij(wrapped_phase, **kwargs), axis = 0), axis = 1)
    denom = 2 * (np.cos(np.pi * ii / M) + np.cos(np.pi * jj / N) - 2)
    denom[0,0] = 1.0
    phi_ij_hat = rho_ij_hat/denom
    unwr_phase = 1.0/((4 * M*N)) * fftp.dct(fftp.dct(phi_ij_hat, axis = 0, type = 3), axis = 1, type = 3)
    return unwr_phase

def ravel_indices(shape, *args):
    """given a certain range of indices which are labeled inside, return an arbitrary number of vectors filtered with those indices"""
    new_positions = []
    for arg in args:
        new_positions.append(np.ravel_multi_index(arg, shape))
    return new_positions

x, y = np.linspace(-1, 1, 256), np.linspace(-1, 1, 256)
xx, yy = np.meshgrid(x, y)
phase = 23 * xx + 25 * yy
phase[40:80, 40:80] += 1 * (np.random.rand(40,40) - 0.5)
wr_phase = wrap_function(phase)
#wr_phase += 0.6 * (np.random.rand(xx.shape[0], yy.shape[1]) - 0.5)

yshape, xshape = list(wr_phase.shape)
x, y = np.linspace(-1, 1, xshape), np.linspace(-1, 1, yshape)
xx, yy = np.meshgrid(x,y)
i, j = np.linspace(0, len(x)-1, len(x)), np.linspace(0, len(y)-1, len(y))
ii, jj = np.meshgrid(i, j)
M, N = len(x), len(y)


unwr_phase = unwrap_phase_dct(wr_phase, xx, yy, ii, jj, M, N)


f, axarr = plt.subplots(2,2)
axarr[0,1].imshow(unwr_phase, cmap = 'bone')
axarr[0,1].set_title('recovered phase')
axarr[0,0].imshow(phase, cmap = 'bone')
axarr[0,0].set_title('original phase')
axarr[1,0].imshow(wr_phase, cmap = 'bone', vmin = -np.pi, vmax = np.pi)
axarr[1,0].set_title('original wr phase')
axarr[1,1].imshow(wrap_function(unwr_phase), cmap = 'bone', vmin = -np.pi, vmax = np.pi)
axarr[1,1].set_title('rewrapped phase')
for i in range(4):
    j = np.unravel_index(i, axarr.shape)
    axarr[j].get_xaxis().set_ticks([])
    axarr[j].get_yaxis().set_ticks([])

plt.show()

W = np.ones((N,N))
W[36:84, 36:84] = 0
Weight = W
indices_0 = np.where(W==0)
weight_factor = 1
Weight[indices_0] = weight_factor


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
    min_mat_1 = np.minimum(zero_weight_x[:, 2:]**2, zero_weight_x[:, 1:shapes[1]+1]**2) ## min(w^2_(i+1, j), w^2_(i,j))
    min_mat_2 = np.minimum(zero_weight_x[:, 1:shapes[1]+1]**2, zero_weight_x[:, :shapes[1]]**2)## min(w^2_(i,j), w^2_(i-1, j)
    zero_weight_y = np.ones((shapes[0] + 2, shapes[1]))
    zero_weight_y[1:shapes[0]+1] = Weight
    min_mat_3 = np.minimum(zero_weight_y[2:, :]**2, zero_weight_y[1:shapes[0]+1, :]**2) ##min(w^2_(i,j+1), w^2_(i,j))
    min_mat_4 = np.minimum(zero_weight_y[1:shapes[0]+1, :]**2, zero_weight_y[:-2, :]**2) ##min(w^2_(i,j), w^2_(i,j-1))
    diag_0_weighted = -min_mat_1.flatten() - min_mat_2.flatten() - min_mat_3.flatten() - min_mat_4.flatten()
    diag_1_weighted = min_mat_1.flatten()[:-1]
    diag_min1_weighted = min_mat_2.flatten()[1:]
    diag_N_weighted = min_mat_3.flatten()[:-N]
    diag_minN_weighted = min_mat_4.flatten()[N:]
    diagonals = np.array([diag_0_weighted, diag_1_weighted, diag_min1_weighted, diag_N_weighted, diag_minN_weighted])
    offsets = np.array([0, 1, -1, N, -N])
    A_weighted = sparse.diags(diagonals, offsets)

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
    #adjust forcing vector
    c[top_left_ravel] *= 0.25
    c[top_right_ravel] *= 0.25
    c[bot_left_ravel] *= 0.25
    c[bot_right_ravel] *= 0.25
    c[indices_top_ravel] *= 0.5
    c[indices_bot_ravel] *= 0.5
    c[indices_left_ravel] *= 0.5
    c[indices_right_ravel] *= 0.5

    c_weighted[top_left_ravel] *= 0.25
    c_weighted[top_right_ravel] *= 0.25
    c_weighted[bot_left_ravel] *= 0.25
    c_weighted[bot_right_ravel] *= 0.25
    c_weighted[indices_top_ravel] *= 0.5
    c_weighted[indices_bot_ravel] *= 0.5
    c_weighted[indices_left_ravel] *= 0.5
    c_weighted[indices_right_ravel] *= 0.5

    #adjust for the fact that orignially it is negative definite and defined on the 2x2 square
    A /= 2
    A_weighted /= 2
    c = c
    c_weighted = c_weighted
    
    A = A.tocsr()
    A_weighted = A_weighted.tocsr()    
    return A, c, A_weighted, c_weighted

A, c, A_weighted, c_weighted = least_squares_matrix(wr_phase, Weight)
### adjust forcing vector values according to Neumann boundary conditions ###


indices_straight = np.arange(N**2)
indices_matrix = np.unravel_index(indices_straight, phase.shape)
#assert (np.all(c[indices_straight] == rho_test_phase[indices_matrix]))
#assert (np.all(rho_test_phase == c.reshape(rho_test_phase.shape)))
#c = sparse.csr_matrix(c)


##W_sparse = sparse.identity(N**2, format = 'lil')
indices_1 = np.ravel_multi_index(indices_0, W.shape)
##indices_weight = ((indices_1, indices_1))
##W_sparse[indices_weight] = weight_factor
##A_weighted[indices_1, :] = 0.0
##A_weighted[:, indices_1] = 0.0
#W_sparse[:, indices_1] = weight_factor
##A = A.tocsr()
##A_weighted = A_weighted.tocsr()
io.savemat('weighted_decompose.mat', dict(A_weighted = A_weighted))
io.savemat('to_decompose.mat', dict(A = A))
L = io.loadmat('chol_upper_permd.mat')
L = L['L'] #take matrix L from the structure L and save it in L
P = io.loadmat('chol_perm_matrix.mat')
P = P['S'] #get permutation matrix
R_t = P.dot(L.transpose())


####W_sparse.tocsr()
####A_t = A.transpose()
####W_t = W_sparse.transpose()
###Q = R_t.dot(W_t.dot(R_t.transpose()))
##Q = R_t.dot(W_sparse.dot(R_t.transpose()))
###Q = R_t.dot(W_t.dot(W_sparse.dot(R_t.transpose())))
####P = A_t.dot(A)

#### solve using cg method
k = 0
phi = np.zeros(N**2)
r_0 = c
r_new = c
r_old = c
def pcg_algorithm(Q, c, c_real, N, tol, xx, yy, ii, jj):
    k = 0
    phi = np.zeros(N**2)
    r_0 = c
    r_new = c
    r_old = c
    while k <= 400:
##        if k == 1:
##            r_new_ravel = c_real.reshape((N, N))
##        else:
##            r_new_ravel = r_new.reshape((N,N))
        r_new_ravel = r_new.reshape((N, N))
        z_new = -1 * unwrap_phase_dct(r_new_ravel, xx, yy, ii, jj, N, N, rho = True).flatten()
        #z_new = r_new
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
        alpha = (rz_dot_new / ((p.T).dot(Qp)))
        phi = phi + alpha * p
        r_old = r_new
        r_new = r_old -  alpha * Qp
        if k%10 == 0:
            print "i = " + str(k) + "  beta = " + str(beta) + "  res = " + str(np.linalg.norm(r_new))
        if np.linalg.norm(r_new) <= tol:
            break
        z_old = z_new
    return phi

phi = pcg_algorithm(A_weighted, c_weighted, c, N, 1e-15, xx, yy, ii, jj)
phi = phi.reshape(phase.shape)
#phi = -phi
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(phase, cmap = 'bone')
axarr[0,1].imshow(phi.reshape(phase.shape), cmap = 'bone')
axarr[1,0].imshow(wrap_function(phase), cmap = 'bone')
axarr[1,1].imshow(wrap_function(phi.reshape(phase.shape)), cmap='bone')
plt.show()
