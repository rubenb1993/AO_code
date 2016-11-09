import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin

def wrap_function(phase):
    number = np.floor((np.abs(phase) - np.pi)/ (2* np.pi)) + 1 #number of times 2pi has to be substracted/added in order to be in [-pi, pi)
    phase_wrap = (phase - (np.sign(phase) * number * 2 * np.pi))
    return phase_wrap

def delta_x(wrapped_phase, **kwargs):
    if 'weighted' in kwargs:
        W_min = np.minimum(W[:, 1:]**2, W[:, :wrapped_phase.shape[1]-1]**2)
        delta_x = W_min * (wrapped_phase[:,1:] - wrapped_phase[:,:wrapped_phase.shape[1]-1])
        zero_padding = np.zeros(wrapped_phase.shape)
        zero_padding[:,:-1] = delta_x
        delta_x_wrapped = wrap_function(zero_padding)
    else:
        delta_x = (wrapped_phase[:,1:] - wrapped_phase[:,:wrapped_phase.shape[1]-1])
        zero_padding = np.zeros(wrapped_phase.shape)
        zero_padding[:,:-1] = delta_x
        delta_x_wrapped = wrap_function(zero_padding)
    return delta_x_wrapped

def delta_y(wrapped_phase, **kwargs):
    if 'weighted' in kwargs:
        W_min = np.minimum(W[1:, :]**2, W[:wrapped_phase.shape[0]-1, :])
        delta_y = W_min * (wrapped_phase[1:,:] - wrapped_phase[:wrapped_phase.shape[0]-1,:])
        zero_padding = np.zeros(wrapped_phase.shape)
        zero_padding[:-1,:] = delta_y
        delta_y_wrapped = wrap_function(zero_padding)
    else:
        delta_y = (wrapped_phase[1:,:] - wrapped_phase[:wrapped_phase.shape[0]-1,:])
        zero_padding = np.zeros(wrapped_phase.shape)
        zero_padding[:-1,:] = delta_y
        delta_y_wrapped = wrap_function(zero_padding)
    return delta_y_wrapped

def rho_ij(wrapped_phase, **kwargs):
    if 'test' in kwargs:
        shapes = list(wrapped_phase.shape)
        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))
        zero_xx[:, 1:] = delta_x(wrapped_phase, **kwargs)
        zero_xx[:, -1] = 0
        zero_yy[1:, :] = delta_y(wrapped_phase, **kwargs)
        zero_yy[-1, :] = 0
        rho_ij = (zero_xx[:, 1:] - zero_xx[:, :-1]) + (zero_yy[1:,:] - zero_yy[:-1, :])
    else:
        shapes = list(wrapped_phase.shape)
        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))
        zero_xx[:, 1:] = delta_x(wrapped_phase, **kwargs)
        zero_yy[1:, :] = delta_y(wrapped_phase, **kwargs)
        rho_ij = (zero_xx[:, 1:] - zero_xx[:, :-1]) + (zero_yy[1:,:] - zero_yy[:-1, :])
    return rho_ij

def unwrap_phase_dct(wrapped_phase, xx, yy, ii, jj, M, N, **kwargs):
##    yshape, xshape = list(wrapped_phase.shape)
##    x, y = np.linspace(-1, 1, xshape), np.linspace(-1, 1, yshape)
##    xx, yy = np.meshgrid(x,y)
##    i, j = np.linspace(0, len(x)-1, len(x)), np.linspace(0, len(y)-1, len(y))
##    ii, jj = np.meshgrid(i, j)
##    M, N = len(x), len(y)

    rho_ij_hat = fftp.dct(fftp.dct(rho_ij(wrapped_phase, **kwargs), axis = 0), axis = 1)
    denom = 2 * (np.cos(np.pi * ii / M) + np.cos(np.pi * jj / N) - 2)
    denom[0,0] = 1.0
    phi_ij_hat = rho_ij_hat/denom
    unwr_phase = 1.0/((4 * M*N)) * fftp.dct(fftp.dct(phi_ij_hat, axis = 0, type = 3), axis = 1, type = 3)
    return unwr_phase

x, y = np.linspace(-1, 1, 512), np.linspace(-1, 1, 512)
xx, yy = np.meshgrid(x, y)
phase = 23 * xx + 25 * yy
#phase[10:30, 10:30] = 3 * (np.random.rand(20,20) - 0.5)
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


#### make matrix for solving 2D weighted wrapped phase
#N = N-2
Bdiag = -4 * sparse.eye(N**2)
Bupper = sparse.eye(N**2, k = 1)
Blower = sparse.eye(N**2, k = -1)
Buppupp = sparse.eye(N**2, k = N)
Blowlow = sparse.eye(N**2, k = -N)
A = Bdiag + Bupper + Blower + Buppupp + Blowlow

def ravel_indices(shape, *args):
    """given a certain range of indices which are labeled inside, return an arbitrary number of vectors filtered with those indices"""
    new_positions = []
    for arg in args:
        new_positions.append(np.ravel_multi_index(arg, shape))
    return new_positions

indices_left = (np.arange(1,N-1), [0]*len(np.arange(1,N-1)))
#indices_left_ravel = np.ravel_multi_index(indices_left, (N, N))

indices_right = (np.arange(1,N-1), [N-1]*len(np.arange(1, N-1)))
#indices_right_ravel = np.ravel_multi_index(indices_right, (N, N))

indices_top = ([0]*len(np.arange(1, N-1)) , np.arange(1, N-1))
#indices_top_ravel = np.ravel_multi_index(indices_top, (N, N))

indices_bot = ([N-1]*len(np.arange(1, N-1)), np.arange(1, N-1))
#indices_bot_ravel = np.ravel_multi_index(indices_bot, (N, N))

indices_left_ravel, indices_right_ravel, indices_top_ravel, indices_bot_ravel = ravel_indices((N, N), indices_left, indices_right, indices_top, indices_bot)

A = A.tolil()
A[indices_left_ravel, indices_left_ravel] *= 0.5
A[indices_left_ravel, indices_left_ravel + N] *= 0.5
A[indices_left_ravel, indices_left_ravel - N] *= 0.5
A[indices_left_ravel, indices_left_ravel-1] = 0.0

A[indices_right_ravel, indices_right_ravel] *= 0.5
A[indices_right_ravel, indices_right_ravel + 1] = 0.0
A[indices_right_ravel, indices_right_ravel + N] *= 0.5
A[indices_right_ravel, indices_right_ravel - N] *= 0.5

A[indices_top_ravel, indices_top_ravel] *= 0.5
A[indices_top_ravel, indices_top_ravel + 1] *= 0.5
A[indices_top_ravel, indices_top_ravel - 1] *= 0.5

A[indices_bot_ravel, indices_bot_ravel] *= 0.5
A[indices_bot_ravel, indices_bot_ravel + 1] *= 0.5
A[indices_bot_ravel, indices_bot_ravel - 1] *= 0.5


index_top_left = ([0],[0])
index_top_right = ([0], [N-1])
index_bot_left = ([N-1], [0])
index_bot_right = ([N-1], [N-1])

top_left_ravel, top_right_ravel, bot_left_ravel, bot_right_ravel = ravel_indices((N, N), index_top_left, index_top_right, index_bot_left, index_bot_right)

A[top_left_ravel, top_left_ravel] *= 0.25
A[top_left_ravel, top_left_ravel +1] *= 0.5
A[top_left_ravel, top_left_ravel +N] *= 0.5

A[top_right_ravel, top_right_ravel] *= 0.25
A[top_right_ravel, top_right_ravel - 1] *= 0.5
A[top_right_ravel, top_right_ravel + N] *= 0.5
A[top_right_ravel, top_right_ravel + 1] *= 0.0

A[bot_left_ravel, bot_left_ravel] *= 0.25
A[bot_left_ravel, bot_left_ravel + 1] *= 0.5
A[bot_left_ravel, bot_left_ravel - 1] *= 0.0
A[bot_left_ravel, bot_left_ravel - N] *= 0.5

A[bot_right_ravel, bot_right_ravel - 1] *= 0.5
A[bot_right_ravel, bot_right_ravel - N] *= 0.5
A[bot_right_ravel, bot_right_ravel] *= 0.25

A /= N**2

##test_phase = np.arange(N**2).reshape(N,N)
##wr_test_phase = wrap_function(test_phase)
W = np.ones((N,N))
W[10:30, 10:30] = 0
Weight = W
indices_0 = np.where(W==0)
Weight[indices_0] = 1
rho_test_phase = rho_ij(wr_phase, test = True)#, weighted = True, W = Weight)
c = rho_test_phase.flatten()
c[top_left_ravel] *= 0.25
c[top_right_ravel] *= 0.25
c[bot_left_ravel] *= 0.25
c[bot_right_ravel] *= 0.25
c[indices_top_ravel] *= 0.5
c[indices_bot_ravel] *= 0.5
c[indices_left_ravel] *= 0.5
c[indices_right_ravel] *= 0.5

indices_straight = np.arange(N**2)
indices_matrix = np.unravel_index(indices_straight, phase.shape)
#assert (np.all(c[indices_straight] == rho_test_phase[indices_matrix]))
#assert (np.all(rho_test_phase == c.reshape(rho_test_phase.shape)))
c = sparse.csr_matrix(c)


indices_0 = np.where(W==0)
W_sparse = sparse.identity(N**2, format = 'lil')
indices_1 = np.ravel_multi_index(indices_0, W.shape)
indices_weight = ((indices_1, indices_1))
W_sparse[indices_weight] = 1
A.tocsr()
W_sparse.tocsr()
A_t = A.transpose()
W_t = W_sparse.transpose()
Q = A_t.dot(W_t.dot(W_sparse.dot(A)))
P = A_t.dot(A)

## precompute static matrices
##yshape, xshape = list(rho_test_phase.shape)
##x, y = np.linspace(-1, 1, xshape), np.linspace(-1, 1, yshape)
##xx, yy = np.meshgrid(x,y)
##i, j = np.linspace(0, len(x)-1, len(x)), np.linspace(0, len(y)-1, len(y))
##ii, jj = np.meshgrid(i, j)
##M, N = len(x), len(y)


#### solve using cg method
#phi = lin.spsolve(A, c.transpose())
k = 0
phi = np.zeros(N**2)
phi = sparse.csr_matrix(phi)
phi = phi.transpose()
##rho_test_phase = rho_ij(wr_phase, weighted = True, W = W)
##c = rho_test_phase.flatten()
##c = np.transpose(c)
r_0 = c
r_new = c.transpose()
r_old = c.transpose()
while k <= N**2:
    #r_new_ravel = r_new.reshape(rho_test_phase.shape)
    z_new = r_new
    #z_new = lin.spsolve(P, r_new)#unwrap_phase_dct(r_new_ravel, xx, yy, ii, jj, M, M).flatten()
    #z_new = sparse.csr_matrix(z_new)
    k += 1
    if k == 1:
        p = z_new
        #p = p.transpose()
        rz_dot_new = (z_new.transpose()).dot(r_new)
        #print str(rz_dot_new)
        beta = 1
    else:
        rz_dot_new = (z_new.transpose()).dot(r_new)
        #print str(rz_dot_new)
        rz_dot_old = (z_old.transpose()).dot(r_old)
        beta =  (rz_dot_new/rz_dot_old).item(0)
        p = z_new + beta * p
        p = p
    Qp = A.dot(p)
    alpha = (rz_dot_new / ((p.transpose()).dot(Qp))).item(0)
    phi += alpha * p
    r_old = r_new
    r_new = r_old -  alpha * Qp
    if k%20 == 0:
        print "i = " + str(k) + "  beta = " + str(beta) + "  res = " + str(lin.norm(r_new)/lin.norm(r_0))
    if lin.norm(r_new)/lin.norm(r_0) <= 1e-8:
        break
    z_old = z_new
    
phi = phi.toarray().reshape(phase.shape)/ (0.5)* N**2)
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(phase, cmap = 'bone')
axarr[0,1].imshow(phi.reshape(phase.shape), cmap = 'bone')
axarr[1,0].imshow(wrap_function(phase))
axarr[1,1].imshow(wrap_function(phi.reshape(phase.shape)))
plt.show()
