import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftp
import scipy.sparse as sparse
import scipy.linalg as lin

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
phase[100:300, 100:300] = 3 * (np.random.rand(200,200) - 0.5)
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
M = 10
N = M - 2
Bdiag = -4 * sparse.eye(N**2)
Bupper = sparse.eye(N**2, k = 1)
Blower = sparse.eye(N**2, k = -1)
Buppupp = sparse.eye(N**2, k = N)
Blowlow = sparse.eye(N**2, k = -N)
A = Bdiag + Bupper + Blower + Buppupp + Blowlow
A /= N**2

test_phase = np.arange(N**2).reshape(N,N)
wr_test_phase = wrap_function(test_phase)
W = np.ones((N,N))
W[2:4, 2:4] = 0
rho_test_phase = rho_ij(wr_test_phase, weighted = True, W = W)
c = rho_test_phase.flatten()
indices_straight = np.arange(N**2)
indices_matrix = np.unravel_index(indices_straight, test_phase.shape)
assert (np.all(c[indices_straight] == rho_test_phase[indices_matrix]))
assert (np.all(rho_test_phase == c.reshape(rho_test_phase.shape)))



indices_0 = np.where(W==0)
W_sparse = sparse.identity(N**2, format = 'lil')
indices_1 = np.ravel_multi_index(indices_0, W.shape)
indices_weight = ((indices_1, indices_1))
W_sparse[indices_weight] = 0
A.tocsr()
W_sparse.tocsr()
A_t = A.transpose()
W_t = W_sparse.transpose()
Q = A_t.dot(W_t.dot(W_sparse.dot(A)))
P = A_t.dot(A)

## precompute static matrices
yshape, xshape = list(rho_test_phase.shape)
x, y = np.linspace(-1, 1, xshape), np.linspace(-1, 1, yshape)
xx, yy = np.meshgrid(x,y)
i, j = np.linspace(0, len(x)-1, len(x)), np.linspace(0, len(y)-1, len(y))
ii, jj = np.meshgrid(i, j)
M, N = len(x), len(y)


#### solve using cg method
k = 0
phi = np.zeros(N**2)
rho_test_phase = rho_ij(wr_test_phase, weighted = True, W = W)
c = rho_test_phase.flatten()
c = np.transpose(c)
r_0 = c
r_new = c
r_old = c
while k <= N**2:
    r_new_ravel = r_new.reshape(rho_test_phase.shape)
    z_new = unwrap_phase_dct(r_new_ravel, xx, yy, ii, jj, M, N).flatten()
    k += 1
    if k == 1:
        p = z_new
        rz_dot_new = np.transpose(r_new).dot(z_new)
        beta = 1
    else:
        rz_dot_new = np.transpose(r_new).dot(z_new)
        rz_dot_old = (np.transpose(r_old).dot(z_old))
        beta =  rz_dot_new/rz_dot_old 
        p = z_new + beta * p
    alpha = rz_dot_new / (np.transpose(p).dot(Q.dot(p)))
    phi += alpha * p
    r_old = r_new
    r_new = r_old -  alpha * Q.dot(p)
    print "i = " + str(k) + "  beta = " + str(beta) + "  res = " + str(np.linalg.norm(r_new)/np.linalg.norm(r_0))
    if np.linalg.norm(r_new) <= 1e-3 * np.linalg.norm(r_0):
        break
    z_old = z_new
    

plt.spy(W_sparse)
plt.show()
