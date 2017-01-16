import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftp
import scipy.sparse as sparse
import scipy.sparse.linalg as lin

def wrap_function(phase, **kwargs):
    """wraps function s.t. phase = 0 is zero, and bounded between -pi and pi.
    include original in kwargs to have the old version of wrapping"""
    if 'original' in kwargs:
        number = np.floor((np.abs(phase) - np.pi)/ (2* np.pi)) + 1 #number of times 2pi has to be substracted/added in order to be in [-pi, pi)
        phase_wrap = (phase - (np.sign(phase) * number * 2 * np.pi))
    else:
        phase_wrap = np.mod(phase - np.pi, 2*np.pi) - np.pi
    return phase_wrap

##def wrap_function(phase):
##    phase_wrap = np.mod(phase - np.pi, 2*np.pi) - np.pi
##    return phase_wrap

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
        W = kwargs['W']
        shapes = list(W.shape)
        zero_weight_x = np.ones((shapes[0], shapes[1]+2))
        zero_weight_y = np.ones((shapes[0] + 2, shapes[1]))
        zero_xx = np.zeros((shapes[0],shapes[1]+1))
        zero_yy = np.zeros((shapes[0]+1, shapes[1]))

        zero_weight_x[:, 1:shapes[1]+1] = W
        zero_weight_y[1:shapes[0]+1] = W
        
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
##    indices_under_top = ([1]*len(np.arange(1, N-1)) , np.arange(1, N-1))
##    indices_bes_left = (np.arange(1,N-1), [1]*len(np.arange(1,N-1)))
##    indices_bes_right = (np.arange(1,N-1), [N-2]*len(np.arange(1, N-1)))
##    indices_over_bot = ([N-2]*len(np.arange(1, N-1)), np.arange(1, N-1))



    index_top_left = ([0],[0])
    index_top_right = ([0], [N-1])
    index_bot_left = ([N-1], [0])
    index_bot_right = ([N-1], [N-1])

    ## Ensure boundary conditions
##    Weight[:30, :] = 1
##    Weight[N-30:N, :] = 1
##    Weight[:, :30] = 1
##    Weight[:, N-30:N] = 1
##    Weight[indices_left] = 1
##    Weight[indices_right] = 1
##    Weight[indices_top] = 1
##    Weight[indices_bot] = 1
##    Weight[index_top_left] = 1
##    Weight[index_top_right] = 1
##    Weight[index_bot_left] = 1
##    Weight[index_bot_right] = 1
##    Weight[indices_under_top] = 1
##    Weight[indices_bes_left] = 1
##    Weight[indices_bes_right] = 1
##    Weight[indices_over_bot] = 1
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

def filter_wrapped_phase(image, k):
    ny, nx = image.shape
    assert(ny == nx) ## assert a square image for simplicity
    if (k%2 == 0):
        print("k  has to be an integer!")
        return
    N = nx
    i, j = np.arange(N), np.arange(N)
    ii, jj = np.meshgrid(i, j)
    filt_psi = np.zeros((N,N))

    inside = (jj[k/2:N-(k/2), k/2:N-(k/2)].flatten(), ii[k/2:N-(k/2), k/2:N-(k/2)].flatten())
    krange = np.linspace(-1 * (k/2), (k/2), k, dtype = 'int64') ## amount of added spaces, if k = 5, it ranges from -2 to 2
    krange_tile = np.tile(krange * N, (k, 1)).T ## tile them to make a (k/2)**2 matrix, containing for instance -2N, -N, 0, N, 2N for k=5
    k_tile = np.tile(krange, (k, 1)) ## tile to add to krange_tile
    coords_add = (krange_tile + k_tile).flatten() ## all coordinates, in a (k/2)**2 matrix, from -2N - 2: -2N + 2, -N-2 : -N+2 , -2 : 2, N -2 : N +2, 2N -2 : 2N +2
    inside = np.ravel_multi_index(inside, (N, N))
    coords_add = np.tile(coords_add, (len(inside), 1)) ## stack all differences to add to inside
    inside_tile = np.tile(inside, (coords_add.shape[1],1)).T ## stack all inside to add to differences
    all_coords = inside_tile + coords_add### a matrix of len(inside) x (k/2)**2 with all coordinates in a k x k square around a certain coordinate
    unrav_coords = np.unravel_index(all_coords, (N, N)) ## unraveled coordinates of all coordinates
    sum_sin_psi = np.sum(np.sin(image[unrav_coords]), axis = 1) ## sum over a sin (psi) over a k x k square
    sum_cos_psi = np.sum(np.cos(image[unrav_coords]), axis = 1) ## sum over a cos (psi) over a k x k square
    psi_app = np.arctan2(sum_sin_psi, sum_cos_psi)
    filt_psi[np.unravel_index(inside, (N, N))] = psi_app 

    #### top layers
    for i in range(k/2):
        ## for indices directly above the "inside square"
        top = (jj[i, k/2:N-(k/2)].flatten(), ii[i, k/2: N - (k/2)].flatten())
        coords_add = (krange_tile + k_tile)[(k/2)-i:, :].flatten()
        top = np.ravel_multi_index(top, (N, N))
        coords_add = np.tile(coords_add, (len(top), 1))
        top_tile = np.tile(top, (coords_add.shape[1],1)).T
        top_coords = top_tile + coords_add
        unrav_coords = np.unravel_index(top_coords, (N, N))
        sum_sin_top = np.sum(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_top = np.sum(np.cos(image[unrav_coords]), axis = 1)
        psi_top = np.arctan2(sum_sin_top, sum_cos_top)
        filt_psi[np.unravel_index(top, (N, N))] = psi_top

        ## indices directly below the "inside square"
        bot = (jj[N- 1 - i, k/2:N-(k/2)].flatten(), ii[N-1-i, k/2: N - (k/2)].flatten()) ## starting at the bottom working inwards
        coords_add = (krange_tile + k_tile)[:(k/2) + 1 + i, :].flatten()
        bot = np.ravel_multi_index(bot, (N, N))
        coords_add = np.tile(coords_add, (len(top), 1))
        bot_tile = np.tile(bot, (coords_add.shape[1],1)).T
        bot_coords = bot_tile + coords_add
        unrav_coords = np.unravel_index(bot_coords, (N, N))
        sum_sin_bot = np.sum(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_bot = np.sum(np.cos(image[unrav_coords]), axis = 1)
        psi_bot = np.arctan2(sum_sin_bot, sum_cos_bot)
        filt_psi[np.unravel_index(bot, (N, N))] = psi_bot

        ## indices directly left of the "inside square"
        left = (jj[k/2:N-(k/2), i].flatten(), ii[k/2:N-(k/2), i].flatten()) ## starting at the bottom working inwards
        coords_add = (krange_tile + k_tile)[:, (k/2)-i:].flatten()
        left = np.ravel_multi_index(left, (N, N))
        coords_add = np.tile(coords_add, (len(left), 1))
        left_tile = np.tile(left, (coords_add.shape[1],1)).T
        left_coords = left_tile + coords_add
        unrav_coords = np.unravel_index(left_coords, (N, N))
        sum_sin_left = np.sum(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_left = np.sum(np.cos(image[unrav_coords]), axis = 1)
        psi_left = np.arctan2(sum_sin_left, sum_cos_left)
        filt_psi[np.unravel_index(left, (N, N))] = psi_left

        ## indices directly left of the "inside square"
        right = (jj[k/2:N-(k/2), N - 1 - i].flatten(), ii[k/2:N-(k/2), N - 1 - i].flatten()) ## starting at the bottom working inwards
        coords_add = (krange_tile + k_tile)[:, :(k/2)+1+i].flatten()
        right = np.ravel_multi_index(right, (N, N))
        coords_add = np.tile(coords_add, (len(right), 1))
        right_tile = np.tile(right, (coords_add.shape[1],1)).T
        right_coords = right_tile + coords_add
        unrav_coords = np.unravel_index(right_coords, (N, N))
        sum_sin_right = np.sum(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_right = np.sum(np.cos(image[unrav_coords]), axis = 1)
        psi_right = np.arctan2(sum_sin_right, sum_cos_right)
        filt_psi[np.unravel_index(right, (N, N))] = psi_right
        
        ## calculate boundaries diagonals
        left_t, right_t, left_b, right_b = (i, i), (i, -1 -i), (-1 - i, i), (-1 - i, -1 - i)       
        left_t, right_t, left_b, right_b = (jj[left_t], ii[left_t]), (jj[right_t], ii[right_t]), (jj[left_b], ii[left_b]), (jj[right_b], ii[right_b])
        left_t, right_t, left_b, right_b = np.ravel_multi_index(left_t, (N, N)), np.ravel_multi_index(right_t, (N, N)), np.ravel_multi_index(left_b, (N, N)), np.ravel_multi_index(right_b, (N, N))
        coord_mat = krange_tile + k_tile
        coords_add_lt, coords_add_rt, coords_add_lb, coords_add_rb = coord_mat[(k/2)-i:, (k/2)-i:].flatten(), coord_mat[(k/2)-i:, :(k/2)+1+i].flatten(), coord_mat[:(k/2)+i+1, (k/2)-i:].flatten(), coord_mat[:(k/2)+i+1, :(k/2)+i+1].flatten()
        coords_add_tot = np.vstack((coords_add_lt, coords_add_rt, coords_add_lb, coords_add_rb))
        lt_tile, rt_tile, lb_tile, rb_tile = np.tile(left_t, (coords_add_lt.shape[0],1)).T, np.tile(right_t, (coords_add_lt.shape[0],1)).T, np.tile(left_b, (coords_add_lt.shape[0],1)).T, np.tile(right_b, (coords_add_lt.shape[0],1)).T
        coords_tile_tot = np.squeeze(np.stack((lt_tile, rt_tile, lb_tile, rb_tile)))
        coords_tot = coords_add_tot + coords_tile_tot
        unrav_coords = np.unravel_index(coords_tot, (N, N))
        sum_sin_diag = np.sum(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_diag = np.sum(np.cos(image[unrav_coords]), axis = 1)
        psi_diag = np.arctan(sum_sin_diag, sum_cos_diag)
        filt_psi[np.unravel_index(np.stack((left_t, right_t, left_b, right_b)), (N, N))] = psi_diag

    return filt_psi


def butter_filter(image, n, f0):
    """Filters the image in the fourier space, using a butter filter of order n and cut-off frequency f0
    It is adviced to pad the image before taking using this funciton, and cut out the result accordingly
    dx = 2/nx because the original image is defined on the circle with radius 1. Hence dfx = 0.5"""
    [ny, nx] = image.shape
    dx = 2.0/nx
    dy = 2.0/nx
    dfx = 0.5
    dfy = 0.5
    fx = np.arange(-0.5/dx, 0.5/dx, dfx)
    fy = np.arange(-0.5/dy, 0.5/dy, dfy)
    FX, FY = np.meshgrid(fx, fy)
    shift = np.exp(-2*np.pi*1j*(FX+FY))
    sin_pad = np.sin(image)
    cos_pad = np.cos(image)
    ft_sin = shift *  np.fft.fftshift(np.fft.fft2(sin_pad))
    ft_cos = shift * np.fft.fftshift(np.fft.fft2(cos_pad))
    butt_filt = 1/(1 + ( np.sqrt(FX**2 + FY**2)/f0)**(2*n))
    sin_filt = np.real(np.fft.ifftshift(np.fft.ifft2(butt_filt * ft_sin)) / shift)
    cos_filt = np.real(np.fft.ifftshift(np.fft.ifft2(butt_filt * ft_cos)) / shift)
    phase_filt = np.arctan2(sin_filt, cos_filt)
    return phase_filt


    
def phase_derivative_var_map(image, k):
    dx_phase = delta_x(image)
    dy_phase = delta_y(image)

    ny, nx = dx_phase.shape
    assert(ny == nx) ## assert a square image for simplicity
    if (k%2 == 0):
        print("k  has to be an uneven integer!")
        return
    N = nx
    i, j = np.arange(N), np.arange(N)
    ii, jj = np.meshgrid(i, j)
    zmn = np.zeros((N,N))
    
    

    inside = (jj[k/2:N-(k/2), k/2:N-(k/2)].flatten(), ii[k/2:N-(k/2), k/2:N-(k/2)].flatten())
    krange = np.linspace(-1 * (k/2), (k/2), k, dtype = 'int64') ## amount of added spaces, if k = 5, it ranges from -2 to 2
    krange_tile = np.tile(krange * N, (k, 1)).T ## tile them to make a (k/2)**2 matrix, containing for instance -2N, -N, 0, N, 2N for k=5
    k_tile = np.tile(krange, (k, 1)) ## tile to add to krange_tile
    coords_add = (krange_tile + k_tile).flatten() ## all coordinates, in a (k/2)**2 matrix, from -2N - 2: -2N + 2, -N-2 : -N+2 , -2 : 2, N -2 : N +2, 2N -2 : 2N +2
    inside = np.ravel_multi_index(inside, (N, N))
    coords_add = np.tile(coords_add, (len(inside), 1)) ## stack all differences to add to inside
    inside_tile = np.tile(inside, (coords_add.shape[1],1)).T ## stack all inside to add to differences
    all_coords = inside_tile + coords_add### a matrix of len(inside) x (k/2)**2 with all coordinates in a k x k square around a certain coordinate
    unrav_coords = np.unravel_index(all_coords, (N, N)) ## unraveled coordinates of all coordinates
    
    avg_x, avg_y = np.sum(dx_phase[unrav_coords], axis = 1)/k**2, np.sum(dy_phase[unrav_coords], axis = 1)/k**2
    avg_x_tile, avg_y_tile = np.tile(avg_x, (all_coords.shape[1], 1)).T, np.tile(avg_y, (all_coords.shape[1], 1)).T
    sum_x, sum_y = np.sum(np.square(dx_phase[unrav_coords] - avg_x_tile), axis = 1), np.sum(np.square(dy_phase[unrav_coords] - avg_y_tile), axis = 1)
    zmn[np.unravel_index(inside, (N, N))] = (np.sqrt(sum_x) + np.sqrt(sum_y)) / (k**2)



    #### top layers
    for i in range(k/2):
        ## for indices directly above the "inside square"
        top = (jj[i, k/2:N-(k/2)].flatten(), ii[i, k/2: N - (k/2)].flatten())
        coords_add = (krange_tile + k_tile)[(k/2)-i:, :].flatten()
        top = np.ravel_multi_index(top, (N, N))
        coords_add = np.tile(coords_add, (len(top), 1))
        top_tile = np.tile(top, (coords_add.shape[1],1)).T
        top_coords = top_tile + coords_add
        unrav_coords = np.unravel_index(top_coords, (N, N))
        avg_x, avg_y = np.sum(dx_phase[unrav_coords], axis = 1)/k**2, np.sum(dy_phase[unrav_coords], axis = 1)/k**2
        avg_x_tile, avg_y_tile = np.tile(avg_x, (top_coords.shape[1], 1)).T, np.tile(avg_y, (top_coords.shape[1], 1)).T
        sum_x, sum_y = np.sum(np.square(dx_phase[unrav_coords] - avg_x_tile), axis = 1), np.sum(np.square(dy_phase[unrav_coords] - avg_y_tile), axis = 1)
        zmn[np.unravel_index(top, (N, N))] = (np.sqrt(sum_x) + np.sqrt(sum_y)) / (k**2)
##        sum_sin_top = np.sum(np.sin(image[unrav_coords]), axis = 1)
##        sum_cos_top = np.sum(np.cos(image[unrav_coords]), axis = 1)
##        psi_top = np.arctan2(sum_sin_top, sum_cos_top)
##        filt_psi[np.unravel_index(top, (N, N))] = psi_top

        ## indices directly below the "inside square"
        bot = (jj[N- 1 - i, k/2:N-(k/2)].flatten(), ii[N-1-i, k/2: N - (k/2)].flatten()) ## starting at the bottom working inwards
        coords_add = (krange_tile + k_tile)[:(k/2) + 1 + i, :].flatten()
        bot = np.ravel_multi_index(bot, (N, N))
        coords_add = np.tile(coords_add, (len(top), 1))
        bot_tile = np.tile(bot, (coords_add.shape[1],1)).T
        bot_coords = bot_tile + coords_add
        unrav_coords = np.unravel_index(bot_coords, (N, N))
        avg_x, avg_y = np.sum(dx_phase[unrav_coords], axis = 1)/k**2, np.sum(dy_phase[unrav_coords], axis = 1)/k**2
        avg_x_tile, avg_y_tile = np.tile(avg_x, (bot_coords.shape[1], 1)).T, np.tile(avg_y, (bot_coords.shape[1], 1)).T
        sum_x, sum_y = np.sum(np.square(dx_phase[unrav_coords] - avg_x_tile), axis = 1), np.sum(np.square(dy_phase[unrav_coords] - avg_y_tile), axis = 1)
        zmn[np.unravel_index(bot, (N, N))] = (np.sqrt(sum_x) + np.sqrt(sum_y)) / (k**2)

        ## indices directly left of the "inside square"
        left = (jj[k/2:N-(k/2), i].flatten(), ii[k/2:N-(k/2), i].flatten()) ## starting at the bottom working inwards
        coords_add = (krange_tile + k_tile)[:, (k/2)-i:].flatten()
        left = np.ravel_multi_index(left, (N, N))
        coords_add = np.tile(coords_add, (len(left), 1))
        left_tile = np.tile(left, (coords_add.shape[1],1)).T
        left_coords = left_tile + coords_add
        unrav_coords = np.unravel_index(left_coords, (N, N))
        avg_x, avg_y = np.sum(dx_phase[unrav_coords], axis = 1)/k**2, np.sum(dy_phase[unrav_coords], axis = 1)/k**2
        avg_x_tile, avg_y_tile = np.tile(avg_x, (left_coords.shape[1], 1)).T, np.tile(avg_y, (left_coords.shape[1], 1)).T
        sum_x, sum_y = np.sum(np.square(dx_phase[unrav_coords] - avg_x_tile), axis = 1), np.sum(np.square(dy_phase[unrav_coords] - avg_y_tile), axis = 1)
        zmn[np.unravel_index(left, (N, N))] = (np.sqrt(sum_x) + np.sqrt(sum_y)) / (k**2)

        ## indices directly left of the "inside square"
        right = (jj[k/2:N-(k/2), N - 1 - i].flatten(), ii[k/2:N-(k/2), N - 1 - i].flatten()) ## starting at the bottom working inwards
        coords_add = (krange_tile + k_tile)[:, :(k/2)+1+i].flatten()
        right = np.ravel_multi_index(right, (N, N))
        coords_add = np.tile(coords_add, (len(right), 1))
        right_tile = np.tile(right, (coords_add.shape[1],1)).T
        right_coords = right_tile + coords_add
        unrav_coords = np.unravel_index(right_coords, (N, N))
        avg_x, avg_y = np.sum(dx_phase[unrav_coords], axis = 1)/k**2, np.sum(dy_phase[unrav_coords], axis = 1)/k**2
        avg_x_tile, avg_y_tile = np.tile(avg_x, (right_coords.shape[1], 1)).T, np.tile(avg_y, (right_coords.shape[1], 1)).T
        sum_x, sum_y = np.sum(np.square(dx_phase[unrav_coords] - avg_x_tile), axis = 1), np.sum(np.square(dy_phase[unrav_coords] - avg_y_tile), axis = 1)
        zmn[np.unravel_index(right, (N, N))] = (np.sqrt(sum_x) + np.sqrt(sum_y)) / (k**2)
##        
##        ## calculate boundaries diagonals
##        left_t, right_t, left_b, right_b = (i, i), (i, -1 -i), (-1 - i, i), (-1 - i, -1 - i)       
##        left_t, right_t, left_b, right_b = (jj[left_t], ii[left_t]), (jj[right_t], ii[right_t]), (jj[left_b], ii[left_b]), (jj[right_b], ii[right_b])
##        left_t, right_t, left_b, right_b = np.ravel_multi_index(left_t, (N, N)), np.ravel_multi_index(right_t, (N, N)), np.ravel_multi_index(left_b, (N, N)), np.ravel_multi_index(right_b, (N, N))
##        coord_mat = krange_tile + k_tile
##        coords_add_lt, coords_add_rt, coords_add_lb, coords_add_rb = coord_mat[(k/2)-i:, (k/2)-i:].flatten(), coord_mat[(k/2)-i:, :(k/2)+1+i].flatten(), coord_mat[:(k/2)+i+1, (k/2)-i:].flatten(), coord_mat[:(k/2)+i+1, :(k/2)+i+1].flatten()
##        coords_add_tot = np.vstack((coords_add_lt, coords_add_rt, coords_add_lb, coords_add_rb))
##        lt_tile, rt_tile, lb_tile, rb_tile = np.tile(left_t, (coords_add_lt.shape[0],1)).T, np.tile(right_t, (coords_add_lt.shape[0],1)).T, np.tile(left_b, (coords_add_lt.shape[0],1)).T, np.tile(right_b, (coords_add_lt.shape[0],1)).T
##        coords_tile_tot = np.squeeze(np.stack((lt_tile, rt_tile, lb_tile, rb_tile)))
##        coords_tot = coords_add_tot + coords_tile_tot
##        unrav_coords = np.unravel_index(coords_tot, (N, N))
##        sum_sin_diag = np.sum(np.sin(image[unrav_coords]), axis = 1)
##        sum_cos_diag = np.sum(np.cos(image[unrav_coords]), axis = 1)
##        psi_diag = np.arctan(sum_sin_diag, sum_cos_diag)
##        filt_psi[np.unravel_index(np.stack((left_t, right_t, left_b, right_b)), (N, N))] = psi_diag


    return zmn



####### set up physical constants and a test phase
##x, y = np.linspace(-1, 1, 512), np.linspace(-1, 1, 512)
##xx, yy = np.meshgrid(x, y)
##phase =  10 * ((xx + 0.1)**2 + (yy - 0.1)**2)
##wr_phase = wrap_function(phase)
###wr_phase += 0.6 * (np.random.rand(xx.shape[0], yy.shape[1]) - 0.5)
##
##yshape, xshape = list(wr_phase.shape)
##x, y = np.linspace(-1, 1, xshape), np.linspace(-1, 1, yshape)
##xx, yy = np.meshgrid(x,y)
##i, j = np.linspace(0, len(x)-1, len(x)), np.linspace(0, len(y)-1, len(y))
##ii, jj = np.meshgrid(i, j)
##M, N = len(x), len(y)
##
########## DCT unwrapping
##unwr_phase = unwrap_phase_dct(wr_phase, xx, yy, ii, jj, M, N)
##
########## start plotting results
##f, axarr = plt.subplots(2,3)
##axarr[0,1].imshow(unwr_phase, cmap = 'bone', vmin = np.floor(np.min(phase)/10)*10, vmax = np.ceil(np.max(phase)/10)*10)
##axarr[0,1].set_title('DCT retrieved phase')
##axarr[0,0].imshow(phase, cmap = 'bone', vmin = np.floor(np.min(phase)/10)*10, vmax = np.ceil(np.max(phase)/10)*10)
##axarr[0,0].set_title('Original phase')
##axarr[1,0].imshow(wr_phase, cmap = 'bone', vmin = -4, vmax = 4)
##axarr[1,0].set_title('Original wr phase')
##axarr[1,1].imshow(wrap_function(unwr_phase), cmap = 'bone', vmin = -4, vmax = 4)
##axarr[1,1].set_title('DCT rewrapped phase')
##
########## Set up weight where there is noise
##Weight_zeros = np.ones((N,N))
####Weight_zeros[10:20, 10:-10] = 0
####Weight_zeros[50:60, 10:-10] = 0
####Weight_zeros[80:90, 10:-10] = 0
####Weight_zeros[200:210, 10:-10] = 0
####Weight_zeros[300:320, 10:-10] = 0
##Weight_zeros[10:100, 10:100] = 0
##Weight = Weight_zeros
##indices_0 = np.where(Weight_zeros==0)
##weight_factor = 0.0
##Weight[indices_0] = weight_factor
##phi = np.zeros((N,N))
##
###A, c, A_weighted, c_weighted = least_squares_matrix(wr_phase, Weight)
###phi = pcg_algorithm(A_weighted, c_weighted, N, 1e-12, xx, yy, ii, jj).reshape(phase.shape)
##
##########plot rest of results
##im2 = axarr[1,2].imshow(wrap_function(phi), cmap='bone', vmin = -4, vmax = 4)
##axarr[1,2].set_title('PCG wrapped phase')
##im =axarr[0,2].imshow(phi, cmap='bone', vmin = np.floor(np.min(phase)/10)*10, vmax = np.ceil(np.max(phase)/10)*10)
##axarr[0,2].set_title('PCG retrieved phase')
##f.subplots_adjust(right=0.8)
##cbar_ax1 = f.add_axes([0.85, 0.55, 0.05, 0.35])
##cbar_ax2 = f.add_axes([0.85, 0.10, 0.05, 0.35])
##f.colorbar(im, cax=cbar_ax1)
##f.colorbar(im2, cax=cbar_ax2)
##for i in range(6):
##    j = np.unravel_index(i, axarr.shape)
##    axarr[j].get_xaxis().set_ticks([])
##    axarr[j].get_yaxis().set_ticks([])
##
##
##### filter_test
####xx = np.arange(10000).reshape(100, 100) / 100.0
####phase = 13 * xx
##unfilter_phase = phase #+ 0.5 * (np.random.rand(xx.shape[0], xx.shape[1]) - 0.5)
##unfilter_phase[100:110, :] += 1 * (np.random.rand(10, xx.shape[1]) - 0.5)
##unfilter_phase[200:210, :] += 1 * (np.random.rand(10, xx.shape[1]) - 0.5)
##unfilter_phase[300:310, :] += 1 * (np.random.rand(10, xx.shape[1]) - 0.5)
##unfilter_phase[400:410, :] += 1 * (np.random.rand(10, xx.shape[1]) - 0.5)
##
##
##unfilter_phase = wrap_function(unfilter_phase)
##
##
##
##f, ax = plt.subplots(2,5)
##for kk in range(3):
##    filter_phase = filter_wrapped_phase(unfilter_phase, 3 + 2 * kk)
##    ax[0,2+kk].imshow(filter_phase, interpolation = 'none', cmap = 'bone')
##    ax[1,2+kk].plot(filter_phase[:, 340])
##    ax[0,2+kk].set_title(str(3 +2*kk) +' x '+ str(3 + 2*kk))
##
##ax[1,1].plot(unfilter_phase[:, 340])
##ax[0,0].imshow(wrap_function(phase), interpolation = 'none', cmap = 'bone')
##ax[0,0].set_title('original signal')
##ax[0,1].imshow(unfilter_phase, interpolation = 'none', cmap = 'bone')
##ax[0,1].set_title('random gaussian noise')
##
##ax[1,0].plot(wrap_function(phase)[:, 340])
##
##
##dx_phase, dy_phase = delta_x(unfilter_phase), delta_y(unfilter_phase)
##qual_map = phase_derivative_var_map(dx_phase, dy_phase, 5)
##
##f, ax = plt.subplots(1,2)
##ax[0].imshow(unfilter_phase, cmap = 'bone', vmin = -np.pi, vmax = np.pi)
##ax[1].imshow(qual_map, cmap = 'bone')
##plt.show()
