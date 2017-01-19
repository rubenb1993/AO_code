import numpy as np
import matplotlib.pyplot as plt
import phase_unwrapping_test as pw
import phase_extraction as PE
import scipy.ndimage as ndimage

def filter_wrapped_phase_med(image, k):
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
    sum_sin_psi = np.median(np.sin(image[unrav_coords]), axis = 1) ## sum over a sin (psi) over a k x k square
    sum_cos_psi = np.median(np.cos(image[unrav_coords]), axis = 1) ## sum over a cos (psi) over a k x k square
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
        sum_sin_top = np.median(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_top = np.median(np.cos(image[unrav_coords]), axis = 1)
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
        sum_sin_bot = np.median(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_bot = np.median(np.cos(image[unrav_coords]), axis = 1)
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
        sum_sin_left = np.median(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_left = np.median(np.cos(image[unrav_coords]), axis = 1)
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
        sum_sin_right = np.median(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_right = np.median(np.cos(image[unrav_coords]), axis = 1)
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
        sum_sin_diag = np.median(np.sin(image[unrav_coords]), axis = 1)
        sum_cos_diag = np.median(np.cos(image[unrav_coords]), axis = 1)
        psi_diag = np.arctan2(sum_sin_diag, sum_cos_diag)
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

## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 310
r_sh_px = 320
r_sh_m = r_sh_px * px_size_int
j_max= 30         # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px    

x0 = 550
y0 = 484
radius = int(310)

constants = px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius

x,y  = np.linspace(-1, 1, 500), np.linspace(-1, 1, 500)
xx, yy = np.meshgrid(x, y)
folder_name = "20161213_new_inters/"
phase, delta_i, sh_spots, image_i0, flat_wf = PE.phase_extraction(constants, folder_name = folder_name, show_id_hat = False, show_hough_peaks = False, min_height = 50)
#phase = np.load("20170112_1_defocus/org_phase.npy")
phase_inspect = phase[...,0]

## butter filter
[ny, nx] = phase_inspect.shape
res = [2**i for i in range(15)]
nx_pad = np.where( res > np.tile(nx, len(res)))
nx_pad = res[nx_pad[0][0]]
dif_x = (nx_pad - int(nx))/2
phase_filt = butter_filter(np.lib.pad(phase_inspect, dif_x, 'reflect'), 2, 15)
phase_filt = phase_filt[dif_x:nx_pad - dif_x, dif_x:nx_pad - dif_x]

phase_inspect_med = ndimage.median_filter(phase_inspect, 5)
phase_med_filt = butter_filter(np.lib.pad(phase_inspect_med, dif_x, 'reflect'),2, 15)
phase_med_filt = phase_med_filt[dif_x:nx_pad - dif_x, dif_x:nx_pad - dif_x]
#phase_avg = pw.filter_wrapped_phase(phase_inspect, 19)#np.load("20170113_test_phase/phase_avg.npy") 
#phase_med = filter_wrapped_phase_med(phase_inspect, 19)#np.load("20170113_test_phase/phase_med.npy") 



f, ax = plt.subplots(2,4)
ax[0,0].imshow(phase_inspect)
ax[0,0].set_title(r'Original')
ax[0,0].set_ylabel(r'Phase')
ax[1,0].set_ylabel(r'Cross section Phase')
ax[0,1].imshow(phase_inspect_med)
ax[0,2].set_title(r'Butter ($f_0 = 15, n = 2$)')
ax[0,2].imshow(phase_filt)
ax[0,1].set_title(r'$5 \times 5$ median $\hat{I}_D$')
ax[0,3].imshow(phase_med_filt)
ax[0,3].set_title(r'median $\hat{I}_D$ and Butter')

ax[1,0].plot(phase_inspect[160, :])
ax[1,1].plot(phase_inspect_med[160, :])
ax[1,2].plot(phase_filt[160, :])
ax[1,3].plot(phase_med_filt[160, :])


plt.show()
