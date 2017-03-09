import numpy as np
import phase_unwrapping_test as pw
import matplotlib.pyplot as plt
import scipy.ndimage as img
from matplotlib import rc
import Zernike as Zn
import PIL.Image 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize as opt
import phase_extraction as PE
import json #to write editable txt files


def rms_piston(piston, *args):
    """function for rms value to be minimized. args should contain j_max, a_filt, N, Z_mat, orig, mask in that order"""
    if args:
        j_max, a_filt, N, Z_mat, orig, mask, fliplr = args
    else:
        print("you should include the right arguments!")
        return
    orig = np.ma.array(orig, mask = mask)
    inter = np.ma.array(Zn.int_for_comp(j_max, a_filt, N, piston, Z_mat, fliplr), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms

def load_json(filepath):
    "returns the array saved in a text filed using json"
    with open(filepath) as json_data:
        data = json.load(json_data)
    return np.asarray(data)

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=True)

## plot making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)
gold = (1 + np.sqrt(5))/2
folder = "20170214_post_processing/"
orig_fold = "20170216_desired_vs_created/"
titles = [r'a)', r'b)', r'c)', r'd)', r'e)', r'f)']

#
#### Uncomment for wrapped phase plots
##nx = 480
##ny = 480
##x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
##xx, yy = np.meshgrid(x, y)
##i, j = np.arange(nx), np.arange(ny)
##ii, jj = np.meshgrid(i, j)
##N = nx
##phase = 15 * xx - 15 * yy
##phase_plt = np.copy(phase)
##phase += np.random.normal(0, 0.3, (nx, ny))
##wr_phase = pw.wrap_function(phase)
##
##wr_phase_mean = img.filters.uniform_filter(wr_phase, 3)
##wr_phase_spec = pw.filter_wrapped_phase(wr_phase, 3)
##
##unwr_phase_mean = pw.unwrap_phase_dct(wr_phase_mean, xx, yy, ii, jj, N, N)
##unwr_phase_spec = pw.unwrap_phase_dct(wr_phase_spec, xx, yy, ii, jj, N, N)
##
##f, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize =int_im_size)
##b_gold = np.pi/gold
##titles = [r'a)', r'b)', r'c)']
##for axes in ax:
##    axes.plot([-1,1], [np.pi, np.pi], 'k--', linewidth = 0.5)
##    axes.plot([-1, 1], [-np.pi, -np.pi], 'k--', linewidth = 0.5)
##    axes.set_yticks([-np.pi, np.pi])
##    axes.set_yticklabels([r'$-\pi$', r'$\pi$'])
##    axes.yaxis.labelpad = -5
##    axes.set_ylim([-np.pi-b_gold, np.pi+b_gold])
##    axes.set(adjustable = 'box-forced', aspect = 1./(np.pi+b_gold))
##    axes.tick_params(axis = 'both', which = 'major', labelsize = 7)
##
##for i in range(3):
##    ax[i].set_title(titles[i], fontsize = 8, loc = 'left')
##ax[0].set_xlabel(r'$x$')
##ax[0].set_ylabel(r'$\mathcal{W}(\varphi)$')
##ax[0].plot(x, wr_phase[:,N/2], 'o-', markersize= 2.5)
##ax[1].plot(x, wr_phase_mean[:,N/2], 'o-', markersize= 2.5)
##ax[2].plot(x, wr_phase_spec[:,N/2], 'o-', markersize= 2.5)
####f.savefig(folder + "wrapped_phase_filtering.png", dpi = dpi_num, bbox_inches = 'tight')
##f2, ax2 = plt.subplots(1,1, figsize = int_im_size_23)
##ax2.plot(x, phase_plt[:,N/2], label = 'orignal phase')
##ax2.plot(x, unwr_phase_mean[:,N/2], 'g-', label = 'uniform filter')
##ax2.plot(x, unwr_phase_spec[:,N/2], 'k', label = 'wrapped phase filter')
##ax2.set_xlabel(r'$x$')
##ax2.set_ylabel(r'$\varphi$')
##ax2.legend(fontsize = 6)
##ax2.set(adjustable = 'box-forced', aspect = 1070./(2620 * 15)) 
####f2.savefig(folder + "unwrapped_filt_phase.png", dpi = dpi_num, bbox_inches = 'tight')
##plt.show()

### uncomment for interferograms
##j_max = 8
##a_def = np.zeros(j_max)
##a_def[2] = -8.
##a_ast = np.zeros(j_max)
##a_ast[4] = 5
##a_com = np.zeros(j_max)
##a_com[6] = 4.
##
##a_stack = np.stack((a_def, a_ast, a_com)).T
##extent = (-1, 1, -1, 1)
##
##fig, axes = plt.subplots(2, 4, figsize =(4./3 * 4.98, 3.07))#, gridspec_kw = {'width_ratios':[4, 4, 4, 0.4]})
##for i in range(3):
##    #ax = fig.add_subplot('23' + str(i+4))
##    axes[0,i].axis('off')
##    axes[0,i+1].axis('off')
##    inters = Zn.imshow_interferogram(j_max, a_stack[...,i], N = 600, ax = axes[1,i], extent = extent)
##    ax3d = fig.add_subplot('24' + str(i+1), projection = '3d')
##    Zn.plot_zernike(j_max, a_stack[...,i], ax = ax3d)
##    ax3d.set_zticks([-4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi])
##    ax3d.zaxis.set_rotate_label(False)
##    ax3d.set_zlabel(r'$\varphi$', fontsize = 7, rotation = 0, labelpad = -10)
##    ax3d.set_zticklabels([r'$-4\pi$', r'$-2\pi$', r'$0$', r'$2\pi$', r'$4\pi$'])
##    ax3d.set_xlabel(r'$x$', fontsize = 7, labelpad = -10)
##    ax3d.set_xticks([-1, 0, 1])
##    ax3d.set_ylabel(r'$y$', fontsize = 7, labelpad = -10)
##    ax3d.set_yticks([-1, 0, 1])
##    ax3d.tick_params(axis = 'both', pad = -3, labelsize = 8)
##    ax3d.set_title(titles[i], loc = 'left', fontsize = 8)
##    axes[1,i].tick_params(axis='both', which='major', labelsize=6)
##    axes[1,i].set_xlabel(r'$x$', fontsize = 7, labelpad = -2)
##    axes[1,i].set_ylabel(r'$y$', fontsize = 7, labelpad = -3.5)
##    axes[1,i].set_xticks([-1, 0, 1])
##    axes[1,i].set_yticks([-1, 0, 1])
##    axes[1,i].set_title(titles[i+3], loc = 'left', fontsize = 8)
##    #ax3d.tick_params(axis = 'y', pad = -3)
###divider = make_axes_locatable(axes[1,2])
###cax = ax.append_axes("right", size="10%", pad=1)
##axes[1,3].axis('off')
##divider = make_axes_locatable(axes[1,3])
##cax = divider.append_axes("left", size="10%", pad= -0.1)
##cbar = plt.colorbar(inters, cax=cax)#, shrink=0.9, aspect = 10, panchor = (1.0, 0.5), pad = -10,use_gridspec = False)
##cbar.ax.tick_params(labelsize=7)
##cbar.ax.set_ylabel('Normalized Intensity', fontsize = 8)
##axes[1,3].set(adjustable = 'box-forced', aspect = 10)
##fig.savefig(folder + "phase_to_int.png", dpi = dpi_num, bbox_inches = 'tight')

##a_dict = np.load(orig_fold + "coeff_dictionary_set_center_r.npy").item()
##a_int = a_dict['coeff_inter']
##j_max = len(a_int)
##a_des = np.zeros(j_max)
##a_des[2] = 4.
##var_dict = np.load(orig_fold + "vars_dictionary_set_center_r.npy").item()
##orig = np.fliplr(np.array(PIL.Image.open(orig_fold + "orig_scale.tif")))
##piston = var_dict['pist_inter']
##[ny, nx] = orig.shape
##N = nx
##x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
##xx, yy = np.meshgrid(x, y)
##mask = [np.sqrt(xx**2 + yy**2) >=1]
##
##j = np.arange(2, j_max) #start at 2, end at j_max-1
##j_range = np.arange(2, j_max+2)
##power_mat = Zn.Zernike_power_mat(j_max+2)
##Z_mat = Zn.Zernike_xy(xx, yy, power_mat, j_range)
##fliplr = False
##orig = np.ma.array(Zn.int_for_comp(j_max, a_int, N, piston, Z_mat, fliplr), mask = mask)
##extent = (-1, 1, -1, 1)
##
##flipint = False
##### optimize piston
##piston_des = -1.9064375#opt.fmin(rms_piston, 0, args = (j_max, a_des, N, Z_mat, orig, mask, flipint), full_output = True)[:1]
##
##f, ax = plt.subplots(1,4, figsize = (0.875 * 4.98, 3.07),
##                     gridspec_kw = {'width_ratios' : [0.4, 4, 4, 6], 'height_ratios' : [4, 4, 4, 4]})
##Zn.imshow_interferogram(j_max, a_des, N, ax = ax[1], piston = piston_des, extent= extent, Z_mat = Z_mat)
##inter = Zn.imshow_interferogram(j_max, a_int, N, ax = ax[2], extent = extent, Z_mat = Z_mat)
##ax[3].plot(j_range,a_int, 'go')
##ax[3].plot((max(j_range)+1)*[0], 'k-', linewidth = 0.5)
##for i in range(1,3):
##    ax[i].tick_params(axis='both', which='major', labelsize=6)
##    ax[i].set_xlabel(r'$x$', fontsize = 7, labelpad = -1)
##    ax[i].set_ylabel(r'$y$', fontsize = 7, labelpad = -3)
##    ax[i].set_xticks([-1, 0, 1])
##    ax[i].set_yticks([-1, 0, 1])
##    ax[i].set_title(titles[i-1], loc = 'left', fontsize = 8)
##
##ax[3].tick_params(axis = 'both', which = 'major', labelsize = 6)
##ax[3].set_yticks([-2, 0, 2, 4])
##ax[3].set_xticks([4, 9, 14, 19, 24, 29])
##ydiff = a_int.max()-a_int.min()
##b = ydiff /gold
##ax[3].set_ylim(bottom = a_int.min()-b/2, top = a_int.max() + b/2)
##ax[3].set_xlim(left = 1, right = 31)
##ax[3].set_title(titles[2], loc = 'left', fontsize = 8)
##ax[3].set_xlabel(r'$j$', fontsize = 7, labelpad = -1)
##ax[3].set_ylabel(r'$a_j$', fontsize = 7, labelpad = -3.3)
##cbar = plt.colorbar(inter, cax=ax[0])#, shrink=0.9, aspect = 10, panchor = (1.0, 0.5), pad = -10,use_gridspec = False)
##ax[0].yaxis.set_ticks_position('left')
##cbar.ax.set_ylabel(r'Normalized Intensity', fontsize = 7)
##cbar.ax.tick_params(labelsize = 7)
##cbar.set_ticks([0, 0.5, 1])
##cbar.ax.yaxis.set_label_position('left')
##f.savefig(folder + "desired_vs_created.png", dpi = dpi_num, bbox_inches = 'tight')
##
##plt.show()

########### RMS VS K$$$$$$$$$$$$$$
## Given paramters for centroid gathering
##px_size_sh = 5.2e-6     # width of pixels
##px_size_int = 5.2e-6
##f_sh = 17.6e-3            # focal length
##r_int_px = 280
##r_sh_px = 340
##r_sh_m = r_sh_px * px_size_int
##j_max= 30         # maximum fringe order
##wavelength = 632e-9 #[m]
##box_len = 35.0 #half width of subaperture box in px
##x0 = 550
##y0 = 484
##radius = int(r_int_px)
##
##constants = np.stack((px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius))
##search_height, look_ahead = 22, 20
###org_phase, delta_i, sh_spots, image_i0, flat_wf = PE.phase_extraction(constants, folder_name = orig_fold, show_id_hat = False, show_hough_peaks = False, min_height = search_height, look_ahead = look_ahead)
##org_phase = np.load(orig_fold + "org_phase.npy")
##filtered_phase = np.load(orig_fold + "filtered_phase.npy")
##[nx, ny] = org_phase[...,0].shape
##
##x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
##i, j = np.linspace(0, nx-1, nx), np.linspace(0, ny-1, ny)
##ii, jj = np.meshgrid(i, j)
##N = nx
##xx, yy = np.meshgrid(x, y)
##mask = [np.sqrt(xx**2 + yy**2) >= 1]
##xy_inside = np.where(np.sqrt(xx**2 + yy**2) <= 1)
##k_I = np.arange(1, 27, 2)
##org_unwr = np.zeros(org_phase.shape)
##mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
##piston_mat, rms_mat_avg = np.zeros(len(k_I)), np.zeros(len(k_I))
##a_def_avg = np.zeros(len(k_I))
##orig = np.fliplr(np.array(PIL.Image.open(orig_fold + "orig_scale.tif")))
##
##j = np.arange(2, j_max) #start at 2, end at j_max-1
##j_range = np.arange(2, j_max+2)
##power_mat = Zn.Zernike_power_mat(j_max+2)
##Z_mat = Zn.Zernike_xy(xx, yy, power_mat, j_range)
##flipint = False
##
##x_in, y_in = xx[xy_inside], yy[xy_inside]
##Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in
##Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
##for i in range(len(j)):
##    Zernike_2d[:, i] = Zernike_3d[...,i].flatten()
##
##res = [2**i for i in range(15)]
##nx_pad = np.where( res > np.tile(nx, len(res)))
##nx_pad = res[nx_pad[0][0]]
##dif_x = (nx_pad - int(nx))/2
##f0 = np.logspace(0,2)
##a_def_butt = np.zeros(len(f0))
##rms_mat_butt = np.zeros(len(f0))
##n = 2
##butter_phase = np.zeros(org_phase.shape)
##butter_unwr = np.zeros(org_phase.shape)
##butter_phase_2 = np.zeros(org_phase.shape)
##butter_unwr_2 = np.zeros(org_phase.shape)
##
##### calculate rms due to butter filtering
####for jjj in range(len(f0)):
####    print("filtering f0 = " + str(f0[jjj]))
####    org_phase = np.load(orig_fold + "org_phase.npy")
####    org_phase_butt = np.load(orig_fold + "org_phase.npy")
####    delta_i = np.load(orig_fold + "delta_i.npy")
####    Un_sol = np.triu_indices(delta_i.shape[-1], 1)
####
####    for k in range(org_phase.shape[-1]):
####        org_pad = np.lib.pad(org_phase_butt[..., k], dif_x,'reflect')
####        butter_pad = pw.butter_filter(org_pad, n, f0[jjj])
####        butter_phase[..., k] = butter_pad[dif_x: nx_pad - dif_x, dif_x: nx_pad - dif_x]
####        butter_unwr[..., k] = pw.unwrap_phase_dct(butter_phase[..., k], xx, yy, ii, jj, N, N)
####        butter_unwr[..., k] -= delta_i[..., Un_sol[0][k]]
####        
####    butter_mask = np.ma.array(butter_unwr, mask = mask_tile)
####    mean_butt = butter_mask.mean(axis=(0,1))
####    butter_unwr -= mean_butt
####    but_med = np.median(butter_unwr, axis = 2)
####
####    but_med_flat = but_med[xy_inside]
####    a_butt = np.linalg.lstsq(Zernike_2d, but_med_flat)[0]
####    piston, rms_mat_butt[jjj] = opt.fmin(rms_piston, 0, args = (j_max, a_butt, N, Z_mat, orig, mask, flipint), full_output = True)[:2]  
####    a_def_butt[jjj] = a_butt[2]
##
####json.dump(rms_mat_butt.tolist(), open(orig_fold + "rms_mat_butt.txt", 'w'))
####json.dump(a_def_butt.tolist(), open(orig_fold + "a_def_butt.txt", 'w'))
##
####for jjj in range(len(k_I)):
####    print("filtering k = " + str(k_I[jjj]))
####    org_phase = np.load(orig_fold + "org_phase.npy")
####    org_phase_butt = np.load(orig_fold + "org_phase.npy")
####    delta_i = np.load(orig_fold + "delta_i.npy")
####    Un_sol = np.triu_indices(delta_i.shape[-1], 1)
####
####    for k in range(org_phase.shape[-1]):
####        org_phase[..., k] = pw.filter_wrapped_phase(org_phase[...,k], k_I[jjj])
####        org_unwr[..., k] = pw.unwrap_phase_dct(org_phase[..., k], xx, yy, ii, jj, N, N)
####        org_unwr[..., k] -= delta_i[..., Un_sol[0][k]]
####
####    org_mask = np.ma.array(org_unwr, mask = mask_tile)
####    mean_unwr = org_mask.mean(axis=(0,1))
####    org_mask -= mean_unwr
####    org_med = np.median(org_mask, axis = 2)
####    
####    org_med_flat = org_med[xy_inside]
####    a_inter = np.linalg.lstsq(Zernike_2d, org_med_flat)[0]
####    piston_mat[jjj], rms_mat_avg[jjj] = opt.fmin(rms_piston, 0, args = (j_max, a_inter, N, Z_mat, orig, mask, flipint), full_output = True)[:2]
####    a_def_avg[jjj] = a_inter[2]
##
####    
####json.dump(rms_mat_avg.tolist(), open(orig_fold + "rms_mat_k_k.txt", 'w'))
####json.dump(a_def_avg.tolist(), open(orig_fold + "a_def_avg.txt", 'w'))
##
##rms_mat_butt = load_json(orig_fold + "rms_mat_butt.txt")
##a_def_butt = load_json(orig_fold + "a_def_butt.txt")
##rms_mat_avg = load_json(orig_fold + "rms_mat_k_k.txt")
##a_def_avg = load_json(orig_fold + "a_def_avg.txt")
##
##f, ax = plt.subplots(figsize = (0.66 * 4.98, 0.66*3.07))
##xaxis = f0 / (nx_pad * px_size_int)
##ax.semilogx(xaxis, rms_mat_butt, 'ko')
##ax.set_xlabel(r'$f_c \left[\textrm{m}^{-1}\right]$', fontsize = 7)
##ax.set_ylabel(r'$\varepsilon$', color  = 'k', fontsize = 7)
##ax.tick_params('y', colors = 'k')
##ax2 = ax.twinx()
##ax2.semilogx(xaxis, a_def_butt, 'bd')
##ax2.set_ylabel(r'$a_2^0$', color = 'b')
##ax2.tick_params(axis = 'y', colors = 'b')
##ax2.set_xlim(left = 200, right = 20000)
##f.savefig(folder + "rms_vs_f0.png", dpi = dpi_num, bbox_inches = 'tight')
##
##f, ax = plt.subplots(figsize = (0.66 * 4.98, 0.66*3.07))
##ax.plot(k_I, rms_mat_avg, 'ko')
##ax.set_xlabel('k')
##ax.set_xticks(k_I)
##ax.set_ylabel(r'$\varepsilon$', color = 'k', fontsize = 7)
##ax.tick_params(axis = 'y', colors = 'k')
##ax2 = ax.twinx()
##ax2.plot(k_I, a_def_avg, 'bd')
##ax2.set_ylabel(r'$a_2^0$', color = 'b',fontsize = 7)
##ax2.tick_params(axis = 'y', colors = 'b')
##ax.set_xlim(left = 0.5, right = max(k_I)+0.5)
##ax2.set_xlim(left = 0.5, right = max(k_I)+0.5)
##f.savefig(folder + "rms_vs_k.png", dpi = dpi_num, bbox_inches = 'tight')

plt.show()
