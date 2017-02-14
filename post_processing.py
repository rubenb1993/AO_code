import numpy as np
import matplotlib.pyplot as plt
import Zernike as Zn
import PIL.Image
import scipy.optimize as opt
import Hartmann as Hm
import mirror_control as mc
import janssen
#import compare_methods as com_met

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


folder_name = "20170117_coma/"
### predetermined constants
x0 = 550
y0 = 484
radius = int(280)
N = 2 * radius
j_max = 30
j_range = np.arange(2, j_max +2)
dpi_num = 600
golden = (1 + 5**0.5)/2.0
int_im_size = (4.98, 3.07)


## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
box_len = 35.0 #half width of subaperture box in px
wavelength = 632e-9 #[m]


xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
xi, yi = np.meshgrid(xi, yi)
folder_names = ["20170126_single_actuators/1/", "20170126_single_actuators/3/", "20170126_single_actuators/6/", "20170126_single_actuators/8/", "20170125_spherical/", "20170125_astigmatism/", "20170125_trifoil/", "20170125_coma/"]
x_labs = ["s.a. 1", "s.a. 3", "s.a. 6", "s.a. 8", "spher", "ast", "tri", "coma"]
#folder_names = ["20170117_coma/"]
#x_ticks = ["coma"]
f_r, ax_r = plt.subplots(1,1, figsize = int_im_size)
f_c, ax_c = plt.subplots(1,1, figsize = int_im_size)
x_pos = np.arange(len(folder_names))

for i in range(len(folder_names)):
    folder_name = folder_names[i]
    ## gather all variables
    variables_dict = np.load(folder_name + "vars_dictionary_norm.npy").item()
    coeff_dict= np.load(folder_name + "coeff_dictionary_norm.npy").item()


    ### coefficients to fix jannsen's a
    image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
    zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"))
    dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"))
    flat_wf = np.array(PIL.Image.open(folder_name + "flat_wf.tif"))
    image_control = np.copy(image_ref_mirror)
    zero_image_zeros = np.copy(zero_pos_dm)
    zero_image = np.copy(zero_pos_dm)
    [ny,nx] = zero_image.shape
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)
    x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image_zeros)
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
    centre = np.zeros(2)
    centre[0] = variables_dict['vars_janss'][1]
    centre[1] = variables_dict['vars_janss'][2]
    r_sh_px = variables_dict['vars_janss'][3]
    r_sh_m_janss = r_sh_px * px_size_sh
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
    inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (box_len/r_sh_px)))
    x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)

    a_janss = janssen.coeff_optimum(x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f, xx, yy, dist_image, image_control, px_size_sh, f_sh, r_sh_m_janss, wavelength, j_max)
    a_lsq = coeff_dict['coeff_lsq']
    a_inter = coeff_dict['coeff_inter']
##    coeff_dict['coeff_janss'] = a_janss
##    np.save(folder_name + "coeff_dictionary_norm.npy", coeff_dict)

    pist_janss = variables_dict['vars_janss'][0]
    pist_lsq = variables_dict['vars_lsq'][0]
    pist_inter = variables_dict['pist_inter']

##    ax_r.plot(x_pos[i], variables_dict['vars_janss'][3], 'r^')
##    ax_r.plot(x_pos[i], variables_dict['vars_lsq'][3], 'bs')

    ax_c.plot(x_pos[i], variables_dict['vars_janss'][1], 'r<')
    ax_c.plot(x_pos[i], variables_dict['vars_janss'][2], 'r^')
    ax_c.plot(x_pos[i], variables_dict['vars_lsq'][1], 'bs')
    ax_c.plot(x_pos[i], variables_dict['vars_lsq'][2], 'bD')
    ax_c.plot(x_pos[i], centre[0], 'g+')
    ax_c.plot(x_pos[i], centre[1], 'gx')
    
##    mask = [np.sqrt((xi) ** 2 + (yi) ** 2) >= 1]
##
##    inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
##    orig = np.fliplr(np.array(PIL.Image.open(folder_name + "orig_scale.tif")))#np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask)
##    ##orig /= orig.max()
##
##
##    power_mat = Zn.Zernike_power_mat(j_max+2)
##    Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_range)
##    flipint = False
##    #piston, inter_rms = opt.fmin(rms_piston, -1.89, args = (j_max, a_inter, N, Z_mat, orig, mask, flipint), full_output = True)[:2]
##    #power_mat
##    #pist_inter, inter_rms = opt.fmin(comp_met.rms_piston, 0, args = (j_max, a_inter, N, Z_mat, orig, mask, flipint), full_output = True)[:2]
##
##    f, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (4.98, 4.98/3), gridspec_kw={'width_ratios':[4, 4, 4, 4, 0.4], 'height_ratios':[1,1]})
##    titles = [r'original', r'Interferogram', r'LSQ', r'Janssen']
##    interf = axes[0,0].imshow(np.ma.array(orig, mask = mask), vmin = 0, vmax = 1, cmap = 'bone', origin = 'lower', interpolation = 'none')
##    Intens_inter = Zn.imshow_interferogram(j_max, a_inter, N, piston = pist_inter, ax = axes[0,1])
##    Intens_lsq = Zn.imshow_interferogram(j_max, a_lsq, N, piston = pist_lsq, ax = axes[0,2], fliplr = False)
##    Intens_janss = Zn.imshow_interferogram(j_max, a_janss, N, piston = pist_janss, ax = axes[0,3], fliplr = False)
##
##    ## create histogram
##    bin_hist = np.linspace(0, 1, 30)
##    width_hist = bin_hist[2]-bin_hist[1]
##    weight_mask = np.asarray(~mask[0], dtype = np.int32)
##    hist_janss, bin_edges = np.histogram(Intens_janss, bins = bin_hist, weights = weight_mask)
##    hist_lsq, bin_edge_lsq = np.histogram(Intens_lsq, bins = bin_hist, weights = weight_mask)
##    hist_inter, bin_edge_inter = np.histogram(Intens_inter, bins = bin_hist, weights = weight_mask)
##    hist_orig, bin_edge_orig = np.histogram(orig, bins = bin_hist, weights = weight_mask)
##
##    axes[1,0].bar(bin_edge_orig[:-1], hist_orig, width = width_hist)
##    axes[1,1].bar(bin_edge_inter[:-1], hist_inter, width = width_hist)
##    axes[1,2].bar(bin_edge_lsq[:-1], hist_lsq, width = width_hist)
##    axes[1,3].bar(bin_edges[:-1], hist_janss, width = width_hist)
##
##    hist_max = np.max(np.vstack((hist_orig, hist_inter, hist_lsq, hist_janss)))
##    hist_max = int(10000 * np.ceil(hist_max/10000))
##
##    for ax in axes[0]:
##        ax.set(adjustable = 'box-forced', aspect = 'equal')
##        ax.get_xaxis().set_ticks([])
##        ax.get_yaxis().set_ticks([])
##
##    for ax in axes[1]:
##        ax.set_ylim(top = hist_max)
##        ax.set(adjustable = 'box-forced', aspect = 1/float(hist_max) )
##        ax.set_xticks([0, 1])
##        ax.set_xticklabels(['0', '1'], fontsize = 4)
##        ax.set_yticks([0, hist_max])
##        ax.set_yticklabels(['0', str(hist_max)], fontsize = 4)
##        
##    for i in range(4):
##        axes[0,i].set_title(titles[i], fontsize = 7)
##    cbar = plt.colorbar(interf, cax = axes[0,-1], ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
##    cbar.ax.tick_params(labelsize=4)
##    axes[0, -1].set(adjustable = 'box-forced', aspect = 10)
##    axes[1,4].axis('off') #don't show this empty axis
##    f.savefig(folder_name + "histogram_and_piston_avg.png", bbox_inches = 'tight', dpi = dpi_num)
##    #f.savefig(fold_name+'methods_compared_additional_filter.png', bbox_inches = 'tight', dpi = dpi_num)
##
##    f, ax = plt.subplots(1,1, figsize = (4.98/2.0, 4.98/2.0))
##    ax.plot(Intens_inter[:, radius], label = 'Reconstruction', linewidth = 1)
##    #ax.plot(Intens_lsq[:, radius], label = 'LSQ', linewidth = 1)
##    #ax.plot(Intens_janss[:, radius], label = 'Janssen', linewidth = 1)
##    ax.plot(orig[:, radius], label = 'Original', linewidth = 1)
##
##    x_ticks = np.linspace(0, N, 3)
##    x_tick_labels = np.linspace(-1, 1, 3)
##    ax.set_xticks(x_ticks)
##    ax.set_xticklabels(x_tick_labels)
##    ax.set_xlabel(r'$y$', fontsize = 5)
##    ax.set_ylabel(r'Normalized Intensity', fontsize = 5)
##    ax.tick_params(axis = 'both', labelsize = 5)
##    ##make legend
##    box = ax.get_position()
##    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
##    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 2, fontsize = 4.5)
##
##    f.savefig(folder_name + "presentation_cutout_avg.png", bbox_inches = 'tight', dpi = dpi_num)
##
##    f2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = (4.98, 4.98/4.4))
##    a_x = np.arange(2, j_max+2)
##    ##msv = np.argmax(np.abs(a_janss)) #most_significant_value
##    ##sign_janss = np.sign(a_janss[msv])
##    ##sign_int = np.sign(a_inter[msv])
##    ##a_inter *= (sign_janss * sign_int) ## make signs equal of janss and interferogram. if signs are equal, will result in 1, else -1.
##    ax2.plot(a_x, a_inter, 'sk', label = 'interferogram')
##    ax2.plot(a_x, a_janss, 'og', label = 'janssen')
##    ax2.plot(a_x, a_lsq, '2b', label = 'LSQ')
##    ax2.set_xlim([0,30.5])
##    min_a = np.min([np.min(a_inter), np.min(a_janss), np.min(a_lsq)])
##    max_a = np.max([np.max(a_inter), np.max(a_janss), np.max(a_lsq)])
##    b = 1 * golden * (max_a - min_a)
##    ax2.set_ylim([min_a - b, max_a + b])
##    ax2.legend(prop={'size':5}, loc = 'best')
##    f2.savefig(folder_name + "new_coefficient_plot_avg.png", bbox_inches = 'tight', dpi = dpi_num)
##ax_r.plot(x_pos[-1], variables_dict['vars_janss'][3], 'r^', label = 'Janssen')
##ax_r.plot(x_pos[-1], variables_dict['vars_lsq'][3], 'bs', label = 'LSQ')

ax_c.plot(x_pos[-1], variables_dict['vars_janss'][1], 'r<', label = 'Janssen x')
ax_c.plot(x_pos[-1], variables_dict['vars_janss'][2], 'r^', label = 'Janssen y')
ax_c.plot(x_pos[-1], variables_dict['vars_lsq'][1], 'bs', label = 'LSQ x')
ax_c.plot(x_pos[-1], variables_dict['vars_lsq'][2], 'bD', label = 'LSQ y')
ax_c.plot(x_pos[-1], centre[0], 'g+', label = 'Guess x')
ax_c.plot(x_pos[-1], centre[1], 'gx', label = 'Guess y')

##ax_r.margins(0.1)
ax_c.margins(0.1)
##ax_r.set_xticks(x_pos)
ax_c.set_xticks(x_pos)
##ax_r.set_xticklabels(x_labs, rotation = 60)
ax_c.set_xticklabels(x_labs, rotation = 60)
##ax_r.tick_params(axis = 'both', labelsize = 7)
ax_c.tick_params(axis = 'both', labelsize = 7)
##box_r = ax_r.get_position()
box_c = ax_c.get_position()
##ax_r.set_position([box_r.x0, box_r.y0, box_r.width, box_r.height * 0.8])
ax_c.set_position([box_c.x0, box_c.y0, box_c.width, box_c.height * 0.8])
##ax_r.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 2,  fontsize = 6.5)
ax_c.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol = 6, fontsize = 6.5)

##f_r.savefig("radius_comparison.png", bbox_inches = 'tight', dpi = dpi_num)
f_c.savefig("centre_comparison_no_radius.png", bbox_inches = 'tight', dpi = dpi_num)

plt.show()
