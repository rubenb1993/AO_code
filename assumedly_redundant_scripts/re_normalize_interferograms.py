import sys
##import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
from matplotlib import rc
import Hartmann as Hm
import displacement_matrix as Dm
import Zernike as Zn
import mirror_control as mc
import LSQ_method as LSQ
import matplotlib.pyplot as plt
import janssen
import phase_extraction_slm as PE
import phase_unwrapping_test as pw
import scipy.optimize as opt
import scipy.ndimage as ndimage
import json #to write editable txt files

folder_extensions = ["Z_5_1/", "Z_5_3/", "Z_5_5/", "Z_4_m2_2_0/", "Z_4_m4_3_3/", "Z_5_m3_4_0/", "Z_6_0_4_0_2_0/", "Z_7_1_5_1_3_1/"]#["coma/", "defocus/", "astigmatism/", "spherical/"]
#folder_extensions = ["coma/", "defocus/", "astigmatism/", "spherical/"]
folder_names = ["SLM_codes_matlab/20170405_" + folder_extensions[i] for i in range(len(folder_extensions))]

#save_string = ["coma", "defocus", "astigmatism", "spherical"]
save_string = ["5_coma", "5_trifoil", "5_pentafoil", "mix_defocus", "mix_trifoil", "mix_spherical", "all_spherical", "all_coma"]#["x_coma", "y_coma", "mix", "x_coma_high_order", "asymetric_coma", "asymetric_astigmatism"]
save_fold = "SLM_codes_matlab/reconstructions/increasing_j/"
radius_fold = "SLM_codes_matlab/reconstructions/"

## centre and radius of interferogam. Done by eye, with the help of define_radius.py
### x0, y0 of 20170404 measurements: 464, 216
### x0, y0 of 20170405 measurements: 434, 254
r_int_px = 600/2
x0 = 434 + r_int_px
y0 = 254 + r_int_px
radius = int(r_int_px)
x = np.linspace(-1, 1, 2*r_int_px)
xx, yy = np.meshgrid(x, x)
mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= 1]
hist_yes = np.where(xx**2 + yy**2 <= 1)
hist_mask = np.zeros(xx.shape)
hist_mask[hist_yes] = 1

cut_off = 8.0 / 100
f, ax = plt.subplots(3, len(folder_extensions))
for ii in range(len(folder_extensions)):
    folder_name = folder_names[ii]
    inter_0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
    orig = np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask, dtype = 'float')#np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))[#
    orig = np.flipud(np.fliplr(orig))
    ax[0, ii].imshow(orig/orig.max(), origin = 'lower', cmap = 'gray')
    bins = np.linspace(0, orig.max(), 50)
    intensities, bin_edges = np.histogram(orig, bins, weights = hist_mask)
    tot = np.sum(orig)
    avgs = np.cumsum(np.diff(bin_edges)) - np.diff(bin_edges)[0]/2
    percentile = 1 - np.cumsum(intensities*avgs/tot)
    normalization_intensity = avgs[np.where(percentile < cut_off)[0][0]]
    orig /= normalization_intensity
    orig[orig>1.0] = 1.0
    bin_edges_norm = np.linspace(0,1, len(bins))
    intensities, bin_edges = np.histogram(orig, bin_edges_norm, weights = hist_mask)
    ax[1,ii].imshow(orig, origin = 'lower', cmap = 'gray')
    ax[2, ii].bar(bin_edges_norm[:-1], intensities, width = 1.0/(len(bins)-1))
    ax[2, ii].set_xlim([0, 1.0])
    ax[2, ii].set_ylim([0, 15000])
##    plt.bar(bin_edges[:-1], np.array(intensities, dtype ='float')/np.max(intensities), width = orig.max()/(len(bins)-1))
##    plt.plot(bin_edges[:-1], percentile, 'r-')
##    plt.plot([0, orig.max()], [0.02, 0.02], color = 'k', linestyle = '-')
##    plt.xlim([0, orig.max()])
##    plt.ylim(bottom = 0)
plt.show()
    #orig /= orig.max()
    
    #orig = np.flipud(np.fliplr(orig))

    
