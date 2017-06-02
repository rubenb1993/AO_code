"""Using principal component analysis, check to see if the shack-hartmann is straight.
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import Hartmann as Hm

def principal_component_analysis(x_vec, y_vec):
    mx, my = 1./len(x_vec) * np.sum(x_vec), 1./len(y_vec) * np.sum(y_vec)
    S00 = np.sum((x_vec - mx)*(x_vec - mx))
    S01 = np.sum((x_vec - mx)*(y_vec - mx))
    S10 = np.sum((x_vec - my)*(y_vec - my))
    S11 = np.sum((y_vec - my) * (y_vec - my))

    S = np.array([[S00, S01], [S10, S11]])
    w, v = np.linalg.eig(S)
    #index = np.argmax(w)
    theta = np.rad2deg(np.arctan(v[1,:]/v[0,:]))
    return theta

#folder_name = "SLM_codes_matlab/20170404_measurements/20170404_astigmatism/"
folder_name = "SLM_codes_matlab/20170504_sub_zerns_4/"
box_len = 30
pad = 15


image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"), dtype = 'float')
mirror_ref = np.copy(image_ref_mirror)

zero_pos_dm = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"), dtype = 'float')
dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"), dtype = 'float')
flat_wf = np.array(PIL.Image.open(folder_name + "flat_wf.tif"), dtype = 'float')
sh_spots = np.dstack((image_ref_mirror, zero_pos_dm, dist_image)) 

zero_image = sh_spots[..., 1]
zero_image_zeros = np.copy(zero_pos_dm)
dist_image = sh_spots[..., 2]
image_control = sh_spots[..., 0]
[ny,nx] = zero_pos_dm.shape
x = np.arange(0, nx, 1)
y = np.arange(ny, 0, -1)
xx, yy = np.meshgrid(x, y)

x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(zero_pos_dm), spotsize = box_len)
min_x, max_x = np.min(x_pos_zero), np.max(x_pos_zero)
min_y, max_y = np.min(y_pos_zero), np.max(y_pos_zero)

print("gathering zeros")
#x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(mirror_ref), spotsize = box_len)
print("amount of spots: " + str(len(x_pos_zero)))
mirror_ref[:, :min_x - pad] = 0.
mirror_ref[:, max_x + pad:] = 0.
mirror_ref[:min_y - pad, :] = 0.
mirror_ref[max_y + pad:, :] = 0.
x_pos_zero, y_pos_zero = Hm.zero_positions(np.copy(mirror_ref), spotsize = box_len)

plt.imshow(mirror_ref)
plt.show()

x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, np.copy(image_ref_mirror), xx, yy, spot_size = box_len)
mx, my = np.mean(x_pos_flat), np.mean(y_pos_flat)

slant = principal_component_analysis(x_pos_flat, y_pos_flat)
print("slanted by " + str(slant[0]) + "degrees")
correction = np.deg2rad(-slant[0])
rotmat = np.array([[np.cos(correction), -np.sin(correction)], [np.sin(correction), np.cos(correction)]])

r =np.vstack((x_pos_flat -mx, y_pos_flat-my))
r_prime = np.dot(rotmat, r)
x_pos_cor, y_pos_cor = r_prime[0], r_prime[1]

f, ax = plt.subplots(1,2)
ax[0].scatter(x_pos_flat -mx, y_pos_flat-my, marker = 'o')
ax[1].scatter(x_pos_cor, y_pos_cor, marker = 's')
plt.show()

### test to see if recovered theta corresponds to "real" theta
##theta = np.deg2rad(0.2) 
##rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
##
##x, y = np.linspace(-100, 100, 50) + 0.00001*np.random.randn(50), np.linspace(-100, 100, 50)+ 0.00001*np.random.randn(50)
##xx, yy = np.meshgrid(x, y)
##
##
##x_prime, y_prime = np.einsum('ji, mni -> jmn', rotmat, np.dstack([xx, yy]))
##
##x_vec = x_prime.flatten()
##y_vec = y_prime.flatten()
####x_vec[:1250] += 0.00001*np.random.randn(1250)
####y_vec[1250:] += 0.00001*np.random.randn(1250)
##
##theta_recov = principal_component_analysis(x_vec, y_vec)
##print(theta_recov)
##n = theta_recov - np.rad2deg(theta)
##
##f, ax = plt.subplots(1,2)
##ax[0].scatter(x_vec, y_vec)
##plt.show()

