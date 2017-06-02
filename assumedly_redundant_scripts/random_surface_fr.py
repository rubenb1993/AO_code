import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt

def make_rand_vec(scale, size, **kwargs):
    # type: (float, int, dict) -> object
    a = np.random.normal(scale = scale, size = size)
    x = np.linspace(-10, 50, size)
    x *= x
    x += 2
    a /= x
    assert(len(a)==len(x))
    xx, yy = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
    xx, yy = np.meshgrid(xx, yy)

    if "power_mat" in kwargs:
        power_mat = kwargs["power_mat"]
    else:
        power_mat = Zn.Zernike_power_mat(len(a)+1)

    if "Z_mat" in kwargs:
        Z_mat = kwargs["Z_mat"]
    else:
        Z_mat = Zn.Zernike_xy(xx, yy, power_mat, np.arange(1, len(a)+1))
    phi = np.dot(Z_mat, a)
    dif_x_phi = np.diff(phi, axis = 0)
    dif_y_phi = np.diff(phi, axis = 1)

    if (np.max(dif_x_phi) or np.max(dif_y_phi)) > 0.1 * np.pi:
        kwarg_dict = {"Z_mat": Z_mat, "power_mat": power_mat}
        return make_rand_vec(scale, size, **kwarg_dict)

    else:
        return np.asarray(a)

a = np.asarray(make_rand_vec(3.0, 100))
print(a)
power_mat = Zn.Zernike_power_mat(len(a)+1)
xx, yy = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
xx, yy = np.meshgrid(xx, yy)

Z_mat = Zn.Zernike_xy(xx, yy, power_mat, np.arange(1, len(a)+1))
phi = np.dot(Z_mat, a)
mask = [xx ** 2 + yy ** 2 > 1]

f, ax = plt.subplots(1,2)
rand_surf = ax[0].imshow(np.ma.array(phi, mask = mask), origin = 'lower')
ax[1].imshow(np.ma.array(np.cos(phi/2.0)**2, mask = mask), origin = 'lower', cmap = 'bone', vmin = 0, vmax = 1)
plt.colorbar(rand_surf)
plt.show()