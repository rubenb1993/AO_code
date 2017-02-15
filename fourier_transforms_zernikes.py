import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
import Zernike as Zn
from matplotlib import rc
import scipy.signal as sn

"""code based on the fourier transforming nature of a lens. In the easiest case
I_front(x, y) scales with |U_back(x/(lambda*f), y/(lambda*f))|^2 """

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=True)

def fourier_zernike(n, m, FX, FY):
##    nx = 540 #size of window of interferogram analysis
##    dx = 1.0/nx
##    dfx = 1 #1/diamter zernike polynomials
    #X_n, Y_n = np.linspace(xmin, xmax, 300), np.linspace(xmin, xmax, 300)
    #k = np.linspace(0.01, 300, 300)
    #phi = np.linspace(0, 2*np.pi, 300)
    #kk, pphi = np.meshgrid(k, phi)
    #X = kk * np.cos(pphi)
    #Y = kk * np.sin(pphi)
    #XX_n, YY_n = np.meshgrid(X_n, Y_n)
    kk = np.sqrt(FX**2 + FY**2)
    pphi = np.arctan2(FY, FX)
    prefact = np.sqrt(n+1) * sp.jv(n+1, 2 * np.pi * kk)/(np.pi * kk)
    if m > 0:
        Q = prefact * (-1)**((n-m)/2) * 1j**m * np.sqrt(2) * np.cos(m*pphi)
    elif m < 0:
        Q = prefact * (-1)**((n-m)/2) * 1j**m * np.sqrt(2) * np.sin(m*pphi)
    else:
        Q = prefact * (-1)**(n/2)
    return Q

##fig, ax = plt.subplots(1,2)
j_max = 40
x, y = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
xx, yy = np.meshgrid(x, y)
power_mat = Zn.Zernike_power_mat(j_max)

wavelength = 632e-9
focal = 25.0e-2
R = 1.0e-2
NA = R / focal
lambda_NA = wavelength / NA

### fourier transform units in cm.....
#dx =  10.0e-2 / (
#dy = 0.001
dfx = 1000
dfy = 1000
x_f, y_f = np.linspace(-30e-2, 30e-2, dfx), np.linspace(-30e-2, 30e-2, dfx)
##xx_f, yy_f = np.meshgrid(x_f, y_f)
##defocus = Zn.Zernike_xy(xx_f*1e2, yy_f*1e2, power_mat, [1,9])
##defocus = np.sum(defocus, axis = 2)
##circ = np.zeros(xx_f.shape)
##circ[np.where(np.sqrt(xx_f**2 + yy_f**2 )< R)] = 1
##defocus *= circ
##plt.imshow(defocus)
dx = x_f[1] - x_f[0]
dy = dx
fx, fy = np.linspace(-0.5/dx, 0.5/dx, dfx), np.linspace(-0.5/dy, 0.5/dy, dfy)
FX, FY = np.meshgrid(fx, fy)
##shift = np.exp(-2*np.pi*1j*(FX+FY))
##ft_znm = 1/(dx**2) * shift * np.fft.fftshift(np.fft.fft2(defocus))

### fourier transform
#Q_11 *= 10

X_l_NA = FX * wavelength * focal / (wavelength / NA)
Y_l_NA = FY * wavelength * focal / (wavelength / NA)
print(X_l_NA[dfx/2, dfx/2 + 2] - X_l_NA[dfx/2, dfx/2+1])
n = [10]
m = [0]
Q_11 = np.zeros(X_l_NA.shape, dtype = np.complex_)
for i in range(len(n)):
    Q_11 += fourier_zernike(n[i], m[i], X_l_NA, Y_l_NA)
#Q_11 = fourier_zernike(10,0, X_l_NA, Y_l_NA)
Q_11 /= np.max(np.abs(Q_11))
##ft_znm /= np.max(np.abs(ft_znm))
##X_real = X * wavelength * focal
##Y_real = Y * wavelength * focal
##X_l_NA = X * R
##Y_l_NA = Y * R
#F_circ = np.pi* R**2 * (2 * sp.jv(1, 2*np.pi* R * np.sqrt(X**2 + Y**2))/(2*np.pi*R * np.sqrt(X**2 + Y**2)))
#F_circ /= np.max(np.abs(F_circ))
#levels = np.linspace(0, np.max(np.abs(F_circ)**2), num = 30)
#f, ax = plt.subplots(1,1)
#ax.plot(X_l_NA[300, :], np.abs(F_circ[300, :])**2)#, levels = levels)
#ax.set_xlabel(r'$x~[\lambda / NA]$')
#ax.set_ylabel(r'$I$ [A.U.]')
#final_spot = sn.convolve2d(F_circ, Q_11, boundary = 'symm', mode = 'same')

##[ny, nx] = X.shape
##ax[0].plot(X[0, :], np.abs(Q_11)[0, :]**2)
##ax[1].plot(kk[0, :], prefact[0, :])
##
##f = plt.figure()
##ax = f.add_subplot(121, projection = '3d')
##ax.plot_surface(X_real, Y_real, np.abs(Q_11)**2, cmap = 'jet')
##ax2 = f.add_subplot(122)
##ax2.contourf(X_l_NA, Y_l_NA, np.abs(final_spot)**2, cmap = 'jet', origin = 'lower', interpolation = 'none')
##ax.set_xlabel('x [nm]')
##ax.set_ylabel('y [nm]')
##ax2.set_xlabel('x [nm]')
##ax2.set_ylabel('y [nm]')
##ax.set_zlabel('Intensity [a.u.]')
##plt.show()

f, ax = plt.subplots(1,2)
ax[0].contourf(X_l_NA, Y_l_NA, np.abs(Q_11)**2, cmap = 'jet', origin = 'lower', interpolation = 'none')
##ax[1].plot(X_l_NA[len(fx)/2, :], np.abs(ft_znm)[len(fx)/2, :]**2, 'b')
ax[1].plot(X_l_NA[len(fx)/2, :], np.abs(Q_11)[len(fx)/2, :]**2, 'g')
#ax[1].plot(X_l_NA[300, :], np.abs(F_circ)[300, :]**2, 'g')
ax[0].set_xlabel(r'$x~[\lambda / NA]$')
ax[0].set_ylabel(r'$y~[\lambda / NA]$')
ax[1].set_xlabel(r'$x~[\lambda / NA]$')
ax[1].set_ylabel(r'$I_{focal}[A.U.]$')
plt.show()
