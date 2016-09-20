import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import Zernike as Zn

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
    
##### Make list of maxima given "flat" wavefront ####

        
#### check the numerical derivatives with analytical ones from Stephenson [1] ####
def check_num_der(savefigure = False):
    """Function to check if the derivatives calculated by xderZ and yderZ comply with analytical values given by Stephenson [1]
    savefigure = True will save the figure generated."""
    rho = np.linspace(0, 1, 50)
    #theta = 0.0 * np.pi/2 * np.ones(rho.shape)
    theta = np.pi/2 * np.ones(rho.shape)
    x, y = pol2cart(rho, theta)
    n = np.array([4.0, 5.0, 5.0, 5.0, 5.0])
    m = np.array([0.0, -1.0, -3.0, 1.0, 3.0])
    j = Zn.Zernike_nm_2_j(n, m)
    
    #set 3d display parameters
    r_mat = np.linspace(0,1,20)
    theta_mat = np.linspace(0, 2*np.pi, 20)
    radius_matrix, theta_matrix = np.meshgrid(r_mat,theta_mat)
    X, Y = radius_matrix*np.cos(theta_matrix), radius_matrix*np.sin(theta_matrix)
    
    xderZ_com = np.zeros((len(x),len(j)))
    yderZ_com = np.zeros((len(x),len(j)))
    xderZ_ana = np.zeros((len(x),len(j)))
    yderZ_ana = np.zeros((len(x),len(j)))
    Z = np.zeros((len(r_mat), len(r_mat), len(j)))
    
    
    for i in range(len(j)):
        xderZ_com[:,i] = Zn.xderZ(j[i], x, y)
        yderZ_com[:,i] = Zn.yderZ(j[i], x, y)
        Z[:,:,i] = Zn.Zernike_nm(n[i], m[i], radius_matrix, theta_matrix)
        
    xderZ_ana[:,0] = 12 * np.sqrt(5) * x * (2 * x**2 + 2 * y**2 -1)
    xderZ_ana[:,1] = 16* np.sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    xderZ_ana[:,2] = 8 * np.sqrt(3) * x * y * (15 * x**2 + 5 * y**2 - 6)
    xderZ_ana[:,3] = 2 * np.sqrt(3) * (50 * x**4 + 12 * x**2 * (5 * y**2 - 3) + 10* y**4 - 12* y**2 + 3)
    xderZ_ana[:,4] = 2 * np.sqrt(3) * (25 * x**4 - 6 * x**2 * (y**2 + 2) - 3 * y**2 * (5 * y**2 -4))
    #2 * np.sqrt(3) * (50 * x**4 + 12 * x**2 * (5 * y**2 - 3) + 10* y**4 - 12* y**2 + 3)
    
    yderZ_ana[:,0] = 12 * np.sqrt(5) * y * (2 * x**2 + 2 * y**2 -1)
    yderZ_ana[:,1] = 2 * np.sqrt(3) * (50 * x**4 + 12 * x**2 * (5 * y**2 - 3) + 10* y**4 - 12* y**2 + 3)
    yderZ_ana[:,2] = 2 * np.sqrt(3) * ( 15 * x**4 - 6 * x**2 * (5* y**2 - 2) - y**2 * (25 * y**2 - 12))
    yderZ_ana[:,3] = 16* np.sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    yderZ_ana[:,4] = -8 * np.sqrt(3) * x * y * (5 * x**2 + 15 * y**2 - 6)
    #16* np.sqrt(3) * x * y * (5 * x**2 + 5 * y**2 - 3)
    
    #makefigure
    f, axarr = plt.subplots(5, 2, sharex='col', sharey='row')
    f.suptitle('theta = ' + str(theta[0]), fontsize = 11)
    f2, axarr2 = plt.subplots(5, 1, sharex = True, sharey = True, figsize=(plt.figaspect(5.)))
    for ii in range(len(j)):
        ana, = axarr[ii,0].plot(rho, xderZ_ana[:,ii], 'r-', label='Analytic')
        comp, = axarr[ii,0].plot(rho, xderZ_com[:,ii], 'bo', markersize = 2, label='computational')
        axarr[ii,1].plot(rho, yderZ_ana[:,ii], 'r-', rho, yderZ_com[:,ii], 'bo', markersize = 2)
        f.legend((ana, comp), ('Analytical', 'Computational') , 'lower right', ncol = 2, fontsize = 9 )
            
        ZZ = axarr2[ii].contourf(X, Y, Z[:,:,ii], rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0 )
        
        axarr2[ii].set_xlim([-1, 1])
        axarr2[ii].set_ylim([-1, 1])
        axarr2[ii].set(adjustable = 'box-forced', aspect = 'equal') 
        cbar = plt.colorbar(ZZ, ax = axarr2[ii])      

    axarr[0,0].set_xlim([0, 1])

    axarr[4,0].set_xlabel(r'$ \rho $')
    axarr[0,0].set_ylabel(r'$Z_4^0$')
    axarr[1,0].set_ylabel(r'$Z_5^{-1}$')
    axarr[2,0].set_ylabel(r'$Z_5^{-3}$')
    axarr[3,0].set_ylabel(r'$Z_5^{1}$')
    axarr[4,0].set_ylabel(r'$Z_5^{3}$')
    axarr2[0].set_ylabel(r'$Z_4^0$')
    axarr2[1].set_ylabel(r'$Z_5^{-1}$')
    axarr2[2].set_ylabel(r'$Z_5^{-3}$')
    axarr2[3].set_ylabel(r'$Z_5^{1}$')
    axarr2[4].set_ylabel(r'$Z_5^{3}$')
    axarr[0,0].set_title(r'$\partial / \partial x$')
    axarr[0,1].set_title(r'$\partial / \partial y$')
    
    
    if savefigure:
        f.savefig('AO_code/derivatives_comparison_theta_pi_over_2.pdf', bbox_inches='tight', pad_inches=0.1)
        f2.savefig('AO_code/Zernikes_for_comparison.pdf', bbox_inches = 'tight', pad_inches = 0.1)
        plt.show()
    else:
        plt.show()
    return

def Check_zernike(savefigure = False):
    n = np.array([4.0, 5.0, 5.0, 5.0, 5.0])
    m = np.array([0.0, -1.0, -3.0, 1.0, 3.0])
    j = Zn.Zernike_nm_2_j(n, m)
    
    #set 3d display parameters
    r_mat = np.linspace(0,1,20)
    theta_mat = np.linspace(0, 2*np.pi, 20)
    radius_matrix, theta_matrix = np.meshgrid(r_mat,theta_mat)
    X, Y = radius_matrix*np.cos(theta_matrix), radius_matrix*np.sin(theta_matrix)
    
    Z_brug = np.zeros((len(r_mat), len(r_mat), len(j)))
    Z_not_brug = np.zeros((len(r_mat), len(r_mat), len(j)))
    power_mat = Zn.Zernike_power_mat(np.max(j))
    
    for i in range(len(j)):
        Z_brug[...,i] = Zn.Zernike_xy(X, Y, power_mat, j[i])
        Z_not_brug[:,:,i] = Zn.Zernike_nm(n[i], m[i], radius_matrix, theta_matrix)
        
    #makefigure
    f, axarr = plt.subplots(5, 2, sharex=True, sharey=True)
    for ii in range(len(j)):
        ZZ = axarr[ii,0].contourf(X, Y, Z_brug[:,:,ii], rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0 )
        ZZ2 = axarr[ii,1].contourf(X, Y, Z_not_brug[...,ii], rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth=0)
        
        axarr[ii,0].set_xlim([-1, 1])
        axarr[ii,0].set_ylim([-1, 1])
        axarr[ii,0].set(adjustable = 'box-forced', aspect = 'equal') 
        axarr[ii,1].set_xlim([-1, 1])
        axarr[ii,1].set_ylim([-1, 1])
        axarr[ii,1].set(adjustable = 'box-forced', aspect = 'equal') 
        cbar = plt.colorbar(ZZ, ax = axarr[ii,0])     
        cbar2 = plt.colorbar(ZZ2, ax = axarr[ii,1]) 


    axarr[4,0].set_xlabel(r'$ \rho $')
    axarr[0,0].set_ylabel(r'$Z_4^0$')
    axarr[1,0].set_ylabel(r'$Z_5^{-1}$')
    axarr[2,0].set_ylabel(r'$Z_5^{-3}$')
    axarr[3,0].set_ylabel(r'$Z_5^{1}$')
    axarr[4,0].set_ylabel(r'$Z_5^{3}$')
    axarr[0,0].set_title(r'Zernike using Brug')
    axarr[0,1].set_title(r'Zernike using Stephenson')
    print('Logic test if Z_brug and Z_not_brug are equal with tol = 1e-05 results in ' + str(np.allclose(Z_brug, Z_not_brug)))
    
    if savefigure:
        f.savefig('AO_code/zernike_comparison.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
    else:
        plt.show()
    return
    
def check_num_brug(savefigure = False):
    """Function to check if the derivatives calculated by xderZ and yderZ comply with analytical values given by Stephenson [1]
    savefigure = True will save the figure generated."""
    rho = np.linspace(0, 1, 50)
    theta = 0.0 * np.pi/2 * np.ones(rho.shape)
    x, y = pol2cart(rho, theta)
    n = np.array([4.0, 5.0, 5.0, 5.0, 5.0])
    m = np.array([0.0, -1.0, -3.0, 1.0, 3.0])
    j = Zn.Zernike_nm_2_j(n, m)
    power_mat = Zn.Zernike_power_mat(np.max(j))
    
    xderZ_com = np.zeros((len(x),len(j)))
    yderZ_com = np.zeros((len(x),len(j)))
    xderZ_ana = np.zeros((len(x),len(j)))
    yderZ_ana = np.zeros((len(x),len(j)))
    
    
    for i in range(len(j)):
        xderZ_com[:,i] = Zn.xderZ(j[i], x, y)
        yderZ_com[:,i] = Zn.yderZ(j[i], x, y)
        xderZ_ana[:,i] = Zn.xder_brug(x, y, power_mat, j[i])
        yderZ_ana[:,i] = Zn.yder_brug(x, y, power_mat, j[i])
    
    #makefigure
    f, axarr = plt.subplots(5, 2, sharex='col', sharey='row')
    f.suptitle('theta = ' + str(theta[0]), fontsize = 11)
    for ii in range(len(j)):
        ana, = axarr[ii,0].plot(rho, xderZ_ana[:,ii], 'r-', label='Brug')
        comp, = axarr[ii,0].plot(rho, xderZ_com[:,ii], 'bo', markersize = 2, label='Stephenson')
        axarr[ii,1].plot(rho, yderZ_ana[:,ii], 'r-', rho, yderZ_com[:,ii], 'bo', markersize = 2)
        f.legend((ana, comp), ('Stephenson', 'Brug') , 'lower right', ncol = 2, fontsize = 9 )     

    axarr[0,0].set_xlim([0, 1])

    axarr[4,0].set_xlabel(r'$ \rho $')
    axarr[0,0].set_ylabel(r'$Z_4^0$')
    axarr[1,0].set_ylabel(r'$Z_5^{-1}$')
    axarr[2,0].set_ylabel(r'$Z_5^{-3}$')
    axarr[3,0].set_ylabel(r'$Z_5^{1}$')
    axarr[4,0].set_ylabel(r'$Z_5^{3}$')
    axarr[0,0].set_title(r'$\partial / \partial x$')
    axarr[0,1].set_title(r'$\partial / \partial y$')
    
    
    if savefigure:
        f.savefig('AO_code/stephenson_brug_comparison.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
    else:
        plt.show()
    return