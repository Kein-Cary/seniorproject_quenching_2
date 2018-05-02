###这个脚本表示colossus怎么做profile图像
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
import matplotlib.pyplot as plt
import numpy as np
def profile_check(c):
    a = [1E12,1E13,1E14]
    for k in range(0,3):
        x0 = a[k]
        cosmology.setCosmology('planck100',\
                           {'flat': True, 'H0': 67.3, 'Om0': 0.315, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95})
        p_nfw = profile_nfw.NFWProfile(M = x0, c = 3.5, z = 0, mdef = 'vir')
        r = np.logspace(-4, 3.5, 1000)
        rho = p_nfw.density(r)
        ##这里的r表示rs
        #plt.loglog(r, rho)
        plt.loglog(10**(-3)*r, 10**9*rho,ls='--')
    #plt.xlabel(r'$kpc/h$')
    #plt.ylabel(r'$h^2M_\odot/kpc^3$')
    plt.xlabel(r'$Mpc/h$')
    plt.ylabel(r'$h^2M_\odot/Mpc^3$')
    #plt.savefig(r'$rho-r$',dpi=600)
    return
profile_check(c=True)
'''
if __name__ == "__main__":
    profile_check(c=True)
    pass
'''