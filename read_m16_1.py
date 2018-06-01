import os.path
import numpy as np
import matplotlib.pyplot as plt
"""Read the Mandelbaum+2016 Weak Lensing data."""
m16path = 'D:/Python1/pydocument/seniorproject_quenching2/practice/data/M16/'
def read_m16_ds_1(use_red=True, mass_bin='10.0_10.4'):
    """Read DeltaSigma data from Mandelbaum+16.
    Parameters
    ---
    use_red: bool
        read data for the red or blue galaxies.
    mass_bin: str
        name of the stellar mass.
    Returns
    ---
    output: list
        [rp, ds, ds_err]
    """
    if use_red:
        # sm_10.0_10.4. - sm_10.4_10.7. - sm_10.7_11.0. - sm_11.0_11.2.\
        #- sm_11.2_11.4. - sm_11.4_11.6. - sm_11.6_15.0. - sm_11.0_15.0.
        fname = os.path.join(m16path, 'planck_lbg.ds.red.out')
        cols_dict ={
                '10.0_10.4': (0, 1, 2),
                '10.4_10.7': (0, 3, 4),
                '10.7_11.0': (0, 5, 6),
                '11.0_11.2': (0, 7, 8),
                '11.2_11.4': (0, 9, 10),
                '11.4_11.6': (0, 11, 12),
                '11.6_15.0': (0, 13, 14),
                '11.0_15.0': (0, 15, 16),
                }
    else:
        # sm_10.0_10.4. - sm_10.4_10.7. - sm_10.7_11.0. - sm_11.0_15.0.
        fname = os.path.join(m16path, 'planck_lbg.ds.blue.out')
        cols_dict ={
                '10.0_10.4': (0, 1, 2),
                '10.4_10.7': (0, 3, 4),
                '10.7_11.0': (0, 5, 6),
                '11.0_15.0': (0, 7, 8),
                }
    # Mpc/h, (h Msun/(physical pc)^2)
    rp, ds, ds_err = np.genfromtxt(fname, usecols=cols_dict[mass_bin],\
                                   unpack=True)
    return(rp, ds, ds_err)
def read_m16_mass_1(use_red):
    if use_red:
        usecols = [0, 1, 3, 4]
        # correction for differences in mass definitions
        dlgms = 0.20
    else:
        usecols = [5, 6, 8, 9]
        dlgms = 0.15
    fname = os.path.join(m16path, 'bootmass_1s_colorsplit_corr.txt')
    # new masses are M200m
    ms, mh, mhlow, mhupp = np.genfromtxt(fname, usecols=usecols,\
                                         unpack=True, skip_footer=1)
    _h = 0.673 # DONT CHANGE THIS
    # first convert ms to Msol/h^2
    ms = ms * _h**2
    lgms = np.log10(ms) + dlgms
    # mh is in Msun/h
    lgmh = np.log10(mh)
    # simply errors
    #emhlow = lgmh - np.log10(mhlow)
    emhupp = np.log10(mhupp) - lgmh
    # errlgmh = (emhlow + emhupp) * 0.5
    # (arbitrarily) take the upper errorbar
    errlgmh = emhupp
    # out = [lgms, lgmh, emhlow, emhupp]
    out = [lgms, lgmh, errlgmh]
    return(out)
def test_read_m16_ds_1(mass_bin):
    """Test the M16 Reader."""
    ##加入数组，记录rp,ds,ds_error的变化和取值。
    rp, ds, ds_err = read_m16_ds_1(use_red=True, mass_bin=mass_bin)
    rsa = np.zeros((2,len(rp)),dtype=np.float)
    dssa = np.zeros((2,len(rp)),dtype=np.float)
    ds_errsa = np.zeros((2,len(rp)),dtype=np.float)
    rsa[0,:] = rp
    dssa[0,:] = ds
    ds_errsa[0,:] = ds_err
    plt.errorbar(rp, ds, yerr=ds_err, marker="o", ms=3, color="red")
    rp, ds, ds_err = read_m16_ds_1(use_red=False, mass_bin=mass_bin)
    rsa[1,:] = rp
    dssa[1,:] = ds
    ds_errsa[1,:] = ds_err
    plt.errorbar(rp, ds, yerr=ds_err, marker="s", ms=3, color="blue")
    plt.xlabel(r"$R\;[Mpc/h]$")
    plt.ylabel(r"$\Delta\Sigma\;[h M_\odot/pc^2]$")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend(['red','blue'])
    #print('rp=',rp)
    #print('ds=',dssa)
    #print('error=',ds_errsa)
    #plt.savefig('Sigma_signal',dpi=600)
    #plt.show()
    return rsa,dssa,ds_errsa
    #-sa表示设置数组
    ##最后图示的是该段代码所做图像
def test_read_m16_mass_1():
    lgms, lgmh, err = read_m16_mass_1(use_red=True)
    plt.errorbar(lgms, lgmh, yerr=err, marker="o", ms=3, color="red")
    lgms, lgmh, err = read_m16_mass_1(use_red=False)
    plt.errorbar(lgms, lgmh, yerr=err, marker="s", ms=3, color="blue")
    plt.xlabel(r"$M_*\;[M_\odot/h^2]$")
    plt.ylabel(r"$M_h\;[M_\odot/h]$")
    plt.grid()
    #plt.show()

if __name__ == "__main__":
    test_read_m16_ds_1(mass_bin='11.0_15.0')
    # test_read_m16_mass()
    pass
