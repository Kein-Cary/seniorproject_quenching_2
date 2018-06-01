#####该部分脚本用于全部sigma信号的作图
###为了说明信号问题
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
import os.path
import numpy as np
import matplotlib.pyplot as plt
"""Read the Mandelbaum+2016 Weak Lensing data."""
m16path = 'D:/Python1/pydocument/seniorproject_quenching2/practice/data/M16/'
def read_m16_ds_2(use_red=True, mass_bin='10.0_10.4'):
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
    elif mass_bin in ['11.0_11.2','11.2_11.4','11.4_11.6','11.6_15.0']:
        fname = os.path.join(m16path, 'planck_lbg.ds.blue.rebinned.out')
        cols_dict ={
                '11.0_11.2': (0, 1, 2),
                '11.2_11.4': (0, 3, 4),
                '11.4_11.6': (0, 5, 6),
                '11.6_15.0': (0, 7, 8),
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

def read_m16_mass_2(use_red):
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
    # simply    
    #emhlow = lgmh - np.log10(mhlow)
    emhupp = np.log10(mhupp) - lgmh
    # errlgmh = (emhlow + emhupp) * 0.5
    # (arbitrarily) take the upper errorbar
    errlgmh = emhupp
    # out = [lgms, lgmh, emhlow, emhupp]
    out = [lgms, lgmh, errlgmh]
    return(out)
def test_read_m16_ds_2(use_red,mass_bin):
    """Test the M16 Reader."""
    ##加入数组，记录rp,ds,ds_error的变化和取值。
    rp, ds, ds_err = read_m16_ds_2(use_red=True, mass_bin=mass_bin)
    rsa = np.zeros(len(rp),dtype=np.float)
    dssa = np.zeros(len(rp),dtype=np.float)
    ds_errsa = np.zeros(len(rp),dtype=np.float)
    if use_red==True:
        rp, ds, ds_err = read_m16_ds_2(use_red=use_red, mass_bin=mass_bin)
        rsa = rp
        dssa = ds
        ds_errsa = ds_err
        plt.errorbar(rp, ds, yerr=ds_err, marker="o", ms=3, color="red", label='red')
    elif use_red==False and (mass_bin=='11.4_11.6' or mass_bin=='11.6_15.0'):
         rp, ds, ds_err = read_m16_ds_2(use_red=False, mass_bin=mass_bin)
         #ds = np.abs(ds)
         ds = ds
         rsa = rp
         dssa = ds
         ds_errsa = ds_err
         plt.errorbar(rp, ds, yerr=ds_err, marker="s", ms=3, color="blue", label='blue')
    else:
         rp, ds, ds_err = read_m16_ds_2(use_red=False, mass_bin=mass_bin)
         rsa = rp
         dssa = ds
         ds_errsa = ds_err
         plt.errorbar(rp, ds, yerr=ds_err, marker="s", ms=3, color="blue", label='blue')
    
    plt.xlabel(r"$R\;[Mpc/h]$")
    plt.ylabel(r"$\Delta\Sigma\;[h M_\odot/pc^2]$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.grid()
    plt.legend(loc=1)
    #print('rp=',rp)
    #print('ds=',dssa)
    #print('error=',ds_errsa)
    plt.title('Observal-signal')
    #plt.savefig('Observal-signal',dpi=600)
    plt.show()
    
    return rsa,dssa,ds_errsa
    #-sa表示设置数组
##最后图示的是该段代码所做图像
def test_read_m16_mass_2():
    lgms, lgmh, err = read_m16_mass_2(use_red=True)
    plt.errorbar(lgms, lgmh, yerr=err, marker="o", ms=3, color="red")
    lgms, lgmh, err = read_m16_mass_2(use_red=False)
    plt.errorbar(lgms, lgmh, yerr=err, marker="s", ms=3, color="blue")
    plt.xlabel(r"$M_*\;[M_\odot/h^2]$")
    plt.ylabel(r"$M_h\;[M_\odot/h]$")
    plt.grid()
    plt.show()
###下面一个函数分离数据
def mass_be_(tt):
    rp, ds, ds_err = read_m16_ds_2(use_red=True, mass_bin='10.0_10.4')
    mr_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_11.2',
             '11.2_11.4','11.4_11.6','11.6_15.0','11.0_15.0']
    #mb_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_15.0']
    ##该句表示四个蓝色星系的版本，对应后面的星系红移选取z_blue用循环赋值的部分
    mb_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_11.2',
             '11.2_11.4','11.4_11.6','11.6_15.0','11.0_15.0']
    rpr = rp
    ds_r = np.zeros((len(mr_be),len(rpr)),dtype=np.float)
    ds_err_r = np.zeros((len(mr_be),len(rpr)),dtype=np.float)     
    for k in range(0,len(mr_be)):
        mass_ = mr_be[k]
        rsa,dssa,ds_errsa = test_read_m16_ds_2(True,mass_bin=mass_)
        ds_r[k,:] = dssa
        ds_err_r[k,:] = ds_errsa
    for k in range(0,len(mb_be)):
        rp, ds, ds_err = read_m16_ds_2(use_red=False, mass_bin=mb_be[k])
        rpb = rp
        ds_b = np.zeros((len(mb_be),len(rpb)),dtype=np.float)
        ds_err_b = np.zeros((len(mb_be),len(rpb)),dtype=np.float) 
        mass_ = mb_be[k]
        rsa,dssa,ds_errsa = test_read_m16_ds_2(False,mass_bin=mass_)
        ds_b[k,:] = dssa
        ds_err_b[k,:] = ds_errsa
    return ds_r,ds_err_r,ds_b,ds_err_b
mass_be_(tt=True)