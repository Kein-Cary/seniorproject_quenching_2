# 02 May 2017 00:48:45
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
#import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
def _read_recdict_from_hdf5(h5file):
    """Read a dict of record arrays from hdf5."""
    f = h5py.File(h5file, "r")
    recdict = {}
    #for grp, val in f.iteritems():
    ##新的h5py模块里面，修改为items包
    for grp, val in f.items():
        print(grp)
        datasets = []
        dtypes = []
        for key in f[grp].keys():
            dset = f[grp][key][:]
            dtypename = f[grp][key].dtype.name
            dtype = (str(key), dtypename)
            datasets.append(dset)
            dtypes.append(dtype)
        print(dtypes)
        recdict[str(grp)] = np.rec.fromarrays(tuple(datasets), dtype=dtypes)
    f.close()
    return(recdict)
def read_mock(mockfile):
    """Read hdf5 galaxy mock data into a numpy.recarray"""
    recdict = _read_recdict_from_hdf5(mockfile)
    mockdata = recdict['galaxy']
    print("The data columns are: ")
    print(mockdata.dtype.names)
    return(mockdata)
def read_mock_hmf(mockfile, mmin=1.e9, mmax=1.e16, nmbin=101, h=0.701, rcube=250.0):
    """Read Halo Mass Function.

    Returns
    ---
    Mh_arr: ndarray
        Halo mass in Msun.

    dndlnMh_arr: ndarray
        Number density in # per lnMsun per Mpc^3.
    """
    galrec = read_mock(mockfile)
    iscen = galrec['lg_halo_mass'] > 0
    Mh_arr = np.logspace(np.log10(mmin), np.log10(mmax), nmbin)
    wid = np.log10(Mh_arr[1] / Mh_arr[0])
    _wid = np.log(Mh_arr[1] / Mh_arr[0])
    # print wid
    _Mh_arr = np.zeros(Mh_arr.size+1)
    _Mh_arr[1:] = Mh_arr * 10**(0.5*wid)
    _Mh_arr[0] = Mh_arr[0] / 10**(0.5*wid)
    dn_arr = np.histogram(galrec['lg_halo_mass'][iscen] - np.log10(h), bins=np.log10(_Mh_arr))[0]
    dndlnMh_arr = dn_arr / _wid
    vol = (rcube / h)**3
    dndlnMh_arr /= vol
    return(Mh_arr, dndlnMh_arr)
def plot_mock_hmf(mockfile, rcube=250.0):
    ##模拟坐标为共栋坐标，尺寸为：250Mpc/h（每个维度）
    """plot the halo mass function in the mock."""
    Mh_arr, dndlnMh_arr = read_mock_hmf(mockfile, mmin=1.e10, mmax=1.e16, nmbin=50, h=0.7, rcube=rcube)
    plt.plot(np.log10(Mh_arr), dndlnMh_arr, 'k-', label="Simulation")
    plt.legend(loc=1)
    plt.yscale('log')
    plt.xlabel(r"$M_h\;[M_\odot]$")
    plt.ylabel(r"$dn/d\ln M_h$")
    plt.ylim(1e-8, 1e-1)
    plt.show()
if __name__ == "__main__":
    #mockfile = '/Users/ying/Data/ihodmock/standard/iHODcatalog_bolshoi.h5'
    #改为本机绝对路径
    mockfile = 'D:/Python1/pydocument/seniorproject_quenching2/practice/iHODcatalog_bolshoi.h5'
    plot_mock_hmf(mockfile)
