# 02 May 2017 00:48:45
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
##件的路径具体到文件夹
import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
####该局是为了调用对数数密度画图
def compare_data(C):
    data_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/data/M16'
    fname = os.path.join(data_path,'hmf.dat')
    lnMh_arr, dndlnMh_arr = np.genfromtxt(fname, unpack=True)
    plt.plot(lnMh_arr,dndlnMh_arr)
    plt.xlabel("$lnMh_{arr}$")
    plt.ylabel("$dndlnMh_{arr}$")
    plt.yscale('log')
    plt.show()
    return
#compare_data(C=True)
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
    #print(wid)
    _Mh_arr = np.zeros(Mh_arr.size+1)
    _Mh_arr[1:] = Mh_arr * 10**(0.5*wid)
    _Mh_arr[0] = Mh_arr[0] / 10**(0.5*wid)
    dn_arr = np.histogram(galrec['lg_halo_mass'][iscen] - np.log10(h), bins=np.log10(_Mh_arr))[0]
    dndlnMh_arr = dn_arr / _wid
    vol = (rcube / h)**3
    dndlnMh_arr /= vol
    ###下面一个数组作为把数据读取和处理的尝试
    viw = galrec['z_rs']
    con_galaxy = galrec['conc']
    col_galaxy = galrec['g-r']
    Mh_galaxy = galrec['lg_halo_mass']
    Mstar_galaxy = galrec['lg_stellar_mass']
    loc1_galaxy = galrec['x']
    loc2_galaxy = galrec['y']
    loc3_galaxy = galrec['z']
    g_id = galrec['halo_id']
    ###建立一个输出字典，把需要处理的和Mh-function有关的物理量取出来
    use_data = {'a':viw, 'b':con_galaxy, 'c':col_galaxy, 'd':Mh_galaxy, 'e':Mstar_galaxy,
                'f':loc1_galaxy, 'g':loc2_galaxy, 'h':loc3_galaxy, 'i':g_id
                }
    return(Mh_arr, dndlnMh_arr, use_data)
def plot_mock_hmf(mockfile, rcube=250.0):
    ##模拟坐标为共栋坐标，尺寸为：250Mpc/h（每个维度）
    """plot the halo mass function in the mock."""
    Mh_arr, dndlnMh_arr, use_data = read_mock_hmf(mockfile, mmin=1.e10, mmax=1.e16, 
                                                  nmbin=50, h=0.7, rcube=rcube)
    plt.plot(np.log10(Mh_arr), dndlnMh_arr, 'k-', label="Simulation")
    plt.legend(loc=1)
    plt.yscale('log')
    plt.xlabel(r"$M_h\;[M_\odot]$")
    plt.ylabel(r"$dn/d\ln M_h$")
    plt.ylim(1e-8, 1e-1)
    plt.show()
    N = np.array(use_data['b'].shape)
    print('size_con=',N)
    ###查看数据量：529308*8~=4000000

    ###把主晕取出来,并把主晕对应的物理量也取出来
    ix = use_data['d']>0
    _halos = use_data['d']
    ##取出主晕
    main_halo = np.array(_halos[ix])
    ##下面把与主晕有关的物理量取出来
    g_color = use_data['c']
    ##取出主晕下星系颜色
    gcolor = np.array(g_color[ix])
    ###尝试读取数据的处理
    M_star = use_data['e']
    ##取出恒星质量
    mstar = np.array(M_star[ix])
    ###求出中央星系的比例
    n = np.array(main_halo.shape)
    print('size_mainhalo=',n)
    frac_ = n[0]/N[0]
    print('f_c=',frac_)
    '''
    ###把模拟数据调入模型,计算透镜信号和密度分布
    ##引入模型计算模块
    from mok_calculation import f_plot
    f_plot(main_halo[0:10],z=0)
    '''
    ###第一部分作图,尝试用grid.spec完成子图分布
    ####该部分说明具体子图排列，用划分坐标轴的做法
    import matplotlib.gridspec as gridspec
    plt.figure()
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0,:])
    ax1.hist(main_halo,100)
    plt.xlabel(r'$Mh[M_\odot/h]$')
    plt.title('Main halo')
    plt.yscale('log')
    #plt.show()
    ax2 = plt.subplot(gs[1:,0])
    xv = ax2.hist(gcolor,100)
    plt.xlabel('g-r')
    #plt.show()
    #print(xv)
    ax3 = plt.subplot(gs[1:,1])
    nu = xv[0]
    nu = (nu-np.min(nu))/(np.max(nu)-np.min(nu))
    nv = xv[1]
    ax3.plot(nv[0:-1],nu)
    #plt.show()
    ##取出参数C
    ax4 = plt.subplot(gs[1:,2])
    con_c = use_data['b']
    con = np.array(con_c[ix])
    yv = ax4.hist(con,100)
    plt.xlabel('concentration')
    plt.yscale('log')
    #plt.show()
    ax5 = plt.subplot(gs[1:,3])
    mu = yv[0]
    mu = (mu-np.min(mu))/(np.max(mu)-np.min(mu))
    mv = yv[1]
    ax5.plot(mv[0:-1],mu)
    #plt.show()
    plt.tight_layout()
    #plt.savefig('Main_halo',dpi=600)
    plt.show()

    ####作图与拟合    
    ###对con,gcolor两个颜色矩阵做设置
    color1 = np.array((con-np.mean(con))/(np.std(con)/np.sqrt(N)))
    ###颜色1和星系的聚集程度对应
    color2 = np.array(gcolor)
    ###颜色2和星系的红化程度对应
  
    ####预先做一个关于SFR截断断和g-r的截断的计算和分布（阶段是指过了该值星系为red），最后需要得到一致的结果
    SFR = mstar-0.36*mstar-6.4
    ###所有量实在对数坐标系下的关系
    g_r = 0.8*(mstar/10.5)**0.6
    sfr = -0.35*(mstar-10.0)-10.23+mstar
    plt.hist2d(mstar,gcolor,bins=[100,100],
               range=[[np.min(mstar),np.max(mstar)],[np.min(gcolor),np.max(gcolor)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.plot(mstar,g_r,'k-')
    plt.show()
    plt.hist2d(SFR,gcolor,bins=[100,100],
               range=[[np.min(SFR),np.max(SFR)],[np.min(gcolor),np.max(gcolor)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.plot(SFR,sfr,'k--')
    plt.plot(SFR,g_r,'k-')
    plt.ylabel(r'$g-r$')
    plt.xlabel(r'$logSFR$')
    #plt.savefig('Hist2d-SFR',dpi=600)
    plt.show()
    ###二维统计分布图
    plt.hist2d(mstar,main_halo,bins=[100,100],
               range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel(r'$logM_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$logMh[M_\odot/h]$')
    #plt.savefig('Hist2d-1',dpi=600)
    plt.show()
    raise
    plt.hist2d(mstar,gcolor,bins=[100,100],
               range=[[np.min(mstar),np.max(mstar)],[np.min(gcolor),np.max(gcolor)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    ###norm.color.LogNorm表示在数密度计数上用对数计数
    plt.colorbar()
    plt.xlabel(r'$logM_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$g-r$')
    #plt.savefig('Hist2d-2',dpi=600)
    plt.show() 

    value3 = plt.hist2d(main_halo,gcolor,bins=[100,100],
               range=[[np.min(main_halo),np.max(main_halo)],[np.min(gcolor),np.max(gcolor)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    value3[0][np.isnan(value3[0])] = 0
    ###这句表示对value3里面nan的数全部都以0来替换
    plt.colorbar()
    plt.xlabel(r'$logMh[M_\odot/h]$')
    plt.ylabel(r'$g-r$')
    #plt.savefig('Hist2d-3',dpi=600)
    plt.show()  
    
    ###调用拟合函数，拟和分布曲线
    import handy
    ##观察 mstar,mh-con 之间的关系
    plt.scatter(mstar,main_halo,s=0.5,c=color1,marker='o',cmap='rainbow',vmin=np.min(color1),
                vmax=1.0,alpha=0.2)
    plt.colorbar(label='con')  
    plt.xlabel(r'$M_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$Mh[M_\odot/h]$')
    plt.title('Mstar-Mh-con')
    handy.compare(mstar,main_halo)
    #plt.savefig('mstar-Mh-con',dpi=600)
    plt.show()

    ###观察 mh，con-color 之间的关系之间的关系
    plt.scatter(main_halo,con,s=0.5,c=color2,marker='o',cmap='rainbow',vmin=np.min(color2),
                vmax=np.max(color2),alpha=0.2)
    plt.colorbar(label='g-r')
    plt.xlabel(r'$Mh[M_\odot/h]$')
    plt.ylabel('c')
    plt.yscale('log')
    plt.title('Mh-con-color')
    handy.compare(main_halo,con)
    #plt.savefig('Mh-con-color',dpi=600)
    plt.show()
    
    ##观察 mstar,con-color 之间的关系
    plt.scatter(mstar,con,s=0.5,c=color2,marker='o',cmap='rainbow',vmin=np.min(color2),
                vmax=np.max(color2),alpha=0.2)
    plt.colorbar(label='g-r')  
    plt.xlabel(r'$M\ast[M_\odot/h^2]$') 
    plt.ylabel('c')
    plt.yscale('log')
    plt.title('Mstar-con-color')
    handy.compare(mstar,con)
    #plt.savefig('Mstar-con-color',dpi=600)    
    plt.show()

    ###观察 mh-color
    plt.scatter(main_halo,gcolor,s=0.5,c=color2,marker='o',cmap='rainbow',vmin=np.min(color2),
                vmax=np.max(color2),alpha=0.2)
    plt.colorbar(label='g-r')  
    plt.xlabel(r'$Mh[M_\odot/h]$')
    plt.ylabel('g-r')
    plt.title('Mh-color')
    handy.compare(main_halo,gcolor)
    #plt.savefig('Mh-color',dpi=600)
    plt.show()
    
    ###观察 mstar-color
    plt.scatter(mstar,gcolor,s=0.5,c=color2,marker='o',cmap='rainbow',vmin=np.min(color2),
                vmax=np.max(color2),alpha=0.2)
    plt.colorbar(label='g-r')  
    plt.xlabel(r'$M\ast[M_\odot/h^2]$')
    plt.ylabel('g-r')
    plt.title('Mstar-color')
    handy.compare(mstar,gcolor)
    #plt.savefig('Mstar-color',dpi=600)
    plt.show()
       
    ##观察 mstar,mh-color 之间的关系
    plt.scatter(mstar,main_halo,s=0.5,c=color2,marker='o',cmap='rainbow',vmin=np.min(color2),
                vmax=np.max(color2),alpha=0.2)
    plt.colorbar(label='g-r')  
    plt.xlabel(r'$M_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$Mh[M_\odot/h]$')
    plt.title('Mstar-Mh-color')
    handy.compare(mstar,main_halo)
    #plt.savefig('mstar-Mh-color',dpi=600)
    plt.show()   
    
    '''
    ##下面取出坐标位置,观察星系-颜色的空间分布
    Position1 = use_data['f']
    pos1 = np.array(Position1[ix])
    Position2 = use_data['g']
    pos2 = np.array(Position2[ix])
    Position3 = use_data['h']
    pos3 = np.array(Position3[ix])
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pos1,pos2,pos3,s=0.5,c=color2,marker='o',cmap='rainbow',vmin=np.min(color2),
                vmax=np.max(color2),alpha=0.2)
    #plt.savefig('3D-distribution-color',dpi=600)
    '''
#   raise
##这句表示节点终段，用于调试程序（raise）
    out_array = [main_halo, mstar, gcolor, con, gcolor, color2, color1]
    return out_array
#if __name__ == "__main__":
def run_control(T):
    #mockfile = '/Users/ying/Data/ihodmock/standard/iHODcatalog_bolshoi.h5'
    #改为本机绝对路径
    #compare_data(C=True)
    mockfile = 'D:/Python1/pydocument/seniorproject_quenching2/practice/iHODcatalog_bolshoi.h5'
    plot_mock_hmf(mockfile)
    return
#run_control(T=True)
