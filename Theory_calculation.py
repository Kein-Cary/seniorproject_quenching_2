##### 这部分脚本用于从理论上给出关于前面mock数据的统计量的分析
##### 主要的数据来源：利用mock的数据，结合SHMR关系，从mock的mstar求出\
### 在相应理论下的mh,在计算的时候，需要用的是反函数。此时需要利用mock给的数据中\
#### mh的部分计算出可用的mstar
####链接路径引入
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
##件的路径具体到文件夹
###库函数调用
import os.path
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats as st
import matplotlib.gridspec as gridspec
import pandas as pa
from scipy import interpolate as sinter
###mock_相关脚本调用，主要用于导入数据
from mock_data_reshow import read_mock_hmf
### section1：导入数据产生需要使用的四个数组：mock_mstar——theory_mh,利用此关系反插的mock_mh——theory_mstar
def doload_mock_data(tt):
    mockfile = 'D:/Python1/pydocument/seniorproject_quenching2/practice/iHODcatalog_bolshoi.h5'
    Mh_arr, dndlnMh_arr, use_data = read_mock_hmf(mockfile, mmin=1.e9, mmax=1.e16, 
                                                  nmbin=101, h=0.701, rcube=250.0)  
    return use_data
def bins_dic(tt):
###该部分需要注意mass function的求解仅仅归一化是不够的,还要注意单位转化
###单位转化过程中mass以Msolar/h为单位,长度以Mpc/h为单位,在模拟数据中盒子边长250Mpc
    use_data = doload_mock_data(tt=True)
    N = np.array(use_data['b'].shape)
    print('size_con=',N)
    ###查看数据量：529308*8~=4000000
    ###把主晕取出来,并把主晕对应的物理量也取出来
    ix = use_data['d']>0
    _halos = use_data['d']
    ##取出主晕
    main_halo = np.array(_halos[ix])
    ##下面把与主晕有关的物理量取出来
    #g_color = use_data['c']
    ##取出主晕下星系颜色
    #gcolor = np.array(g_color[ix])
    ###尝试读取数据的处理
    M_star = use_data['e']
    ##取出恒星质量
    mstar = np.array(M_star[ix])
    ###求出中央星系的比例
    n = np.array(main_halo.shape)
    print('size_mainhalo=',n)
    frac_ = n[0]/N[0]
    print('f_c=',frac_)
    #con_c = use_data['b']
    #con = np.array(con_c[ix])
    ###盒子长度
    #h = 0.7
    #L = 250*h##单位Mpc/h
    L = 1
###下面为质量函数
####mh-function
    #value_mh = plt.hist(main_halo,101,normed=True)
    #plt.show()
    value_mh = st.binned_statistic(main_halo,main_halo,statistic='count',bins=100)
    f_N_mh = np.array(value_mh[0])
    x_N_mh = np.array(value_mh[1])
    #print(np.sum(value_mh[0]*(x_N_mh[2]-x_N_mh[1])))
    S0 = np.sum(value_mh[0]*(x_N_mh[2]-x_N_mh[1]))
    media_mh = np.zeros(value_mh[0].size,dtype=np.float)
    dn_dlgMh = np.zeros(value_mh[0].size,dtype=np.float)
    for k in range(1,value_mh[0].size):
            media_mh[k] = (x_N_mh[k]+x_N_mh[k-1])/2
            dn_dlgMh[k] = f_N_mh[k]/(x_N_mh[1]-x_N_mh[0])
    media_mh[0] = media_mh[1]
    dn_dlgMh[0] = f_N_mh[0]/(x_N_mh[1]-x_N_mh[0])
    dn_dlgMh = dn_dlgMh*np.log(10)/L**3
    f_N_mh1 = f_N_mh/S0
    p_lgMh = f_N_mh1###已完成归一化
####mstar-function    
    #value_ms = plt.hist(mstar,101,normed=True)
    #plt.show()
    value_ms = st.binned_statistic(mstar,mstar,statistic='count',bins=100)
    f_N_ms = np.array(value_ms[0])
    x_N_ms = np.array(value_ms[1])
    #print(np.sum(value_ms[0]*(x_N_ms[2]-x_N_ms[1])))
    S1 = np.sum(value_ms[0]*(x_N_ms[2]-x_N_ms[1]))
    media_ms = np.zeros(value_ms[0].size,dtype=np.float)
    dn_dlgms = np.zeros(value_ms[0].size,dtype=np.float)
    for k in range(1,value_ms[0].size):
            media_ms[k] = (x_N_ms[k]+x_N_ms[k-1])/2
            dn_dlgms[k] = f_N_ms[k]/(x_N_ms[1]-x_N_ms[0])
    media_ms[0] = media_ms[1]
    dn_dlgms[0] = f_N_ms[0]/(x_N_ms[1]-x_N_ms[0])
    dn_dlgms = dn_dlgms*np.log(10)/L**3
    f_N_ms1 = f_N_ms/S1
    p_lgms = f_N_ms1##已完成归一化
    print('load mass function successfully')
    return media_mh,dn_dlgMh,media_ms,dn_dlgms,p_lgMh,p_lgms
def mass_function_use(uu):
    media_mh,dn_dlgMh,media_ms,dn_dlgms,p_lgMh,p_lgm = bins_dic(tt=True)
    plt.plot(media_mh,dn_dlgMh,label='dn_dlgMh')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.ylabel(r'$dn/dlgM_h$')
    plt.grid()
    plt.title('Mh_fun')
    #plt.savefig('fun_Mh',dpi=600)
    plt.show()
    plt.plot(media_ms,dn_dlgms,label=r'$dn_dlgM_\ast$')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.ylabel(r'$dn/dlgM_\ast$')
    plt.grid()
    plt.title('stellarmass_fun')
    #plt.savefig('fun_Ms',dpi=600)
    plt.show() 
    return
def the_probability(pp):
    ####自假设数据
    mhalo = np.logspace(10,15,1000)
    ms = np.logspace(5,12,950)
    M0 = 2*10**10###单位是Msolar/h^2
    M1 = 1.3*10**12###单位是Msolar/h
    belta = 0.33
    sigma = 0.42
    gamma = 1.21
    ##下面计算为了和模拟数据保持一直,把SHMR的质量转为以10为底的对数
    Theory_Mh = 10**(np.log10(M1)+belta*(np.log10(ms)-np.log10(M0))+\
    ((ms/M0)**sigma/(1+(ms/M0)**(-gamma))-1/2))
    ###反过来求出需要使用的理论上的mstar
    Theory_mstar = np.interp(mhalo,Theory_Mh,ms)
    ita = -0.04
    sigma_ms = np.zeros(len(mhalo),dtype=np.float)
    for k in range(0,len(mhalo)):
        if mhalo[k]<M1:
            sigma_ms[k] = 0.5
        else:
            sigma_ms[k] = 0.5+ita*(np.log10(mhalo[k])-np.log10(M1))
    N_ms_Mh1 = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    N_ms_Mh2 = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    N_ms_Mh3 = np.zeros((len(mhalo),len(ms)),dtype=np.float)    
    for k in range(0,len(mhalo)):
        N_ms_Mh1[k,:] = (1/(sigma_ms[k]*np.sqrt(2*np.pi)))*np.exp(-(np.log(ms)\
                    -np.log(Theory_mstar[k]))**2/(2*sigma_ms[k]**2))
        ####下面一个是L11 function的结果
        N_ms_Mh2[k,:] = (1/(np.log(10)*sigma_ms[k]*np.sqrt(2*np.pi)))*np.exp(-(np.log10(ms)\
                     -np.log10(Theory_mstar[k]))**2/(2*sigma_ms[k]**2))
        ####为了对比，假设一个N_ms_Mh3
        N_ms_Mh3[k,:] = (1/(sigma_ms[k]*np.sqrt(2*np.pi)))*10**(-(np.log(ms)\
                    -np.log(Theory_mstar[k]))**2/(2*sigma_ms[k]**2)) 
    return mhalo,ms,N_ms_Mh1,N_ms_Mh2,N_ms_Mh3,Theory_Mh,Theory_mstar,sigma_ms

def figure_1(a):
    mhalo,ms,N_ms_Mh1,N_ms_Mh2,N_ms_Mh3,Theory_Mh,Theory_mstar,sigma_ms = \
    the_probability(pp=True)
    plt.loglog(ms,Theory_Mh,label=r'$f_{SHMR} ^{-1}$')
    plt.legend(loc=2)
    #plt.savefig('ms-Mh',dpi=600)
    plt.show()
    plt.errorbar(mhalo,Theory_mstar,yerr=[sigma_ms,sigma_ms],fmt="r^-",linewidth=0.5,
                                elinewidth=0.5,ecolor='r',capsize=1,capthick=1,label='SHMR')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.legend(loc=4)
    #plt.savefig('Mh-Ms',dpi=600)
    plt.show()
    #plt.plot(np.log10(mhalo),np.log10(Theory_mstar)/np.log10(mhalo),'r')
    #plt.show()
    #plt.plot(np.log10(mhalo),np.log10(Theory_mstar/mhalo),'b')
    #plt.xlabel(r'$M_h [M_\odot h^{-1}]$')
    #plt.ylabel(r'$M_\ast /M_h [h-1]$')
    #plt.show()
    plt.plot(np.log10(mhalo),np.log10(Theory_mstar),'r')
    plt.fill_between(np.log10(mhalo),np.log10(Theory_mstar)-sigma_ms,np.log10(Theory_mstar)+sigma_ms,facecolor='g',alpha=0.2)
    plt.xlabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.legend(loc=4)
    #plt.savefig('Mh-Ms-fill',dpi=600)
    plt.show()
    plt.pcolormesh(np.log10(mhalo),np.log10(ms),N_ms_Mh1.T,cmap='rainbow',vmin=0.01,vmax=1.5,alpha=1,
              norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('Mapping_2015')
    #plt.savefig('Theory_DN_function_map2015',dpi=600)
    plt.show()    
    plt.pcolormesh(np.log10(mhalo),np.log10(ms),N_ms_Mh2.T,cmap='rainbow',vmin=0.01,vmax=1,alpha=1,
               norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('L11_2011')
    #plt.savefig('Theory_DN_function_L11',dpi=600)
    plt.show()
    plt.pcolormesh(np.log10(mhalo),np.log10(ms),N_ms_Mh3.T,cmap='rainbow',vmin=0.01,vmax=1,alpha=1,
               norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('Assumption')
    #plt.savefig('Theory_DN_function_Assumption',dpi=600)
    plt.show()
    return

def Theory_fun3(ff3):
####对mhalo,ms同时做限制的情况
    media_mh,dn_dlgMh,media_ms,dn_dlgms,p_lgMh,p_lgms = bins_dic(tt=True)
    mhalo,ms,N_ms_Mh1,N_ms_Mh2,N_ms_Mh3,Theory_Mh,Theory_mstar,sigma_ms =\
    the_probability(pp=True)
####下面计算p(mstar,mh),即为联合分布概率P_joint,注意需要划分质量区间——对引入的质量函数的适应
    #ng=1##ng表示宇宙平均星系数密度
    x1 = np.linspace(np.min(media_mh),np.max(media_mh),len(mhalo))
    x2 = np.linspace(np.min(media_ms),np.max(media_ms),len(ms))
    f1 = sinter.interpolate.interp1d(media_mh,p_lgMh)
    f2 = sinter.interpolate.interp1d(media_ms,p_lgms)
    plgMh = f1(x1)
    plgms = f2(x2)
    _halo = x1
    ms_use = x2    
    f3 = sinter.interpolate.interp2d(ms,mhalo,N_ms_Mh1)
    N_ms_Mh = f3(10**x2,10**x1)
    pmh = plgMh/(np.log(10)*10**_halo)
    pms = plgms/(np.log(10)*10**ms_use)
####调用mass_function部分可以看到实际上pmh并不连续,因此考虑如下质量函数
    data_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/data/M16'
    fname = os.path.join(data_path,'hmf.dat')
    lnMh_arr, dndlnMh_arr = np.genfromtxt(fname, unpack=True)
    d_lnMh = (np.max(lnMh_arr)-np.min(lnMh_arr))/len(lnMh_arr)
    N_lnMh = dndlnMh_arr*d_lnMh
    S_lnMh = np.sum(dndlnMh_arr*d_lnMh)
    p_lnMh = N_lnMh/S_lnMh
    f4 = sinter.interpolate.interp1d(lnMh_arr,p_lnMh)
    ###插值计算概率时需要注意概率和数密度之间的关系
    plnMh = f4(np.log(10)*_halo)
    pmh1 = plnMh/(10**_halo)
    #下面把pmh,pms归一化
    A = np.sum((10**_halo[-1]-10**_halo[0])*pmh/len(_halo))
    B = np.sum((10**ms_use[-1]-10**ms_use[0])*pms/len(ms_use))
    C = np.sum((10**_halo[-1]-10**_halo[0])*pmh1/len(_halo))
    pmh = pmh/A
    pms = pms/B
    pmh1 = pmh1/C   
    ##求解联合分布概率
    f5 = sinter.interpolate.interp1d(media_mh,dn_dlgMh)
    dn_dMh = f5(x1)
    dn_dMh = dn_dMh/(np.log(10)*10**x1)
    ##对N_ms_Mh也做归一化
    N_ms_Mh_1 = np.zeros((len(_halo),len(ms_use)),dtype = np.float)
    for k in range(len(_halo)):
        ss = np.sum(N_ms_Mh[k,:]*(10**ms_use[-1]-10**ms_use[0])/len(ms_use))
        N_ms_Mh_1[k,:] = N_ms_Mh[k,:]/ss 
    ##求联合分布
    p_joint = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    p_joint1 = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(len(mhalo)):
        #p_joint[k,:] = (np.log10(np.e)/(10**ms_use*ng))*N_ms_Mh[k,:]*dn_dMh[k] 
        ###HOD模型下，联合概率密度分布表达为上式
        p_joint[k,:] = N_ms_Mh_1[k,:]*pmh[k] 
        p_joint1[k,:] = N_ms_Mh_1[k,:]*pmh1[k] 
    P_Joint3 = p_joint
    P_Joint3_1 = p_joint1
    #print(np.max(P_Joint3))
    #print(np.max(P_Joint3_1))    
####下面求条件概率p_Mh_ms代替p(Mh|ms),p_ms_Mh代替p(ms|Mh)
    p_Mh_ms = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    p_Mh_ms1 = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    p_ms_Mh = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    p_ms_Mh1 = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(len(mhalo)):
        p_ms_Mh[k,:] = P_Joint3[k,:]/pmh[k]
        p_ms_Mh1[k,:] = P_Joint3_1[k,:]/pmh1[k]
    for k in range(len(ms)):
        p_Mh_ms[:,k] = P_Joint3[:,k]/pms[k]
        p_Mh_ms1[:,k] = P_Joint3_1[:,k]/pms[k]
    p_Mh_ms[np.isinf(p_Mh_ms)]=0
    p_Mh_ms1[np.isinf(p_Mh_ms1)]=0
    p_ms_Mh[np.isinf(p_ms_Mh)]=0
    p_ms_Mh1[np.isinf(p_ms_Mh1)]=0
    p_Mh_ms[np.isnan(p_Mh_ms)]=0
    p_Mh_ms1[np.isnan(p_Mh_ms1)]=0
    p_ms_Mh[np.isnan(p_ms_Mh)]=0
    p_ms_Mh1[np.isnan(p_ms_Mh1)]=0
    return _halo,ms_use,N_ms_Mh,p_joint,\
            p_Mh_ms,p_ms_Mh,P_Joint3,\
            P_Joint3_1,p_ms_Mh1,p_Mh_ms1,p_joint1

def figure_2(b):
    _halo,ms_use,N_ms_Mh,p_joint,p_Mh_ms,p_ms_Mh,P_Joint3,P_Joint3_1,p_ms_Mh1,p_Mh_ms1,p_joint1\
    = Theory_fun3(ff3=True) 
    plt.pcolormesh(np.log10(10**_halo),np.log10(10**ms_use),N_ms_Mh.T,cmap='rainbow',
                   vmin=0.01,vmax=1,alpha=1,norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('fit_L11_2011') 
    #plt.savefig('Theory_DN_function_contrain_mh_ms',dpi=600)
    plt.show()
    #print(np.min(p_joint))
    #print(np.max(p_joint))
    plt.pcolormesh(np.log10(10**_halo),np.log10(10**ms_use),p_joint.T,cmap='rainbow',
                   vmin=1e-33,vmax=np.max(p_joint),alpha=1,norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$p(M_\ast - M_h)$')
    plt.ylabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.xlabel(r'$M_h[M_\odot h^{-1}]$')
    plt.title('P')
    #plt.savefig('Theory_DN_probablity',dpi=600)
    plt.show()
    ###下面计算p(lgMh|M*),p(lgM*|Mh)并显示
    p_lgMh_ms = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    plgMhms = np.zeros((len(_halo),len(ms_use)),dtype=np.float)##归一化表示   
    p_lgms_Mh = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    plgmsMh = np.zeros((len(_halo),len(ms_use)),dtype=np.float)##归一化表示    
    for k in range(len(ms_use)):     
        p_lgMh_ms[:,k] = np.log(10)*10**_halo*p_Mh_ms[:,k]
        min_mh = np.min(p_lgMh_ms[:,k])
        max_mh = np.max(p_lgMh_ms[:,k])
        plgMhms[:,k] = (p_lgMh_ms[:,k]-min_mh)/(max_mh -min_mh)
    for k in range(len(_halo)):
        p_lgms_Mh[k,:] = np.log(10)*10**ms_use*p_ms_Mh[k,:]
        min_ms = np.min(p_lgms_Mh[k,:])
        max_ms = np.max(p_lgms_Mh[k,:])
        plgmsMh[k,:] = (p_lgms_Mh[k,:]-min_ms)/(max_ms-min_ms)
    gs1 = mpl.gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs1[0,0])
    ax1.plot(_halo,p_lgMh_ms[:,949])
    plt.yscale('log')
    plt.xlabel(r'$M_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$P(lgM_h-M_\ast)$')
    #plt.show() 
    ax2 = plt.subplot(gs1[1,0])
    ax2.plot(ms_use,p_lgms_Mh[850,:])
    plt.yscale('log')
    plt.xlabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$P(lgM_\ast-M_h)$')  
    #plt.show()
    ax3 = plt.subplot(gs1[0,1])
    ax3.plot(_halo,plgMhms[:,949])
    plt.xlabel(r'$M_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$P(lgM_h-M_\ast)$')
    #plt.yscale('log')
    #plt.show()
    ax4 = plt.subplot(gs1[1,1])
    ax4.plot(ms_use,plgmsMh[850,:])
    #plt.yscale('log')
    plt.xlabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$P(lgM_\ast-M_h)$')
    plt.tight_layout()
    #plt.savefig('Theory-Condition-Probability',dpi=600)
    plt.show()
####下面为连续质量函数的情况#####
    #print(np.min(p_joint1))
    #print(np.max(p_joint1))
    plt.pcolormesh(np.log10(10**_halo),np.log10(10**ms_use),p_joint1.T,cmap='rainbow',
                   vmin=1e-32,vmax=np.max(p_joint1),alpha=1,norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$p(M_\ast - M_h)$')
    plt.ylabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.xlabel(r'$M_h[M_\odot h^{-1}]$')
    plt.title('P')
    #plt.savefig('Theory_DN_probablity_continue',dpi=600)
    plt.show()
    ###下面计算p(lgMh|M*),p(lgM*|Mh)并显示
    p_lgMh_ms1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    plgMhms1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)##归一化表示   
    p_lgms_Mh1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    plgmsMh1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)##归一化表示    
    for k in range(len(ms_use)):
        p_lgMh_ms1[:,k] = np.log(10)*10**_halo*p_Mh_ms1[:,k]
        min_mh = np.min(p_lgMh_ms1[:,k])
        max_mh = np.max(p_lgMh_ms1[:,k])
        plgMhms1[:,k] = (p_lgMh_ms1[:,k]-min_mh)/(max_mh-min_mh)
    for k in range(len(_halo)):
        p_lgms_Mh1[k,:] = np.log(10)*10**ms_use*p_ms_Mh1[k,:]
        min_ms = np.min(p_lgms_Mh1[k,:])
        max_ms = np.max(p_lgms_Mh1[k,:])
        plgmsMh1[k,:] = (p_lgms_Mh1[k,:]-min_ms)/(max_ms-min_ms)
    gs2 = mpl.gridspec.GridSpec(2,2)
    bx1 = plt.subplot(gs2[0,0])
    bx1.plot(_halo,p_lgMh_ms1[:,949])
    plt.yscale('log')
    plt.xlabel(r'$M_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$P(lgM_h-M_\ast)$')
    #plt.show()
    bx2 = plt.subplot(gs2[1,0])
    bx2.plot(ms_use,p_lgms_Mh1[850,:])
    plt.yscale('log')
    plt.xlabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$P(lgM_\ast-M_h)$')  
    #plt.show()
    bx3 = plt.subplot(gs2[0,1])
    bx3.plot(_halo,plgMhms1[:,949])
    plt.xlabel(r'$M_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$P(lgM_h-M_\ast)$')
    #plt.yscale('log')
    #plt.show()
    bx4 = plt.subplot(gs2[1,1])
    bx4.plot(ms_use,plgmsMh1[850,:])
    #plt.yscale('log')
    plt.xlabel(r'$M_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$P(lgM_\ast-M_h)$')
    plt.tight_layout()
    #plt.savefig('Theory-Condition-Probability-ln',dpi=600)
    plt.show()
    return

def R_Mh_Ms(c):
    _halo,ms_use,N_ms_Mh,p_joint,p_Mh_ms,p_ms_Mh,P_Joint3,P_Joint3_1,p_ms_Mh1,p_Mh_ms1,p_joint1\
    = Theory_fun3(ff3=True)
####标号1的表示连续质量函数的计算结果
####下面计算Mh关于Ms的变换关系
    #print(np.max(_halo))
    #print(np.min(_halo))
    ###先把p_Mh_ms归一化
    p_Mh_ms_ = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        ss = np.sum(p_Mh_ms[:,k]*(10**_halo[-1]-10**_halo[0])/len(_halo))
        p_Mh_ms_[:,k] = p_Mh_ms[:,k]/ss 
    plt.plot(10**_halo,p_Mh_ms_[:,949])
    #plt.yscale('log')
    plt.xscale('log')
    plt.show()
    ###在把p_Mh_ms归一到0~1之间
    pMhms = np.zeros((len(_halo),len(ms_use)),dtype=np.float) 
    for k in range(len(ms_use)):
        min_mh = np.min(p_Mh_ms[:,k])
        max_mh = np.max(p_Mh_ms[:,k])
        pMhms[:,k] = (p_Mh_ms[:,k]-min_mh)/(max_mh-min_mh)
    plt.plot(10**_halo,pMhms[:,949])
    plt.xscale('log')
    plt.show()
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    dmh = (10**_halo[-1]-10**_halo[0])/len(_halo)
    Mh_ms = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        #dMh = (np.log(10)*10**_halo)*(_halo[-1]-_halo[0])/len(_halo)
        Mh_ms[k] = np.sum(p_Mh_ms[:,k]*10**_halo*dmh)/(np.sum(p_Mh_ms[:,k]*dmh))
    ###下面求errorbar
    Mh_err = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = p_Mh_ms[:,k]
        F_m1 = np.zeros(len(mh_err),dtype=np.float)
        F_m1[0] = 0
        for t in range(len(_halo)):
            F_m1[t] = F_m1[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err1 = np.interp(0.1585,F_m1,mh_err)-np.log10(Mh_ms[k])
        va_err2 = np.interp(0.8415,F_m1,mh_err)-np.log10(Mh_ms[k])
        Mh_err[k,:] = np.array([va_err1,va_err2]) 
    ###尝试求解<lnMH|m*>
    lnMh_ms = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        #dMh = (np.log(10)*10**_halo)*(_halo[-1]-_halo[0])/len(_halo)
        lnMh_ms[k] = np.sum(p_Mh_ms[:,k]*np.log(10**_halo)*dmh)/(np.sum(p_Mh_ms[:,k]*dmh))
    lnMh_ms = lnMh_ms/np.log(10)
    ###下面求errorbar
    lnMh_err = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = p_Mh_ms[:,k]
        F_m2 = np.zeros(len(mh_err),dtype=np.float)
        F_m2[0] = 0
        for t in range(len(_halo)):
            F_m2[t] = F_m2[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err3 = np.interp(0.1585,F_m2,mh_err)-lnMh_ms[k]
        va_err4 = np.interp(0.8415,F_m2,mh_err)-lnMh_ms[k]
        lnMh_err[k,:] = np.array([va_err3,va_err4]) 
    Mh_err[np.isnan(Mh_err)]=0
    Mh_err[np.isinf(Mh_err)]=0
    lnMh_err[np.isnan(lnMh_err)]=0
    lnMh_err[np.isinf(lnMh_err)]=0
    return ms_use,_halo,Mh_ms,Mh_err,lnMh_ms,lnMh_err
def fig_R_Mh_Ms(d):
    ms_use,_halo,Mh_ms,Mh_err,lnMh_ms,lnMh_err = R_Mh_Ms(c=True) 
    plt.plot(ms_use,np.log10(Mh_ms))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.show()
    plt.errorbar(ms_use,np.log10(Mh_ms),yerr=abs(Mh_err.T),fmt="k^-",linewidth=0.5,
                 elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$< M_h-M_\ast >$')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory-Mh-Ms',dpi=600)
    plt.show()
    plt.plot(ms_use,lnMh_ms)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lg(lnM_h[M_\odot h^{-1}])$')
    plt.show() 
    plt.errorbar(ms_use,lnMh_ms,yerr=abs(lnMh_err.T),fmt="k^-",linewidth=0.5,
                 elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$< lnM_h-M_\ast >$')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lg(lnM_h[M_\odot h^{-1}])$')
    plt.legend(loc=2)
    #plt.savefig('Theory-lnMh-Ms',dpi=600)
    plt.show()
    plt.figure()
    plt.plot(ms_use,np.log10(Mh_ms),'r-',label=r'$<M_h-M_\ast>$')
    plt.fill_between(ms_use,np.log10(Mh_ms)+Mh_err[:,0],np.log10(Mh_ms)+Mh_err[:,1],
                     facecolor='r',alpha=0.2)
    plt.plot(ms_use,lnMh_ms,'b--',label=r'$<lnM_h-M_\ast>$')
    plt.fill_between(ms_use,lnMh_ms+lnMh_err[:,0],lnMh_ms+lnMh_err[:,1],
                     facecolor='b',alpha=0.2)
    plt.legend(loc=4)
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lg(lnM_h[M_\odot h^{-1}])$')
    #plt.savefig('Theory-Mh_ms-comparation',dpi=600)
    plt.show()
    return
#########################
#下面分析红蓝两个序列的Mh-M*质量关系
def func_fred(f):
    _halo,ms_use,N_ms_Mh,p_joint,p_Mh_ms,p_ms_Mh,P_Joint3,P_Joint3_1,p_ms_Mh1,p_Mh_ms1,p_joint1\
    = Theory_fun3(ff3=True)
##考虑恒星质量是主要quenching的原因：f_red_ms,ms_q表示quenching的临界质量
##下面的代码中Mh,Ms中M大写的表示条件概率,小写的msmh连在一起表示联合分布概率,分开的表示边界分布
##此外,标号1的表示以恒星质量为主要quenching机制,标号2的表示以暗晕质量为机制
    ms_q = 10.55#单位为 Msolar h^-2
    miu_ms = 0.69
    f_red_ms = 1-np.exp(-(10**ms_use/10**ms_q)**miu_ms)
    dms = (10**ms_use[-1]-10**ms_use[0])/len(ms_use)
    S1 = np.zeros(len(_halo),dtype=np.float)
    for k in range(len(_halo)):
        S1[k] = np.sum(f_red_ms*p_joint[k,:]*dms)    
    dmh = (10**_halo[-1]-10**_halo[0])/len(_halo)
    S2 = np.sum(S1*dmh)
    tot_f_red_ms = S2
    ###for red sequence,the joint distribution as flow
    p_red_msmh1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(_halo)):
        p_red_msmh1[k,:] = p_joint[k,:]*f_red_ms/tot_f_red_ms
    #求恒星质量的边界分布
    p_red_ms1 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        p_red_ms1[k] = np.sum(p_red_msmh1[:,k]*dmh)
    #the condition distribution:p_red(Mh|m*) as red_p_Mh_ms
    red_p_Mh_ms1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        red_p_Mh_ms1[:,k] = p_red_msmh1[:,k]/p_red_ms1[k]
    red_p_Mh_ms1[np.isnan(red_p_Mh_ms1)]=0
    red_p_Mh_ms1[np.isinf(red_p_Mh_ms1)]=0
    ###对条件概率做归一化
    red_pMhms1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)   
    for k in range(len(ms_use)):
        s = np.sum(red_p_Mh_ms1[:,k]*dmh)
        red_pMhms1[:,k] = red_p_Mh_ms1[:,k]/s
    #############
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    red_Mh_ms1 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        red_Mh_ms1[k] = np.sum(red_p_Mh_ms1[:,k]*10**_halo*dmh)/(np.sum(red_p_Mh_ms1[:,k]*dmh))
    ###下面求errorbar
    '''
    red_Mh_err1 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):      
        mh_err = _halo
        p_m = red_p_Mh_ms1[:,k]
        mh = np.sum(dMh*p_m*(10**mh_err-red_Mh_ms1[k])**2)/np.sum(p_m*dMh)
        mh = np.sqrt(mh)
        red_Mh_err1[k] = np.log10(mh)
    '''
    red_Mh_err1 = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = red_p_Mh_ms1[:,k]
        F_m1 = np.zeros(len(mh_err),dtype=np.float)
        F_m1[0] = 0
        for t in range(len(_halo)):
            F_m1[t] = F_m1[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err1 = np.interp(0.1585,F_m1,mh_err)-np.log10(red_Mh_ms1[k])
        va_err2 = np.interp(0.8415,F_m1,mh_err)-np.log10(red_Mh_ms1[k])
        red_Mh_err1[k,:] = np.array([va_err1,va_err2]) 
    ####此时对blue sequence的求解
    S3 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        S3[k] = np.sum(p_joint[:,k]*dmh)
    P_tot1 = np.sum(S3*dms)
    #求解该情况下联合分布概率密度
    p_blue_msmh1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    f_blue_ms = 1 - f_red_ms
    for k in range(len(_halo)):
        p_blue_msmh1[k,:] = f_blue_ms*p_joint[k,:]/(P_tot1 - tot_f_red_ms)
    #求解恒星质量的边界分布
    p_blue_ms1 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        p_blue_ms1[k] = np.sum(p_blue_msmh1[:,k]*dmh)
    #the condition distribution:p_blue(Mh|m*) as blue_p_Mh_ms
    blue_p_Mh_ms1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        blue_p_Mh_ms1[:,k] = p_blue_msmh1[:,k]/p_blue_ms1[k]
    blue_p_Mh_ms1[np.isnan(blue_p_Mh_ms1)]=0
    blue_p_Mh_ms1[np.isinf(blue_p_Mh_ms1)]=0
    #对条件概率归一化
    blue_pMhms1 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        s = np.sum(blue_p_Mh_ms1[:,k]*dmh)
        blue_pMhms1[:,k] = blue_p_Mh_ms1[:,k]/s
    #########
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    blue_Mh_ms1 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        blue_Mh_ms1[k] = np.sum(blue_p_Mh_ms1[:,k]*10**_halo*dmh)/(np.sum(blue_p_Mh_ms1[:,k]*dmh))
    ###下面求errorbar
    '''
    blue_Mh_err1 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):      
        mh_err = _halo
        p_m = blue_p_Mh_ms1[:,k]
        mh = np.sum(dMh*(10**mh_err-blue_Mh_ms1[k])**2*p_m)/np.sum(p_m*dMh)
        mh = np.sqrt(mh)
        blue_Mh_err1[k] = np.log10(mh)
    '''
    blue_Mh_err1 = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = blue_p_Mh_ms1[:,k]
        F_m2 = np.zeros(len(mh_err),dtype=np.float)
        F_m2[0] = 0
        for t in range(len(_halo)):
            F_m2[t] = F_m2[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err3 = np.interp(0.1585,F_m2,mh_err)-np.log10(blue_Mh_ms1[k])
        va_err4 = np.interp(0.8415,F_m2,mh_err)-np.log10(blue_Mh_ms1[k])
        blue_Mh_err1[k,:] = np.array([va_err3,va_err4]) 
####################
##考虑暗晕质量是主要的quenching的原因：f_red_mh,mh_q表示quenching的临界质量
    #mh_q = 13.5
    #miu_mh = 1.25###for M16 comparation
    mh_q = 11.25
    miu_mh = 0.6###for mock data
    f_red_mh = 1-np.exp(-(10**_halo/10**mh_q)**miu_mh)
    S4 = np.zeros(len(_halo),dtype=np.float)
    for k in range(len(ms_use)):
        S4[k] = np.sum(f_red_mh*p_joint[:,k]*dmh)  
    tot_f_red_mh = np.sum(S4*dms)
    #for red sequence,the joint distribution as flow 
    p_red_msmh2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        p_red_msmh2[:,k] = p_joint[:,k]*f_red_mh/tot_f_red_mh
    #求解横行质量的边界分布
    p_red_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        p_red_ms2[k] = np.sum(p_red_msmh2[:,k]*dmh)
    #the condition distribution:p_red(Mh|m*) as red_p_Mh_ms
    red_p_Mh_ms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        red_p_Mh_ms2[:,k] = p_red_msmh2[:,k]/p_red_ms2[k]
    red_p_Mh_ms2[np.isnan(red_p_Mh_ms2)]=0
    red_p_Mh_ms2[np.isinf(red_p_Mh_ms2)]=0
    ###对条件概率做归一化
    red_pMhms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)   
    for k in range(len(ms_use)):
        s = np.sum(red_p_Mh_ms2[:,k]*dmh)
        red_pMhms2[:,k] = red_p_Mh_ms2[:,k]/s
    #############
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    red_Mh_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        red_Mh_ms2[k] = np.sum(red_p_Mh_ms2[:,k]*10**_halo*dmh)/(np.sum(red_p_Mh_ms2[:,k]*dmh))
    ###下面求errorbar
    '''
    red_Mh_err2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):      
        mh_err = _halo
        p_m = red_p_Mh_ms2[:,k]
        mh = np.sum(dMh*(10**mh_err-red_Mh_ms2[k])**2*p_m)/np.sum(p_m*dMh)
        mh = np.sqrt(mh)
        red_Mh_err2[k] = np.log10(mh)
    '''
    red_Mh_err2 = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = red_p_Mh_ms2[:,k]
        F_m3 = np.zeros(len(mh_err),dtype=np.float)
        F_m3[0] = 0
        for t in range(len(_halo)):
            F_m3[t] = F_m3[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err5 = np.interp(0.1585,F_m3,mh_err)-np.log10(red_Mh_ms2[k])
        va_err6 = np.interp(0.8415,F_m3,mh_err)-np.log10(red_Mh_ms2[k])
        red_Mh_err2[k,:] = np.array([va_err5,va_err6]) 
    ####此时对blue sequence的求解
    S5 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        S5[k] = np.sum(p_joint[:,k]*dmh)
    P_tot2 = np.sum(S5*dms)
    #求解该情况下联合分布概率密度
    f_blue_mh = 1-f_red_mh
    p_blue_msmh2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        p_blue_msmh2[:,k] = f_blue_mh*p_joint[:,k]/(P_tot2 - tot_f_red_mh)
    #求解恒星质量的边界分布
    p_blue_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        p_blue_ms2[k] = np.sum(p_blue_msmh2[:,k]*dmh)
    #the condition distribution:p_blue(Mh|m*) as blue_p_Mh_ms
    blue_p_Mh_ms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        blue_p_Mh_ms2[:,k] = p_blue_msmh2[:,k]/p_blue_ms2[k]
    blue_p_Mh_ms2[np.isnan(blue_p_Mh_ms2)]=0
    blue_p_Mh_ms2[np.isinf(blue_p_Mh_ms2)]=0
    #对条件概率归一化
    blue_pMhms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        s = np.sum(blue_p_Mh_ms2[:,k]*dmh)
        blue_pMhms2[:,k] = blue_p_Mh_ms2[:,k]/s
    #########
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    blue_Mh_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        blue_Mh_ms2[k] = np.sum(blue_p_Mh_ms2[:,k]*10**_halo*dmh)/(np.sum(blue_p_Mh_ms2[:,k]*dmh))
    ###下面求errorbar
    '''
    blue_Mh_err2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):      
        mh_err = _halo
        p_m = blue_p_Mh_ms2[:,k]
        mh = np.sum(dMh*(10**mh_err-blue_Mh_ms2[k])**2*p_m)/np.sum(p_m*dMh)
        mh = np.sqrt(mh)
        blue_Mh_err2[k] = np.log10(mh)
    '''
    blue_Mh_err2 = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = blue_p_Mh_ms2[:,k]
        F_m4 = np.zeros(len(mh_err),dtype=np.float)
        F_m4[0] = 0
        for t in range(len(_halo)):
            F_m4[t] = F_m4[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err7 = np.interp(0.1585,F_m4,mh_err)-np.log10(blue_Mh_ms2[k])
        va_err8 = np.interp(0.8415,F_m4,mh_err)-np.log10(blue_Mh_ms2[k])
        blue_Mh_err2[k,:] = np.array([va_err7,va_err8]) 
    return f_red_ms,tot_f_red_ms,p_red_msmh1,p_red_ms1,red_p_Mh_ms1,red_pMhms1,red_Mh_ms1,red_Mh_err1,\
           f_blue_ms,p_blue_msmh1,p_blue_ms1,blue_p_Mh_ms1,blue_pMhms1,blue_Mh_ms1,blue_Mh_err1,\
           f_red_mh,tot_f_red_mh,p_red_msmh2,p_red_ms2,red_p_Mh_ms2,red_pMhms2,red_Mh_ms2,red_Mh_err2,\
           f_blue_mh,p_blue_msmh2,p_blue_ms2,blue_p_Mh_ms2,blue_pMhms2,blue_Mh_ms2,blue_Mh_err2,\
           _halo,ms_use
def fig_func_fred(g):
    f_red_ms,tot_f_red_ms,p_red_msmh1,p_red_ms1,red_p_Mh_ms1,red_pMhms1,red_Mh_ms1,red_Mh_err1,\
    f_blue_ms,p_blue_msmh1,p_blue_ms1,blue_p_Mh_ms1,blue_pMhms1,blue_Mh_ms1,blue_Mh_err1,\
    f_red_mh,tot_f_red_mh,p_red_msmh2,p_red_ms2,red_p_Mh_ms2,red_pMhms2,red_Mh_ms2,red_Mh_err2,\
    f_blue_mh,p_blue_msmh2,p_blue_ms2,blue_p_Mh_ms2,blue_pMhms2,blue_Mh_ms2,blue_Mh_err2,\
    _halo,ms_use = func_fred(f=True)
###作图显示两类quenching主导机制下的f_red,f_blue的变化情况
    plt.plot(ms_use,f_red_ms,'r',label=r'$f_{M_\ast}^{red}$')
    plt.plot(ms_use,f_blue_ms,'b',label=r'$f_{M_\ast}^{blue}$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel('fractio-stellar-mass')
    plt.legend(loc=4)
    #plt.savefig('Theory_f_red_stellar_mass',dpi=600)
    plt.show()
    plt.plot(_halo,f_red_mh,'r',label=r'$f_{M_h}^{red}$')
    plt.plot(_halo,f_blue_mh,'b',label=r'$f_{M_h}^{blue}$')
    plt.xlabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.ylabel('fractio-halo-mass')
    plt.legend(loc=4)
    #plt.savefig('Theory_f_red_halo_mass',dpi=600)
    plt.show()
###做图显示stellar mass主导quenching的情况
    plt.plot(ms_use,np.log10(red_Mh_ms1),label=r'$QE-M_\ast-red$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    plt.show()
    plt.errorbar(ms_use,np.log10(red_Mh_ms1),yerr=abs(red_Mh_err1.T),fmt="k^-",linewidth=0.5,
             elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$red < M_h-M_\ast > ms$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory_QE_ms_red',dpi=600)
    plt.show()

    plt.plot(ms_use,np.log10(blue_Mh_ms1),label=r'$QE-M_\ast-blue$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    plt.show()
    plt.errorbar(ms_use,np.log10(blue_Mh_ms1),yerr=abs(blue_Mh_err1.T),fmt="k^-",linewidth=0.5,
             elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$blue < M_h-M_\ast > ms$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory_QE_ms_blue',dpi=600)
    plt.show()
    
    plt.plot(ms_use,np.log10(red_Mh_ms1),'r-',label=r'$QE-M_\ast-red$')
    plt.fill_between(ms_use,np.log10(red_Mh_ms1)+red_Mh_err1[:,0],np.log10(red_Mh_ms1)+red_Mh_err1[:,1],
                     facecolor='r',alpha=0.2)
    plt.plot(ms_use,np.log10(blue_Mh_ms1),'b--',label=r'$QE-M_\ast-blue$')
    plt.fill_between(ms_use,np.log10(blue_Mh_ms1)+blue_Mh_err1[:,0],np.log10(blue_Mh_ms1)+blue_Mh_err1[:,1],
                     facecolor='b',alpha=0.2)
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory_QE_ms_comparation',dpi=600)
    plt.show()
###做图显示halo mass主导quenching的情况
    plt.plot(ms_use,np.log10(red_Mh_ms2),label=r'$QE-M_h-red$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    plt.show()
    plt.errorbar(ms_use,np.log10(red_Mh_ms2),yerr=abs(red_Mh_err2.T),fmt="k^-",linewidth=0.5,
             elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$red < M_h-M_\ast > mh$')    
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory_QE_mh_red',dpi=600)
    plt.show()
    
    plt.plot(ms_use,np.log10(blue_Mh_ms2),label=r'$QE-M_h-blue$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    plt.show()
    plt.errorbar(ms_use,np.log10(blue_Mh_ms2),yerr=abs(blue_Mh_err2.T),fmt="k^-",linewidth=0.5,
             elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$blue < M_h-M_\ast > mh$')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory_QE_mh_blue',dpi=600)
    plt.show() 
    
    plt.plot(ms_use,np.log10(red_Mh_ms2),'r-',label=r'$QE-M_h-red$')
    plt.fill_between(ms_use,np.log10(red_Mh_ms2)+red_Mh_err2[:,0],np.log10(red_Mh_ms2)+red_Mh_err2[:,1],
                     facecolor='r',alpha=0.2)
    plt.plot(ms_use,np.log10(blue_Mh_ms2),'b--',label=r'$QE-M_h-blue$')
    plt.fill_between(ms_use,np.log10(blue_Mh_ms2)+blue_Mh_err2[:,0],np.log10(blue_Mh_ms2)+blue_Mh_err2[:,1],
                     facecolor='b',alpha=0.2)
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.legend(loc=2)
    #plt.savefig('Theory_QE_mh_comparation',dpi=600)
    plt.show()
    return 
def comparation_M16(g):
    h = 0.72
    delta_value = np.log10(h)
    t = g
###输入数据并对比,g=1,表示M16的数据对比,g=0,2表示模拟数据的对比
    f_red_ms,tot_f_red_ms,p_red_msmh1,p_red_ms1,red_p_Mh_ms1,red_pMhms1,red_Mh_ms1,red_Mh_err1,\
    f_blue_ms,p_blue_msmh1,p_blue_ms1,blue_p_Mh_ms1,blue_pMhms1,blue_Mh_ms1,blue_Mh_err1,\
    f_red_mh,tot_f_red_mh,p_red_msmh2,p_red_ms2,red_p_Mh_ms2,red_pMhms2,red_Mh_ms2,red_Mh_err2,\
    f_blue_mh,p_blue_msmh2,p_blue_ms2,blue_p_Mh_ms2,blue_pMhms2,blue_Mh_ms2,blue_Mh_err2,\
    _halo,ms_use = func_fred(f=True)
    ###数据导入
    if t==0:
        ####导入M16观测数据
        mh_r = np.array([12.17,12.14,12.50,12.89,13.25,13.63,14.05])
        mh_r_err = np.array([[0.19,0.12,0.04,0.04,0.03,0.03,0.05],
                              [-0.24,-0.14,-0.05,-0.04,-0.03,-0.03,-0.05]])
        ms_r = np.array([10.28,10.58,10.86,11.10,11.29,11.48,11.68])
        mh_b = np.array([11.80,11.73,12.15,12.61,12.69,12.79,12.79])
        mh_b_err = np.array([[0.16,0.13,0.08,0.10,0.19,0.43,0.58],
                              [-0.20,-0.17,-0.10,-0.11,-0.25,-1.01,-2.23]])
        ms_b = np.array([10.24,10.56,10.85,11.10,11.28,11.47,11.68])
    elif t==1:
        ####导入模拟的观测数据(数据点比较多情况)
        mh_r= np.array([11.84144843,11.84144843,11.89535823,11.95580005,12.02537811,\
              12.10329057,12.19181432,12.29147487,12.40331193,12.5262711,\
              12.66457476,12.80943267,12.96171603,13.12992385,13.30603297,\
              13.49122876,13.65855888,13.85375991,14.01142053,14.05900916,\
              14.18654906])
        ms_r = np.array([9.55516636,9.66545281,9.77573927,9.88602573,9.99631218,\
              10.10659864,10.2168851,10.32717155,10.43745801,10.54774447,\
              10.65803092,10.76831738,10.87860384,10.98889029,11.09917675,\
              11.20946321,11.31974967,11.43003612,11.54032258,11.65060904,11.76089549])
        mh_r_err= np.array([0.01,0.51726065,0.513776,0.51061946,0.50786479,0.50636558,\
              0.5054495,0.505326,0.5058986,0.50688928,0.50698652,0.5107636,\
              0.51497867,0.51649112,0.51241142,0.49964074,0.48506296,0.47290132,\
              0.45835034,0.48461951,0.04812004])
        mh_b= np.array([11.56549616,11.56549616,11.61073136,11.66002609,11.71386246,\
              11.77294388,11.83778009,11.90955185,11.98354181,12.06735616,\
              12.15251732,12.24396081,12.33427442,12.4449957,12.54540254,\
              12.66468538,12.82667239,13.01481881,13.08768458,13.21836804,\
              13.27014179])
        ms_b= np.array([9.54956774,9.64870165,9.74783556,9.84696947,9.94610338,\
              10.0452373,10.14437121,10.24350512,10.34263903,10.44177294,\
              10.54090685,10.64004076,10.73917467,10.83830858,10.93744249,\
              11.0365764,11.13571031,11.23484422,11.33397813,11.43311204,11.53224595])
        mh_b_err= np.array([0.01,0.31073456,0.31014416,0.31049053,0.31156874,0.31375724,\
              0.3163361,0.3207385,0.32540602,0.32897889,0.33508065,0.34308683,\
              0.34839135,0.3588951,0.36576389,0.39988588,0.40469186,0.34897021,\
              0.34346791,0.22070814,0.28744449]) 
    else:
        ####导入模拟的观测数据(数据点比较少情况)
        mh_r= np.array([11.93213772,11.93213772,12.12434067,12.38838582,12.73441978,\
               13.15024239,13.62083619,14.05767997])
        ms_r = np.array([9.6447741,9.93427605,10.223778,10.51327995,10.8027819,\
                 11.09228385,11.3817858,11.67128775])
        mh_r_err = np.array([0.01,0.51185669,0.50611527,0.50602658,0.5087693,0.51648376,\
                 0.48655984,0.42089061])
        mh_b = np.array([11.64123409,11.64123409,11.7885786,11.97499485,12.19929409,\
               12.45631101,12.75363835,13.02485297])
        ms_b = np.array([9.63011405,9.89034056,10.15056708,10.41079359,10.67102011,\
             10.93124662,11.19147314,11.45169965])
        mh_b_err = np.array([0.01,0.31022394,0.31428295,0.32420174,0.33966039,0.36085605,\
                 0.4146244,0.333523])
    ##下面做图对比
    if t==0:
        line2,caps2,bars2=plt.errorbar(ms_r,mh_r,yerr=abs(mh_r_err)[::-1],fmt="ro--",linewidth=1,
                                    elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='red(M16)')
        line4,caps4,bars3=plt.errorbar(ms_b,mh_b,yerr=abs(mh_b_err)[::-1],fmt="bo--",linewidth=1,
                                    elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='blue(M16)')
        #plt.errorbar(ms_use,np.log10(red_Mh_ms1),yerr=abs(red_Mh_err1.T),fmt="r^-",linewidth=0.5,
        #         elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$red < M_h-M_\ast > ms$')
        #plt.errorbar(ms_use,np.log10(blue_Mh_ms1),yerr=abs(blue_Mh_err1.T),fmt="bs-",linewidth=0.5,
        #         elinewidth=0.5,ecolor='b',capsize=0.5,capthick=0.5,label=r'$blue < M_h-M_\ast > ms$')
        #plt.errorbar(ms_use,np.log10(red_Mh_ms2),yerr=abs(red_Mh_err2.T),fmt="r^-.",linewidth=0.5,
        #         elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$red < M_h-M_\ast > mh$')  
        #plt.errorbar(ms_use,np.log10(blue_Mh_ms2),yerr=abs(blue_Mh_err2.T),fmt="bs-.",linewidth=0.5,
        #         elinewidth=0.5,ecolor='b',capsize=0.5,capthick=0.5,label=r'$blue < M_h-M_\ast > mh$')
        #plt.plot(ms_use-2*delta_value,np.log10(red_Mh_ms1),'r-',label=r'$QE-M_\ast-red$')
        #plt.plot(ms_use-2*delta_value,np.log10(blue_Mh_ms1),'b-',label=r'$QE-M_\ast-blue$')
        plt.plot(ms_use-2*delta_value,np.log10(red_Mh_ms2),'r-.',label=r'$QE-M_h-red$')
        plt.plot(ms_use-2*delta_value,np.log10(blue_Mh_ms2),'b-.',label=r'$QE-M_h-blue$')
        plt.xlabel(r'$lgM_\ast [M_\odot ]$')
        plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
        plt.legend(loc=2)
        #plt.savefig('Correct_Theory_compare_data',dpi=600)
        plt.show()
    else:
        line2,caps2,bars2=plt.errorbar(ms_r-2*delta_value,mh_r,yerr=[abs(mh_r_err),abs(mh_r_err)],fmt="ro--",linewidth=1,
                                    elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='red(M16)')
        line4,caps4,bars3=plt.errorbar(ms_b-2*delta_value,mh_b,yerr=[abs(mh_b_err),abs(mh_b_err)],fmt="bo--",linewidth=1,
                                    elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='blue(M16)')
        #plt.errorbar(ms_use,np.log10(red_Mh_ms1),yerr=abs(red_Mh_err1.T),fmt="r^-",linewidth=0.5,
        #         elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$red < M_h-M_\ast > ms$')
        #plt.errorbar(ms_use,np.log10(blue_Mh_ms1),yerr=abs(blue_Mh_err1.T),fmt="bs-",linewidth=0.5,
        #         elinewidth=0.5,ecolor='b',capsize=0.5,capthick=0.5,label=r'$blue < M_h-M_\ast > ms$')
        #plt.errorbar(ms_use,np.log10(red_Mh_ms2),yerr=abs(red_Mh_err2.T),fmt="r^-.",linewidth=0.5,
        #         elinewidth=0.5,ecolor='r',capsize=0.5,capthick=0.5,label=r'$red < M_h-M_\ast > mh$')  
        #plt.errorbar(ms_use,np.log10(blue_Mh_ms2),yerr=abs(blue_Mh_err2.T),fmt="bs-.",linewidth=0.5,
        #         elinewidth=0.5,ecolor='b',capsize=0.5,capthick=0.5,label=r'$blue < M_h-M_\ast > mh$')
        #plt.plot(ms_use-2*delta_value,np.log10(red_Mh_ms1),'r-',label=r'$QE-M_\ast-red$')
        #plt.plot(ms_use-2*delta_value,np.log10(blue_Mh_ms1),'b-',label=r'$QE-M_\ast-blue$')
        plt.plot(ms_use-2*delta_value,np.log10(red_Mh_ms2),'r-.',label=r'$QE-M_h-red$')
        plt.plot(ms_use-2*delta_value,np.log10(blue_Mh_ms2),'b-.',label=r'$QE-M_h-blue$')
        plt.xlabel(r'$lgM_\ast [M_\odot ]$')
        plt.ylabel(r'$lgM_h [M_\odot h^{-1}]$')
        plt.legend(loc=2)    
        #plt.savefig('Correct_parameter_data',dpi=600)
        #plt.savefig('Mock_parameter_data',dpi=600)
        plt.show()
    return
#########################
def control_file(R):
    #doload_mock_data(tt=True)###模拟数据导入
    #bins_dic(tt=True)###计算边界条件概率分布（对数空间概率密度分布），并且完成概率归一化
    #mass_function_use(uu=True)####把实际调用的从模拟数据的质量函数表示出来
    #the_probability(pp=True)###理论计算的第一步，计算P(m*|Mh)
    #figure_1(a=True)###做图显示上一步计算结果
    #Theory_fun3(ff3=True)###计算联合概率分布，以及P(Mh|m*)
    #figure_2(b=True)#####作图显示上一步结果
    #R_Mh_Ms(c=True)####求解mh-m*的关系
    #fig_R_Mh_Ms(d=True)####作图显示上一步结果
    #func_fred(f=True)###求解红蓝星系的质量函数关系
    #fig_func_fred(g=True)###作图显示上一步结果
    comparation_M16(g=2)###输入数据并对比,g=1,表示M16的数据对比,g=0,2表示模拟数据的对比
    return
control_file(R=True)