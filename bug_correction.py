######该脚本保存有待修改的部分，包括对原模拟数据的分区件处理和根据质量函数的分布区间选择
####链接路径引入
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
##件的路径具体到文件夹
###库函数调用
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
from Theory_calculation import the_probability
from Theory_calculation import bins_dic
### section1：导入数据产生需要使用的四个数组：mock_mstar——theory_mh,利用此关系反插的mock_mh——theory_mstar
def doload_mock_data(tt):
    mockfile = 'D:/Python1/pydocument/seniorproject_quenching2/practice/iHODcatalog_bolshoi.h5'
    Mh_arr, dndlnMh_arr, use_data = read_mock_hmf(mockfile, mmin=1.e9, mmax=1.e16, 
                                                  nmbin=101, h=0.701, rcube=250.0)  
    return use_data
###################对原模拟数据的分析部分
def mass_function_mock(dd):
    use_data = doload_mock_data(tt=True)
    N = np.array(use_data['b'].shape)
    print('total_number=',N)
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
    print('use_number=',n)
    frac_ = n[0]/N[0]
    print('fraction_sample=',frac_)
    #con_c = use_data['b']
    #con = np.array(con_c[ix])
###下面为质量函数
####mh-function
    value_mh = plt.hist(main_halo,100,normed=True)
    plt.yscale('log')
    #plt.savefig('mock_Mh',dpi=600)
    plt.show()
    p_Mh = np.array(value_mh[0])
    f_N_mh = np.array(value_mh[0])
    x_N_mh = np.array(value_mh[1])
    print(value_mh[0].size)
    #print(value_mh[1])
    media_mh = np.zeros(value_mh[0].size,dtype=np.float)
    dn_dMh = np.zeros(value_mh[0].size,dtype=np.float)
    for k in range(1,value_mh[0].size):
            media_mh[k] = (x_N_mh[k]+x_N_mh[k-1])/2
            dn_dMh[k] = f_N_mh[k]/(x_N_mh[1]-x_N_mh[0])
    media_mh[0] = media_mh[1]
    dn_dMh[0] = f_N_mh[0]/(x_N_mh[1]-x_N_mh[0])
    plt.plot(media_mh,dn_dMh,label='mock dn_dlgMh')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.ylabel(r'$dn/dlgM_h$')
    plt.grid()
    plt.title('Mh_fun')
    #plt.savefig('fun_Mh',dpi=600)
    plt.show()
####mstar-function    
    value_ms = plt.hist(mstar,100,normed=True)
    plt.yscale('log')
    #plt.savefig('mock_mstar',dpi=600)
    plt.show()
    p_ms = np.array(value_ms[0])
    f_N_ms = np.array(value_ms[0])
    x_N_ms = np.array(value_ms[1])
    print(value_ms[0].size)
    #print(value_ms[1])
    media_ms = np.zeros(value_ms[0].size,dtype=np.float)
    dn_dms = np.zeros(value_ms[0].size,dtype=np.float)
    for k in range(1,value_ms[0].size):
            media_ms[k] = (x_N_ms[k]+x_N_ms[k-1])/2
            dn_dms[k] = f_N_ms[k]/(x_N_ms[1]-x_N_ms[0])
    media_ms[0] = media_ms[1]
    dn_dms[0] = f_N_ms[0]/(x_N_ms[1]-x_N_ms[0])
    plt.plot(media_ms,dn_dms,label=r'$mock-dn_dlgM_\ast$')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.ylabel(r'$dn/dlgM_\ast$')
    plt.grid()
    plt.title('stellarmass_fun')
    #plt.savefig('fun_Ms',dpi=600)
    plt.show() 
####下面求联合质量分布函数
    vl_joint = plt.hist2d(mstar,main_halo,bins=[100,100],
              range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]]
               ,normed=False,cmin=0,cmap='rainbow',vmin=1,vmax=1e5,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.xlabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.ylabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.show()
    print(vl_joint[0].shape)
    ##第一行表示的是各个划分区间的数目多少
    #print(vl_joint[1])
    ##第二行表示对恒星质量的划分区间
    #print(vl_joint[2])
    ##第三行表示对暗晕质量的划分区间
    N_joint = np.array(vl_joint[0])
    ms_joint = np.array(vl_joint[1])
    mh_joint = np.array(vl_joint[2])
    dN_mh_joint = np.zeros((len(N_joint[:,0]),len(N_joint[0,:])),dtype=np.float)
    dN_ms_joint = np.zeros((len(N_joint[0,:]),len(N_joint[:,0])),dtype=np.float)
    for k in range(0,len(N_joint[:,0])):
        dN_ms_joint[k,:] = N_joint[k,:]/(ms_joint[1]-ms_joint[0])
    for k in range(0,len(N_joint[0,:])):
        dN_mh_joint[:,k] = N_joint[:,k]/(mh_joint[1]-mh_joint[0])
    bin1 = np.linspace(np.min(mstar), np.max(mstar), 100)
    bin2 = np.linspace(np.min(main_halo),np.max(main_halo),100) 
    mstar_ = mstar
    mainhalo = main_halo
    mstar_, mainhalo = np.meshgrid(bin2, bin1)
    plt.pcolormesh(mstar_,mainhalo,dN_ms_joint,cmap='rainbow',vmin=1,vmax=1e5,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    #plt.savefig('DN',dpi=600)
    plt.show()
####下面带入公式计算，产生另外两个数组
    ##带入恒星质量mstar，计算理论值Mh
    M0 = 2*10**10###单位是Msolar/h^2
    M1 = 1.3*10**12###单位是Msolar/h
    belta = 0.33
    sigma = 0.42
    gamma = 1.21
    ##下面计算为了和模拟数据保持一直,把SHMR的质量转为以10为底的对数
    Theory_Mh = np.log10(M1)+belta*(mstar-np.log10(M0))+\
        ((10**mstar/M0)**sigma/(1+(10**mstar/M0)**(-gamma))-1/2)
    plt.plot(mstar,Theory_Mh,'*',label='SHMR-curve')
    plt.xlabel(r'$mock-M_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$Theory-M_h [M_\odot h^{-1}]$')
    plt.show()
    ###画出SHMR解析关系下的质量函数
    Mh_theory = plt.hist(Theory_Mh,100,alpha=0.2)
    plt.yscale('log')
    #plt.savefig('Theory_Mh',dpi=600)
    plt.show()
    N_Mh_theory = np.array(Mh_theory[0])
    x_Mh_theory = np.array(Mh_theory[1])
    print(Mh_theory[0].size)
    media_Mh_theory = np.zeros(Mh_theory[0].size,dtype=np.float)
    dn_dMh_theory = np.zeros(Mh_theory[0].size,dtype=np.float)
    for k in range(1,Mh_theory[0].size):
            media_Mh_theory[k] = (x_Mh_theory[k]+x_Mh_theory[k-1])/2
            dn_dMh_theory[k] = N_Mh_theory[k]/(x_Mh_theory[1]-x_Mh_theory[0])
    media_Mh_theory[0] = media_Mh_theory[1]
    dn_dMh_theory[0] = N_Mh_theory[0]/(x_Mh_theory[1]-x_Mh_theory[0])
    plt.plot(media_Mh_theory,dn_dMh_theory,label=r'$theory-dn_dlgM_h$')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$dn/dlgM_h$')
    plt.grid()
    plt.title('Theory_Mh_fun')
    #plt.savefig('fun_Mh_theory',dpi=600)
    plt.show() 
###下面做插值处理,带入暗晕质量，插值出理论mstar
    Theory_mstar = np.interp(main_halo,Theory_Mh,mstar)
    plt.plot(main_halo,Theory_mstar,'*')
    plt.xlabel(r'$mock-M_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$Theory-M_\ast [M_\odot h^{-2}]$')
    plt.show()
    ###画出给定模型理论下的恒星质量函数
    mstar_theory = plt.hist(Theory_mstar,100,alpha=0.2)
    plt.yscale('log')
    #plt.savefig('Theory_mstar',dpi=600)
    plt.show()
    N_mstar_theory = np.array(mstar_theory[0])
    x_mstar_theory = np.array(mstar_theory[1])
    print(mstar_theory[0].size)
    media_mstar_theory = np.zeros(mstar_theory[0].size,dtype=np.float)
    dn_dms_theory = np.zeros(mstar_theory[0].size,dtype=np.float)
    for k in range(1,mstar_theory[0].size):
            media_mstar_theory[k] = (x_mstar_theory[k]+x_mstar_theory[k-1])/2
            dn_dms_theory[k] = N_mstar_theory[k]/(x_mstar_theory[1]-x_mstar_theory[0])
    media_mstar_theory[0] = media_mstar_theory[1]
    dn_dms_theory[0] = N_mstar_theory[0]/(x_mstar_theory[1]-x_mstar_theory[0])
    plt.plot(media_mstar_theory,dn_dms_theory,label=r'$theory-dn_dlgM_\ast$')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$dn/dlgM_\ast$')
    plt.grid()
    plt.title('Theory_mstar_fun')
    #plt.savefig('fun_mstar_theory',dpi=600)
    plt.show() 
    out1 = {'a':mstar,'b':Theory_Mh,'c':dn_dms,'d':dn_dMh_theory}
    out2 = {'a':main_halo,'b':Theory_mstar,'c':dn_dMh,'d':dn_dms_theory}
####计算模拟数据的<N(M*|Mh)>,该部分有待修正，需要对质量区间重新划分
    return out1,out2,p_Mh,p_ms
###############################
##############################对质量函数的区间选取部分
def Theory_fun1(ff1):
####只对mhalo做限制的情况
    media_mh,dn_dlgMh,media_ms,dn_dlgms,p_lgMh,p_lgms = bins_dic(tt=True)
    mhalo,ms,N_ms_Mh1,N_ms_Mh2,N_ms_Mh3,Theory_Mh,Theory_mstar,sigma_ms\
    = the_probability(pp=True)
####下面计算p(mstar,mh),即为联合分布概率P_joint,注意需要划分质量区间——对引入的质量函数的适应
    ng = 1
    x1 = media_mh
    y1 = dn_dlgMh
    f1 = sinter.interpolate.interp1d(x1,y1)
    ###下面这句对质量函数的自适应
    _halo = np.linspace(np.min(x1),np.max(np.log10(mhalo)),len(mhalo))
    dn_dMh_theory = f1(_halo)
    dn_dMh = dn_dMh_theory/(10**_halo*np.log(10))
    plt.plot(_halo,dn_dMh)
    plt.xlabel("$lgM_h [M_\odot h^{-1}]$")
    plt.ylabel("$dn/dMh$")
    plt.yscale('log')
    plt.grid()
    plt.show()
    ###下面对N_ms_Mh1也做相应的质量范围截断
    ###对halo质量，N_ms_Mh都做出重新的选择对ms也要做出相范围内的选择,下面这句表示对ms的调整
    halo_ = np.linspace(np.min(x1),np.max(np.log10(mhalo)),len(ms))
    ms_use = np.interp(halo_,np.log10(mhalo[0:950]),np.log10(ms))    
    ms_use = 10**ms_use
    N_ms_Mh = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(0,len(ms)):
        ix = np.log10(mhalo)>=np.min(x1)
        x2 = np.array(mhalo[ix])
        y2 = N_ms_Mh1[ix,k]
        N_ms_Mh[:,k] = np.interp(10**_halo,x2,y2)
    plt.pcolormesh(np.log10(10**_halo),np.log10(ms_use),N_ms_Mh.T,cmap='rainbow',vmin=0.01,vmax=1,alpha=1,
               norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('L11_2011')
    #plt.savefig('Theory_DN_function_contrain_mh',dpi=600)
    plt.show()
    p_joint = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(len(ms)):
        p_joint[:,k] = ((np.log10(np.e))/(ms_use[k]*ng))*N_ms_Mh[:,k]*dn_dMh  
    ##下面对p_joint做归一化检验
    sum1 = np.zeros(len(mhalo),dtype=np.float)
    for k in range(len(mhalo)):
        sum1[k] = np.sum(p_joint[k,:]*ms_use*(np.log10(ms_use[2])-np.log10(ms_use[1]))) 
    sum2 = np.sum(sum1*10**_halo*(_halo[2]-_halo[1]))
    A = sum2
    print(A)
    P_Joint1 = p_joint/A   
####下面求条件概率p_Mh_ms代替p(Mh|ms),p_ms_Mh代替p(ms|Mh)
####往下求条件概率分布遇到范围限制的困难：原因是求解的联合分布质量范围和边界条件分布不一致
    return P_Joint1

def Theory_fun2(ff2):
####只对ms做限制的情况
    media_mh,dn_dlgMh,media_ms,dn_dlgms,p_lgMh,p_lgms = bins_dic(tt=True)
    mhalo,ms,N_ms_Mh1,N_ms_Mh2,N_ms_Mh3,Theory_Mh,Theory_mstar,sigma_ms\
    = the_probability(pp=True)
####下面计算p(mstar,mh),即为联合分布概率P_joint,注意需要划分质量区间——对引入的质量函数的适应
    ng = 1
    x1 = media_ms
    y1 = dn_dlgms
    f1 = sinter.interpolate.interp1d(x1,y1)
    x2 = media_mh
    y2 = dn_dlgMh
    f2 = sinter.interpolate.interp1d(x2,y2)
    ###下面这句对质量函数的自适应
    _ms = np.linspace(np.min(x1),np.max(x1),len(ms))
    dn_dms_theory = f1(_ms)
    dn_dms = dn_dms_theory/(10**_ms*np.log(10))
    plt.plot(_ms,dn_dms)
    plt.xlabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.ylabel(r'$dn/dms$')
    plt.yscale('log')
    plt.grid()
    plt.show()
    ###下面对N_ms_Mh1也做相应的质量范围截断
    ###对halo质量，N_ms_Mh都做出重新的选择对mh也要做出相范围内的选择,下面这句表示对ms的调整
    ms_ = np.linspace(np.min(x1),np.max(x1),len(mhalo))
    mh_use = np.interp(ms_,np.log10(ms),np.log10(mhalo[0:950]))
    dn_dMh = f2(mh_use)/(10**mh_use*np.log(10))
    '''
    plt.plot(mh_use,dn_dMh)
    plt.xlabel(r'$lgM_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$dn/dMh$')
    plt.yscale('log')
    plt.grid()
    plt.show()
    '''
    mhuse = 10**mh_use
    N_ms_Mh = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(0,len(mhalo)):
        ix = np.log10(ms)>=np.min(x1)
        x2 = np.array(ms[ix])
        y2 = N_ms_Mh1[k,ix]
        N_ms_Mh[k,:] = np.interp(10**_ms,x2,y2)
    plt.pcolormesh(np.log10(mhuse),_ms,N_ms_Mh.T,cmap='rainbow',vmin=0.01,vmax=1,alpha=1,
               norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast [M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('L11_2011')
    #plt.savefig('Theory_DN_function_contrain_ms',dpi=600)
    plt.show()
    p_joint = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(len(ms)):
        p_joint[:,k] = ((np.log10(np.e))/(10**_ms[k]*ng))*N_ms_Mh[:,k]*dn_dMh
    ##下面对p_joint做归一化检验
    sum1 = np.zeros(len(mhalo),dtype=np.float)
    for k in range(len(mhalo)):
        sum1[k] = np.sum(p_joint[k,:]*10**_ms*(_ms[2]-_ms[1])) 
    sum2 = np.sum(sum1*mhuse*(np.log10(mhuse[2])-np.log10(mhuse[1])))
    B = sum2
    print(B)
    P_Joint2 = p_joint/B
####下面求条件概率p_Mh_ms代替p(Mh|ms),p_ms_Mh代替p(ms|Mh)
####往下求条件概率分布遇到范围限制的困难：原因是求解的联合分布质量范围和边界条件分布不一致
    return P_Joint2
def Theory_fun_assum(fa):
    ####对mhalo,ms同时做限制的情况
    media_mh,dn_dlgMh,media_ms,dn_dlgms,p_lgMh,p_lgms = bins_dic(tt=True)
    mhalo,ms,N_ms_Mh1,N_ms_Mh2,N_ms_Mh3,Theory_Mh,Theory_mstar,sigma_ms =\
    the_probability(pp=True)
####下面计算p(mstar,mh),即为联合分布概率P_joint,注意需要划分质量区间——对引入的质量函数的适应
    #ng=1###宇宙平均星系数密度
    x1 = np.linspace(np.min(media_mh),np.max(media_mh),len(mhalo))
    x2 = np.linspace(np.min(media_ms),np.max(media_ms),len(ms))
    f1 = sinter.interpolate.interp1d(media_mh,p_lgMh)
    f2 = sinter.interpolate.interp1d(media_ms,p_lgms)
    plgMh = f1(x1)
    plgms = f2(x2)
    _halo = x1
    ms_use = x2    
    pmh = plgMh/(np.log(10)*10**_halo)
    pms = plgms/(np.log(10)*10**ms_use)
    A = np.sum((_halo[2]-_halo[1])*pmh)
    B = np.sum((ms_use[2]-ms_use[1])*pms)
    pmh=pmh/A
    pms=pms/B
    f3 = sinter.interpolate.interp2d(ms,mhalo,N_ms_Mh3)
    N_ms_Mh = f3(10**x2,10**x1)
    plt.pcolormesh(np.log10(10**_halo),np.log10(10**ms_use),N_ms_Mh.T,cmap='rainbow',
                   vmin=0.01,vmax=1,alpha=1,norm = mpl.colors.LogNorm()) 
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.title('assumption')
    #plt.savefig('Theory_DN_function_comparation',dpi=600)
    plt.show()
    ##求解联合分布概率
    f3 = sinter.interpolate.interp1d(media_mh,dn_dlgMh)
    #dn_dMh = f3(x1)/(np.log(10)*10**x1)#在HOD模型下需要带入质量函数计算
    p_joint = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(len(ms)):
        #p_joint[:,k] = ((np.log10(np.e))/(10**ms_use[k]*ng))*N_ms_Mh[:,k]*dn_dMh 
        ####在HOD模型下，联合分布概率从上式求解
        p_joint[:,k] = N_ms_Mh[:,k]*pmh
    ##下面对p_joint做归一化检验
    sum1 = np.zeros(len(mhalo),dtype=np.float)
    for k in range(len(mhalo)):
        sum1[k] = np.sum(p_joint[k,:]*10**ms_use*(ms_use[2]-ms_use[1])) 
    sum2 = np.sum(sum1*10**_halo*(_halo[2]-_halo[1]))
    SA = sum2
    print(SA)
    P_Joint_as = p_joint/SA
####下面求条件概率p_Mh_ms代替p(Mh|ms),p_ms_Mh代替p(ms|Mh)
    p_Mh_ms = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    p_ms_Mh = np.zeros((len(mhalo),len(ms)),dtype=np.float)
    for k in range(len(mhalo)):
        p_ms_Mh[k,:] = P_Joint_as[k,:]/pmh[k]
    for k in range(len(ms)):
        p_Mh_ms[:,k] = P_Joint_as[:,k]/pms[k] 
    plt.plot(_halo,np.log(10)*10**_halo*p_Mh_ms[:,1])
    plt.yscale('log')
    plt.show()
    plt.plot(ms_use,np.log(10)*10**ms_use*p_ms_Mh[500,:])
    plt.yscale('log')
    plt.show()
    ###注意因为pms和pmh包含0,上述两个条件概率存在Inf和nan
    p_Mh_ms[np.isinf(p_Mh_ms)]=0
    p_ms_Mh[np.isinf(p_ms_Mh)]=0
    p_Mh_ms[np.isnan(p_Mh_ms)]=0
    p_ms_Mh[np.isnan(p_ms_Mh)]=0
    return
def run_control(t):
    doload_mock_data(tt=True)
    mass_function_mock(dd=True)
    Theory_fun1(ff1=True)
    Theory_fun2(ff2=True)
    Theory_fun_assum(fa=True)###对比部分
    return
run_control(t=True)