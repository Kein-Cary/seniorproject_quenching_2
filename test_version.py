##该脚本用于把red和blue两类型系分开，做出全部M*质量区间的
####red有7个bins,蓝星系有4个bins
##Section 1:把观测数据脚本从新整理，尝试分离数据
import os.path
import numpy as np
import matplotlib.pyplot as plt
"""Read the Mandelbaum+2016 Weak Lensing data."""
m16path = 'D:/Python1/pydocument/seniorproject_quenching2/practice/data/M16/'
def read_m16_ds_2(use_red, mass_bin):
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
#read_m16_ds_1(use_red=True, mass_bin='10.0_10.4')
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
    # simply errors
    #emhlow = lgmh - np.log10(mhlow)
    emhupp = np.log10(mhupp) - lgmh
    # errlgmh = (emhlow + emhupp) * 0.5
    # (arbitrarily) take the upper errorbar
    errlgmh = emhupp
    # out = [lgms, lgmh, emhlow, emhupp]
    out = [lgms, lgmh, errlgmh]
    return(out)
#read_m16_mass_1(use_red=True)
def test_read_m16_ds_2(use_red,mass_bin):
    """Test the M16 Reader."""
    ##加入数组，记录rp,ds,ds_error的变化和取值。
    if use_red==True:
        rp, ds, ds_err = read_m16_ds_2(use_red=use_red, mass_bin=mass_bin)
        rsa = np.zeros(len(rp),dtype=np.float)
        dssa = np.zeros(len(rp),dtype=np.float)
        ds_errsa = np.zeros((2,len(rp)),dtype=np.float)
        rsa= rp
        dssa = ds
        ds_errsa = ds_err
        #plt.errorbar(rp, ds, yerr=ds_err, marker="o", ms=3, color="red")
    else:
        rp, ds, ds_err = read_m16_ds_2(use_red=use_red, mass_bin=mass_bin)
        rsa = rp
        dssa = ds
        ds_errsa = ds_err
        #plt.errorbar(rp, ds, yerr=ds_err, marker="s", ms=3, color="blue")
    '''
    plt.xlabel(r"$R\;[Mpc/h]$")
    plt.ylabel(r"$\Delta\Sigma\;[h M_\odot/pc^2]$")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend(['red','blue'])
    #print('rp=',rp)
    #print('ds=',dssa)
    #print('error=',ds_errsa)
    #plt.show()
    '''
    return rsa,dssa,ds_errsa
    #-sa表示设置数组
    ##最后图示的是该段代码所做图像
#test_read_m16_ds_1(mass_bin=True)
def test_read_m16_mass_2():
    lgms, lgmh, err = read_m16_mass_2(use_red=True)
    plt.errorbar(lgms, lgmh, yerr=err, marker="o", ms=3, color="red")
    lgms, lgmh, err = read_m16_mass_2(use_red=False)
    plt.errorbar(lgms, lgmh, yerr=err, marker="s", ms=3, color="blue")
    plt.xlabel(r"$M_*\;[M_\odot/h^2]$")
    plt.ylabel(r"$M_h\;[M_\odot/h]$")
    plt.grid()
    #plt.show()
#test_read_m16_mass_1(True):
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
    rpb = rp
    ds_r = np.zeros((len(mr_be),len(rpr)),dtype=np.float)
    ds_b = np.zeros((len(mb_be),len(rpb)),dtype=np.float)
    ds_err_r = np.zeros((len(mr_be),len(rpr)),dtype=np.float)
    ds_err_b = np.zeros((len(mb_be),len(rpb)),dtype=np.float)      
    for k in range(0,len(mr_be)):
        mass_ = mr_be[k]
        rsa,dssa,ds_errsa = test_read_m16_ds_2(True,mass_bin=mass_)
        ds_r[k,:] = dssa
        ds_err_r[k,:] = ds_errsa
    for k in range(0,len(mb_be)):
        mass_ = mb_be[k]
        rsa,dssa,ds_errsa = test_read_m16_ds_2(False,mass_bin=mass_)
        ds_b[k,:] = dssa
        ds_err_b[k,:] = ds_errsa
    return ds_r,ds_err_r,ds_b,ds_err_b
#mass_be_(tt=True)
##################    
##################
##Section 2：该部分尝试计算最佳拟合质量，并给出误差棒图像
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
#import os.path
from read_m16_1 import test_read_m16_ds_1
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from scipy.stats import t as st
#import seaborn as sns
#from scipy.stats import chisquare as chi
#画误差棒函数须引入scipy库
#导入的质量区间为11~15dex-Msolar
def input_mass_bin(v):
    _mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/'
    fname = os.path.join(_mh_path,'z_eff.txt')
    z_ = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)  
    z = np.array(z_)
    N = 401
    m = np.linspace(10.5,14.5,N)
    #针对不同质量区间设置拟合质量范围
    #print(z)
    return m,N,z
#input_mass_bin(v=True)
def calcu_sigmaz(Rpc,m_,x,z):
    global omega_m
    omega_m = 0.315
    omegam = omega_m
#a flat CDM model,omegak=0;omegalamd=0
    global h_
    h_ = 0.673
    h = h_
    global G_
    G_ = 6.67*10**(-11)
    G = G_
    #global ms_
    #ms_ = 1.989*10**(30)
    #ms = ms_  
    global c_
    c_ = 6
    ##调试结果c=3.25~3.75之间比较合适
    c = c_
    #c = (9/(1+z))*(m_/1.686)**(-0.13)
    #下面开始计算
    #对导入的数据做单位转化转为:太阳质量/h
    m1 = h*m_*10**x
    L = len(Rpc)
    Rps = np.zeros(L,dtype=np.float)
    g_x = np.zeros(L,dtype=np.float)
    deltaSigma = np.zeros(L,dtype=np.float)
    Sigma = np.zeros(L,dtype=np.float)
    # rs =  np.zeros(LL,dtype=np.float)
    # r_200 =  np.zeros(LL,dtype=np.float)
    # rho_0 = np.zeros(LL,dtype=np.float)
    #单位修正因子
    #Qc = 0.1412#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    #Q200 = 0.7
    ##重新对单位修正因子订正如下：
    Qc = 3.084*10**(-2)/(1.989*h**2)#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    #Qc = 3.084*10**(-2)/(1.989)#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    #rhom = 52700242579.0
    #m1 = m_*10**x
    Q200 = 1#对r200的单位修正（此时对rs也做了修正）
    H = h*100
    ##修正可以选择两处，对于计算过程涉及的物理量修正，或者对于常数修正
    rhoc = (Qc*(3*H**2)/(8*np.pi*G))/(1/(1+z))**3
    rhom = 200*rhoc*omegam
    #rhom=52700242579.0*200/h**2
    #考虑所有的红移为0.1
    r_200 = Q200*(3*m1/(4*np.pi*rhom))**(1/3)
    rs = r_200/c
    rho_0 = m1/((np.log(1+c)-c/(1+c))*4*np.pi*rs**3)
    for t in range(0,L):
        Rps[t] = Rpc[t]
        #引入中间函数
        f0 = Rps[t]/rs#这是考虑把R参数化的情况
        if Rps[t]<rs:
           f1 = np.arctanh(np.sqrt((1-f0)/(1+f0)))
           f2 = np.log(f0/2)
           f4 = f0**2*np.sqrt(1-f0**2)
           f5 = (f0**2-1)*np.sqrt(1-f0**2)
           g_x[t] = 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
           delta_segma = rs*rho_0*g_x[t]
          # deltasigma[n,t] = delta_segma  #(原有计算,面积单位为Mpc^2)
          # 为了和观测对比，把单位转为Msun*h/pc^2,即把面积单位转为pc^2
           deltaSigma[t] = delta_segma*10**(-12)
           Sigma[t] = (2*rs*rho_0/(f0**2-1))*(1-2*f1/(np.sqrt(1-f0**2)))
        elif Rps[t]==rs:
             g_x[t] = 10/3+4*np.log(1/2)
             delta_segma = rs*rho_0*g_x[t]
            # deltasigma[n,t] = delta_segma  #(原有计算,面积单位为Mpc^2)
            # 为了和观测对比，把单位转为Msun*h/pc^2,即把面积单位转为pc^2
             deltaSigma[t] = delta_segma*10**(-12)
             Sigma[t] = 2*rs*rho_0/3
        else:
             f1 = np.arctan(np.sqrt((f0-1)/(f0+1)))
             f2 = np.log(f0/2)
             f4 = f0**2*np.sqrt(f0**2-1)
             f5 = (f0**2-1)**(3/2)
             g_x[t]= 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
             delta_segma = rs*rho_0*g_x[t]
            # deltasigma[n,t] = delta_segma  #(原有计算,面积单位为Mpc^2)
            # 为了和观测对比，把单位转为Msun*h/pc^2,即把面积单位转为pc^2
             deltaSigma[t] = delta_segma*10**(-12)
             Sigma[t] = 2*rs*rho_0*(1-2*f1/np.sqrt(f0**2-1))/(f0**2-1)
    return Rpc,Sigma,deltaSigma,rs
#calcu_sigmaz(Rpc=True,m_=True,x=True,z=True)
##定义一个五个质量最佳预测和观测的对比图象
def fig_0(Rpc,ds_sim,lmw,r,mass_bin):
    test_read_m16_ds_1(mass_bin=mass_bin)
    for k in range(0,lmw):
        pa = r[k,1]
        plt.plot(Rpc[:],ds_sim[k,pa,:],'-*')
        nn = ds_sim[k,pa,:].size#求样本大小
        x_mean = np.mean(ds_sim[k,pa,:])#求算术平均值
        x_std = np.std(ds_sim[k,pa,:])#求标准偏差
        x_se = x_std/np.sqrt(nn)#求标准误差
        dof = nn-1#自由度计算
        alpha = 1.0-0.95
        conf_region = st.ppf(1-alpha/2.,dof)*x_std*\
        np.sqrt(1.+1./nn)#设置置信区间
        plt.errorbar(Rpc[:],ds_sim[k,pa,:],yerr=x_std,fmt='-',linewidth=0.5)
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma-hM_\odot/{pc^2}$')   
    return x_mean,x_se,conf_region
#fig_0(Rpc=True,ds_sim=True,lmw=True,r=True)
#fig_0(f=True)
##定义一个质量最佳预测和观测的对比图象
def fig_0f1(Rpc,ds_sim,k,mass_bin):
    test_read_m16_ds_1(mass_bin=mass_bin)
    plt.plot(Rpc[:],ds_sim[k,:],'--')
    nn = ds_sim[k,:].size#求样本大小
    x_mean = np.mean(ds_sim[k,:])#求算术平均值
    x_std = np.std(ds_sim[k,:])#求标准偏差
    x_se = x_std/np.sqrt(nn)#求标准误差
    dof = nn-1#自由度计算
    alpha = 1.0-0.95
    conf_region = st.ppf(1-alpha/2.,dof)*x_std*\
    np.sqrt(1.+1./nn)#设置置信区间
    plt.errorbar(Rpc[:],ds_sim[k,:],yerr=x_std,fmt='-',linewidth=0.5)
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma-hM_\odot/{pc^2}$')
    return x_mean,x_se,conf_region
#fig_0f1(f=True)
def fig_0f2(Rpc,ds_sim,k,t,mass_bin):
    test_read_m16_ds_1(mass_bin=mass_bin)
    plt.plot(Rpc[:],ds_sim[k,t,:],'--',)
    nn = ds_sim[k,t,:].size#求样本大小
    x_mean = np.mean(ds_sim[k,t,:])#求算术平均值
    x_std = np.std(ds_sim[k,t,:])#求标准偏差
    x_se = x_std/np.sqrt(nn)#求标准误差
    dof = nn-1#自由度计算
    alpha = 1.0-0.95
    conf_region = st.ppf(1-alpha/2.,dof)*x_std*np.sqrt(1.+1./nn)#设置置信区间
    plt.errorbar(Rpc[:],ds_sim[k,t,:],yerr=x_std,fmt='-',linewidth=0.5)
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma-hM_\odot/{pc^2}$')   
    return x_mean,x_se,conf_region
#fig_0f2(f=True)
#def fit_datar(y):
def fit_datar_2(use_red,mass_bin,z_inpt,T):
    h = 0.673
    #先找出观测值对应的rp
    rsa,dssa,ds_errsa = test_read_m16_ds_2(use_red=use_red,mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    #a = np.shape(dssa)
    #print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    #z_r = 0.1
    #a_r = 1/(1+z_r)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[k] for k in range(0,len(rsa)) if rsa[k]*h<=5])
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    #print(rp)
    b = len(rp)
    m,N,z = input_mass_bin(v=True)
    mr = m
    lm_bin = len(mr) 
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simr = np.zeros((lm_bin,b),dtype=np.float)
    #rr = np.zeros(lm_bin,dtype=np.float)
    #该句用于下面相应作图部分
    for t in range(0,lm_bin):
        m_ = 1
        x = mr[t]
        z_r = z_inpt
        #Rpc = rp
        #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)    
        #ds_simr[k,t,:] = deltaSigma
        #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
        #计算模型在对应的投射距离上的预测信号取值
        Rpc = rp
        Rpc,Sigma,deltaSigma,rs = calcu_sigmaz(Rpc,m_,x,z_r)
        #对预测信号做相应的变化，把共动的转为物理的
        ds_simr[t,:] = deltaSigma
        #下面部分作图对比
        #rr[t] = rs
        #fig_0f1(rp,ds_simr,t,mass_bin)
    #plt.title('Red')
    #plt.legend(m[:],bbox_to_anchor=(1,1))
    #plt.legend(m[:])   
    #plt.savefig('Fit-red2.png',dpi=600)
    #plt.savefig('compare_1_rd.png',dpi=600)
    #plt.show()
    yy = np.shape(ds_simr)
    #print('rr=',rr)#输出查看ds_sim的维度，即模型预测下的透镜信号    
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    delta_r = np.zeros(lm_bin,dtype=np.float)
    for t in range(0,lm_bin):
        d_r = 0
        for n in range(0,yy[1]):
            d_r = d_r+((ds_simr[t,n]-dssa[n])/ds_errsa[n])**2
        delta_r[t] = d_r
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    deltar = delta_r.tolist()
    xr = deltar.index(min(deltar))
    bestfmr = mr[xr]
    best_mr = bestfmr+np.log10(h)
    #作图对比最佳情况
    '''
    plt.figure()
    k = xr
    fig_0f1(rp,ds_simr,k,mass_bin)
    plt.title('Red')
    #plt.savefig('fit_r.png',dpi=600)
    plt.show()
    '''
    #print('x^2=',min(deltar))
    #print('mr=',best_mr)
    #print('positionr=',xr)
    return rp,best_mr,delta_r,bestfmr
#fit_datar_1(y=True)
#def fit_datab_1(y):
def fit_datab_2(use_red,mass_bin,z_inpt,T):
    h = 0.673
    #先找出观测值对应的rp
    rsa,dssa,ds_errsa = test_read_m16_ds_2(use_red=use_red,mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    #a = np.shape(dssa)
    #print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    #z_b = 0.1
    #a_b = 1/(1+z_b)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[k] for k in range(0,len(rsa)) if rsa[k]*h<=5]) 
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    #print(rp)
    b = len(rp)
    m,N,z = input_mass_bin(v=True)
    mb = m
    #print('mr=',mb)
    lm_bin = len(mb) 
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simb = np.zeros((lm_bin,b),dtype=np.float)
    #rb = np.zeros(lm_bin,dtype=np.float)
    #该句对应下面作图部分
    for t in range(0,lm_bin):
        m_ = 1
        x = mb[t]
        z_b = z_inpt
        #Rpc = rp
        #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)    
        #ds_simr[k,t,:] = deltaSigma
        #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
        #计算模型在对应的投射距离上的预测信号取值
        Rpc = rp
        Rpc,Sigma,deltaSigma,rs = calcu_sigmaz(Rpc,m_,x,z_b)
        #对预测信号做相应的变化，把共动的转为物理的
        ds_simb[t,:] = deltaSigma
        ##下面部分作图对比
        #rb[t] = rs
        #fig_0f1(rp,ds_simb,t,mass_bin)
    #plt.title('Blue')
    #plt.legend(m[:],bbox_to_anchor=(1,1))   
    #plt.legend(m[:])   
    #plt.savefig('Fit-blue2.png',dpi=600)
    #plt.savefig('compare_2_bl.png',dpi=600)
    #plt.show()
    yy = np.shape(ds_simb)
    #print('rr=',rr)#输出查看ds_sim的维度，即模型预测下的透镜信号    
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    delta_b = np.zeros(lm_bin,dtype=np.float)
    for t in range(0,lm_bin):
        d_b = 0
        for n in range(0,yy[1]):
            d_b = d_b+((ds_simb[t,n]-dssa[n])/ds_errsa[n])**2
        delta_b[t] = d_b
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    deltab = delta_b.tolist()
    xb = deltab.index(min(deltab))
    bestfmb = mb[xb]
    best_mb = bestfmb+np.log10(h)
    #作图对比最佳情况
    '''
    plt.figure()
    k = xb
    fig_0f1(rp,ds_simb,k,mass_bin)
    plt.title('Blue')
    plt.savefig('fit_r.png',dpi=600)
    plt.show()
    '''
    #print('x^2=',min(deltab))
    #print('mb=',best_mb)
    #print('positionr=',xb)
    return rp,best_mb,delta_b,bestfmb
#fit_datab_1(y=True)
#下面做图比较最佳预测值与观测的对比情况
##下面的函数用于检测x^2的分布形式
def fig_2(tt):
    m,N,z = input_mass_bin(v=True)
    #调用时需要修改下面两句函数的关键字
    rp,best_mr,delta_r,bestfmr = fit_datar_2(y=True)
    rp,best_mb,delta_b,bestfmb = fit_datab_2(y=True)
    plt.subplot(121)
    plt.plot(m,np.log10(delta_r),'r')
    plt.title('R-galaxy')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.grid()
    plt.subplot(122)
    plt.plot(m,np.log10(delta_b),'b')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.title('B-galaxy')
    plt.grid()
    plt.tight_layout()
    plt.show()  
    print('mr=',best_mr)
    print('mb=',best_mb)
    return m,N,best_mr,best_mb
#fig_2(tt=True)
def fig_mass(pp):
    rp,ds,ds_err = read_m16_ds_2(use_red=True, mass_bin='10.0_10.4')
    #mr_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_11.2',
    #        '11.2_11.4','11.4_11.6','11.6_15.0','11.0_15.0']
    #mb_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_11.2',
    #        '11.2_11.4','11.4_11.6','11.6_15.0','11.0_15.0']
    mr_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_11.2',
             '11.2_11.4','11.4_11.6','11.6_15.0']
    mb_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_11.2',
             '11.2_11.4','11.4_11.6','11.6_15.0']
    #mb_be = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_15.0']
    ##此句表示四个蓝色星系的版本，对应下面的z_blue用循环赋值部分
    #作图取点
    mbinr = [10.28,10.58,10.86,11.10,11.29,11.48,11.68]
    mbinb = [10.24,10.56,10.85,11.10,11.28,11.47,11.68]
    ##同样的，下面这句表示四个蓝色星系质量区间的版本
    #mbinb = [10.24,10.56,10.85,11.12]
    m,N,z = input_mass_bin(v=True)
    z_red = np.zeros(len(mr_be),dtype=np.float)
    z_blue = np.zeros(len(mb_be),dtype=np.float)
    for k in range(0,len(mr_be)):
        z_red[k] = z[0,k]
    for k in range(0,len(mb_be)):
        z_blue[k] = z[1,k]
    '''
    ##下面这部分对红移的赋值针对四个质量区间的蓝星系情况
    for k in range(0,len(mb_be)):
        if k==len(mb_be)-1:
           z_blue[k] = z[1,-1]
        else:
            z_blue[k] = z[1,k]
    '''
    print(z_red)
    print(z_blue)
    ##计算误差棒并存储的数组
    err_bar_r = np.zeros((len(mr_be),2),dtype=np.float)
    err_bar_b = np.zeros((len(mb_be),2),dtype=np.float)
    #该句导入拟合质量
    Pr_mh_r = np.zeros(len(mr_be),dtype=np.float)
    Pr_mh_b = np.zeros(len(mb_be),dtype=np.float)
    b_m_r = np.zeros(len(mr_be),dtype=np.float)
    b_m_b = np.zeros(len(mb_be),dtype=np.float)
    ##用两个数组分别记录delta_x^2
    Dchi_r = np.zeros((len(mr_be),len(m)),dtype=np.float)
    Dchi_b = np.zeros((len(mb_be),len(m)),dtype=np.float)
    #Pr_mh的第一行存red的质量，第二行存blue的质量,x_std也一样
    ##给定下面两个数据，用于记录x^2最小值的变化情况（在这里针对单参数拟合，
    ##可以不用记录，但是对于高位参数拟合，需要记录x^2的最小值情况，以便MCMC的自检测拟合）
    chi_min_r = np.zeros(len(mr_be),dtype=np.float)
    chi_min_b = np.zeros(len(mb_be),dtype=np.float)
    for k in range(0,len(mr_be)):
        rp,best_mr,delta_r,bestfmr = fit_datar_2(True,mr_be[k],z_red[k],k)
        Pr_mh_r[k] = best_mr
        b_m_r[k] = bestfmr
        chi_min_r[k] = np.min(delta_r)
        ##下面把x^2转化为相应的概率分布，并让mh的划分区间服从这个分布
        mr = m
        rsa,dssa,ds_errsa = test_read_m16_ds_2(True,mass_bin=mr_be[k])
        pr = np.ones(len(mr),dtype=np.float)
        ds_errsar = ds_errsa[0:len(rp)]
        ##记录delta_x^2
        dchir = delta_r-np.min(delta_r)
        Dchi_r[k,:] = dchir
        for q in range(0,len(mr)):
            ss=1
            for t in range(0,len(rp)):
                ss = ss*(1/(np.sqrt(np.pi*2)*ds_errsar[t]))*np.exp((-1/2)*delta_r[q])
            pr[q]=ss
        ##第一次归一化，把概率密度函数归一到0~1之间
        pr = (pr-np.min(pr))/(np.max(pr)-np.min(pr))
        '''
        plt.figure()
        plt.plot(mr,pr,'r')
        plt.title('Probability distribution')
        plt.xlabel(r'$log(M_h/M_\odot)$')
        plt.ylabel(r'$dP(M_h|M_\ast)/dM_h$')
        plt.show()
        '''
        #print('r=',delta_r)
        ##第二次归一化，目的是让所有概率求和为1
        fr=np.zeros(len(mr),dtype=np.float)
        ss=0
        Fr=np.zeros(len(mr),dtype=np.float)
        for q in range(0,len(mr)):
            ss=ss+pr[q]*(mr[1]-mr[0])
            fr[q]=ss
        Ar = np.max(fr)
        ss=0
        for q in range(0,len(mr)):
            ss=ss+pr[q]*(mr[1]-mr[0])/Ar
            Fr[q]=ss  
        #plt.figure()
        #plt.plot(mr,Fr,'r')
        #plt.show()
        vr = np.interp(0.16,Fr,mr)
        ur = np.interp(0.84,Fr,mr)
        err_bar_r[k,:] = [vr-bestfmr,ur-bestfmr]
    for k in range(0,len(mb_be)):
        rp,best_mb,delta_b,bestfmb = fit_datab_2(False,mb_be[k],z_blue[k],k)
        Pr_mh_b[k] = best_mb
        b_m_b[k] = bestfmb
        chi_min_b[k] = np.min(delta_b)
        ##下面把x^2转化为相应的概率分布，并让mh的划分区间服从这个分布
        mb = m
        rsa,dssa,ds_errsa = test_read_m16_ds_2(False,mass_bin=mb_be[k])
        pb = np.ones(len(mb),dtype=np.float)
        ds_errsab = ds_errsa[0:len(rp)]
        ##记录delta_x^2
        dchib = delta_b-np.min(delta_b)
        Dchi_b[k,:] = dchib
        for q in range(0,len(mb)):
            qq=1
            for t in range(0,len(rp)):
                qq = qq*(1/(np.sqrt(np.pi*2)*ds_errsab[t]))*np.exp((-1/2)*delta_b[q])
            pb[q]=qq
        ##第一次归一化，把概率密度函数归一到0~1之间
        pb = (pb-np.min(pb))/(np.max(pb)-np.min(pb))
        '''
        plt.figure()
        plt.plot(mb,pb,'b')
        plt.title('Probability distribution')
        plt.xlabel(r'$log(M_h/M_\odot)$')
        plt.ylabel(r'$dP(M_h|M_\ast)/dM_h$')
        plt.show()
        '''
        #print('b=',delta_b)
        ##第二次归一化，目的是让所有概率求和为1
        fb=np.zeros(len(mb),dtype=np.float)
        qq=0
        Fb=np.zeros(len(mb),dtype=np.float)
        for q in range(0,len(mb)):
            qq=qq+pb[q]*(mb[1]-mb[0])
            fb[q]=qq
        Ab = np.max(fb)
        qq=0
        for q in range(0,len(mb)):
            qq=qq+pb[q]*(mb[1]-mb[0])/Ab
            Fb[q]=qq   
        #plt.figure()
        #plt.plot(mb,Fb,'b')
        #plt.show()
        vb = np.interp(0.16,Fb,mb)
        ub = np.interp(0.84,Fb,mb)
        err_bar_b[k,:] = [vb-bestfmb,ub-bestfmb]
    ##对应for循环前面的plt.figure
    #print('phy_mr_r=',Pr_mh_r[:,0])
    #print('phy_mb_b=',Pr_mh_b[:,1])
    #print('errbar_r=',err_bar_r)
    #print('errbar_b=',err_bar_b) 
    ###观察delta_x^2的变化情况
    plt.figure()
    for k in range(0,len(mr_be)+len(mb_be)+1):
        plt.subplot((len(mr_be)+len(mb_be)+1)/3,3,1+k)
        ###修改设置，让其根据输入质量区间画图
        if k<7:
            plt.plot(m,Dchi_r[k],'r')
            #plt.title(r'$\Delta\chi^2_{red}$')
            plt.grid()
            plt.axis([b_m_r[k]-0.3,b_m_r[k]+0.3,0,1])
        elif k<len(mr_be)+len(mb_be):
            plt.plot(m,Dchi_b[k-7],'b')
            #plt.title(r'$\Delta\chi^2_{red}$')
            plt.grid()
            plt.axis([b_m_b[k-7]-1,b_m_b[k-7]+0.5,0,1])
        else:
            plt.plot(b_m_r,chi_min_r,'r')
            plt.plot(b_m_b,chi_min_b,'b')
    plt.tight_layout()
    #plt.savefig('delta_x^2_r_comparation',dpi=600)
    plt.show()  
    plt.figure()
    #scatter函数参数设置
    color_list = ['r','b','g']
    #bar_list = ['s','^','o']
    line1,caps1,bars1=plt.errorbar(mbinr,Pr_mh_r,yerr=[err_bar_r[:,0],err_bar_r[:,1]],fmt="rs-",linewidth=1,
                                elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='red')
    line3,caps3,bars3=plt.errorbar(mbinb,Pr_mh_b,yerr=[err_bar_b[:,0],err_bar_b[:,1]],fmt="b^-",linewidth=1,
                                elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='blue')
    plt.fill_between(mbinr,Pr_mh_r+err_bar_r[:,0],Pr_mh_r+err_bar_r[:,1],
                         facecolor=color_list[0],alpha=.20)
    plt.fill_between(mbinb,Pr_mh_b+err_bar_b[:,0],Pr_mh_b+err_bar_b[:,1],
                         facecolor=color_list[1],alpha=.20)   
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$log{\langle M_{200} \rangle/[M_\odot h^{-1}]}$')    
    plt.legend(loc=2)
    plt.title(r'$Standard Deviation-1\sigma_p$')
    plt.axis([10,12.5,11,14.5])
    #plt.savefig('Standard_deviation_new_blue.png',dpi=600)
    plt.show()
    ##下面直接从x^2得到1-sigma的图像
    Dchir = np.zeros((len(mr_be),2),dtype=np.float)
    Dchib = np.zeros((len(mb_be),2),dtype=np.float) 
    #print('mr=',b_m_r)
    for k in range(0,len(mr_be)):
        mr_ = m
        ##把相应的x^2的差值分两个区间
        Dchir1_ = Dchi_r[k]
        ix = mr_ < b_m_r[k]
        Dchir_1 = Dchir1_[ix]
        mr1 = mr_[ix]
        Dchir_2 = Dchir1_[~ix]
        mr2 = mr_[~ix]
        ##下面求1-sigma的点
        m_r1 = np.interp(-1.0,-Dchir_1,mr1)
        m_r2 = np.interp(1.0,Dchir_2,mr2) 
        ##插值后比较
        Dchir[k,:] = np.array([m_r1-b_m_r[k],m_r2-b_m_r[k]])
    #print('mb=',b_m_b)
    for k in range(0,len(mb_be)-2):
    ##对于11.4——11.6和11.6——15.0的质量区间不能求出对应的拟合质量，
    ##这是因为数据点的给定并不能利用此计算给出结果，会出现不符合描述的x^2
        mb_ = m
        ##把相应的x^2的差值分两个区间
        Dchib1_ = Dchi_b[k]
        ix = mb_ < b_m_b[k]
        ###ix返回的是一组布尔型数据（True or False）
        Dchib_1 = Dchib1_[ix]
        mb1 = mb_[ix]
        Dchib_2 = Dchib1_[~ix]
        mb2 = mb_[~ix]
        ##下面求1-sigma的点
        m_b1 = np.interp(-1.0,-Dchib_1,mb1)
        m_b2 = np.interp(1.0,Dchib_2,mb2)
        ##插值后比
        Dchib[k,:] = np.array([m_b1-b_m_b[k],m_b2-b_m_b[k]])
    #print(Dchir)
    #print(Dchib)
    plt.figure()
    ##data存储观测数据
    ##########下面部分记录errorbar的画图与调整，fill_between的画图和调整
    data_r = np.array([12.17,12.14,12.50,12.89,13.25,13.63,14.05])
    data_r_err = np.array([[0.19,0.12,0.04,0.04,0.03,0.03,0.05],
                          [-0.24,-0.14,-0.05,-0.04,-0.03,-0.03,-0.05]])
    m_binr = np.array([10.28,10.58,10.86,11.10,11.29,11.48,11.68])
    data_b = np.array([11.80,11.73,12.15,12.61,12.69,12.79,12.79])
    data_b_err = np.array([[0.16,0.13,0.08,0.10,0.19,0.43,0.58],
                          [-0.20,-0.17,-0.10,-0.11,-0.25,-1.01,-2.23]])
    m_binb = np.array([10.24,10.56,10.85,11.10,11.28,11.47,11.68])
    ###把预测数据和观测data做对比
    Dchir = Dchir.T
    Dchib = Dchib.T
    line1,caps1,bars1=plt.errorbar(mbinr,Pr_mh_r,yerr=abs(Dchir)[::-1],fmt="rs-",linewidth=1,
                                elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='red')
    line2,caps2,bars2=plt.errorbar(m_binr,data_r,yerr=abs(data_r_err)[::-1],fmt="ro--",linewidth=1,
                                elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='red(M16)')
    line3,caps3,bars3=plt.errorbar(mbinb[0:-2],Pr_mh_b[0:-2],yerr=abs(Dchib[:,0:-2])[::-1],fmt="b^-",linewidth=1,
                                elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='blue')
    line4,caps4,bars3=plt.errorbar(m_binb,data_b,yerr=abs(data_b_err)[::-1],fmt="bo--",linewidth=1,
                                elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='blue(M16)')
    #plt.fill_between(mbinr,Pr_mh_r[:]+Dchir[0,:],Pr_mh_r[:]+Dchir[1,:],
    #                     facecolor=color_list[0],alpha=.30)
    #plt.fill_between(mbinb,Pr_mh_b[:]+Dchib[0,:],Pr_mh_b[:]+Dchib[1,:],
    #                     facecolor=color_list[1],alpha=.30)
    #plt.fill_between(m_binr,data_r[:]+data_r_err[0],data_r[:]+data_r_err[1],
    #                     facecolor=color_list[2],alpha=.20)
    #plt.fill_between(m_binb,data_b[:]+data_b_err[0],data_b[:]+data_b_err[1],
    #                     facecolor=color_list[2],alpha=.20)
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$log{\langle M_{200} \rangle/[M_\odot h^{-1}]}$')    
    plt.legend(loc=2)
    plt.title(r'$Standard Deviation-1\sigma_{\chi^2}$')
    #plt.axis([10,12.5,11,14.5])
    #plt.savefig('Prediction_correct.png',dpi=600)
    plt.show()
    print('errorbar_r=',Dchir)
    print('errorbar_b=',Dchib)
    print('Mh_r=',Pr_mh_r)
    print('Mh_b=',Pr_mh_b)
    ####下面对计算做假设检验
    from scipy.stats import chi2
    p_val_r = np.zeros(len(mr_be),dtype=np.float)
    p_val_b = np.zeros(len(mb_be),dtype=np.float)    
    for k in range(0,len(mr_be)):
        p_val_r[k] = 1-chi2.cdf(chi_min_r[k],14)
    for k in range(0,len(mb_be)):
        p_val_b[k] = 1-chi2.cdf(chi_min_b[k],14)
    print('Pvalue_r=',p_val_r)
    print('Pvalue_b=',p_val_b)    
    return Pr_mh_r,Pr_mh_b
fig_mass(pp=True)