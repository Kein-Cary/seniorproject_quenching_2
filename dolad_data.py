import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
from read_m16 import test_read_m16_ds
#导入的质量区间为11~15dex-Msolar
import os.path
#该句表示路径包调用
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.stats import t as st
#画误差棒函数须引入scipy库
def dolad_data(m,hz):
    _mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/'
    fname = os.path.join(_mh_path,'data_mh1.txt')
    #上述两句表示文件调用并打开数据
    m = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)    
    print('m=',m)
    fname = os.path.join(_mh_path,'data_z.txt')
    hz = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)
    # print('hz=',hz)
    #plt.plot(m[0,:],m[1,:],label='Doload-successfully')
    #plt.legend()
    #plt.show()
    lmw = len(m[:,0])
    lml = len(m[0,:])
    m_dex = np.linspace(11,15,lmw)
    # print('lmw=',lmw)
    # print('lml=',lml)
    # print('dex=',m_dex)
    return(m,m_dex,lmw,lml)
#dolad_data(m=True,hz=True)

#sction2:数据处理
def semula_tion(m_,x):
    #考虑共动坐标分析，令z=0
    # hz[0] = 0
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
  #  global ms_
  #  ms_ = 1.989*10**(30)
  #  ms = ms_  
    global c_
    c_ = 6
    ##调试结果c=3.25~3.75之间比较合适
    c = c_
#下面开始计算
    #对导入的数据做单位转化转为:太阳质量/h
    print('mh=',m_)
    m1 = h*m_*10**x
    #m1 = m_*10**x
    LL = len(m1)
    R = np.linspace(0,100,1500)
    L = len(R)
    Rp = np.zeros((LL,L),dtype=np.float)
    g_x = np.zeros((LL,L),dtype=np.float)
    deltasigma = np.zeros((LL,L),dtype=np.float)
    sigma = np.zeros((LL,L),dtype=np.float)
    rs =  np.zeros(LL,dtype=np.float)
    r = np.zeros((LL,L),dtype=np.float)
    r_200 =  np.zeros(LL,dtype=np.float)
    rho_0 = np.zeros(LL,dtype=np.float)
    #检测质量
    inm = np.zeros(LL,dtype=np.float)
    #加入投射距离上的密度变化
    rho_R = np.zeros((LL,L),dtype=np.float)
    #单位修正因子
    #Qc = 0.1412#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    #Q200 = 0.7
    ##重新对单位修正因子订正如下：
    Qc = 3.084*10**(-2)/(1.989*h**2)#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    Q200 = 1#对r200的单位修正（此时对rs也做了修正）
    for n in range(0,LL):
        # E = np.sqrt(omegam*(1+hz[0]))
        # H = h*100*E
        H = h*100
        #修正
        #E = np.sqrt((1+z)**3*omegam)
        #H = h*100*E
        #
        rouc = Qc*(3*H**2)/(8*np.pi*G)
        roum = 200*rouc*omegam
        r_200[n] = Q200*(3*m1[n]/(4*np.pi*roum))**(1/3)
        rs[n] = r_200[n]/c
        rho_0[n] = m1[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rs[n]**3)
        for t in range(0,L):
            Rp[n,t] = R[t]*rs[n]
            r[n,t] = R[t]*rs[n]
            #加入对投射方向密度的计算
            rho_R[n,t] = rho_0[n]*rs[n]**3/(r[n,t]*(rs[n]+r[n,t])**2)
            #密度单位转化
            ##检测积分质量，并换位太阳质量单位
            inm[n] = (4*np.pi*rho_0[n]*rs[n]**3*(np.log(1+\
               r_200[n]/rs[n])-r_200[n]/(rs[n]+r_200[n])))/h
            ##
            #引入中间函数
            f0 = Rp[n,t]/rs[n]#这是考虑把R参数化的情况
            if Rp[n,t]<rs[n]:
               f1 = np.arctanh(np.sqrt((1-f0)/(1+f0)))
               f2 = np.log(f0/2)
               f4 = f0**2*np.sqrt(1-f0**2)
               f5 = (f0**2-1)*np.sqrt(1-f0**2)
               g_x[n,t] = 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
               delta_segma = rs[n]*rho_0[n]*g_x[n,t]
              # deltasigma[n,t] = delta_segma  #(原有计算,面积单位为Mpc^2)
              # 为了和观测对比，把单位转为Msun*h/pc^2,即把面积单位转为pc^2
               deltasigma[n,t] = delta_segma*10**(-12)
               sigma[n,t] = (2*rs[n]*rho_0[n]/(f0**2-1))*(1-2*f1/(np.sqrt(1-f0**2)))
            elif Rp[n,t]==rs[n]:
                 g_x[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs[n]*rho_0[n]*g_x[n,t]
                # deltasigma[n,t] = delta_segma  #(原有计算,面积单位为Mpc^2)
                # 为了和观测对比，把单位转为Msun*h/pc^2,即把面积单位转为pc^2
                 deltasigma[n,t] = delta_segma*10**(-12)
                 sigma[n,t] = 2*rs[n]*rho_0[n]/3
            else:
                 f1 = np.arctan(np.sqrt((f0-1)/(f0+1)))
                 f2 = np.log(f0/2)
                 f4 = f0**2*np.sqrt(f0**2-1)
                 f5 = (f0**2-1)**(3/2)
                 g_x[n,t]= 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
                 delta_segma = rs[n]*rho_0[n]*g_x[n,t]
                # deltasigma[n,t] = delta_segma  #(原有计算,面积单位为Mpc^2)
                # 为了和观测对比，把单位转为Msun*h/pc^2,即把面积单位转为pc^2
                 deltasigma[n,t] = delta_segma*10**(-12)
                 sigma[n,t] = 2*rs[n]*rho_0[n]*(1-2*f1/np.sqrt(f0**2-1))/(f0**2-1)
    return(inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma)
#semula_tion(m_=True,x=True)
##下面把deltasigma的计算转为Rp的函数关系，以方便调用
#def calcu_sigma(Rpc,m_,x):
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
  #  global ms_
  #  ms_ = 1.989*10**(30)
  #  ms = ms_  
    global c_
    c_ = 6
    ##调试结果c=3.25~3.75之间比较合适
    c = c_
#下面开始计算
    #对导入的数据做单位转化转为:太阳质量/h
    m1 = h*m_*10**x
    #m1 = m_*10**x
    # LL = len(m1)
    # R = np.linspace(0,100,1500)
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
    Q200 = 1#对r200的单位修正（此时对rs也做了修正）
    H = h*100
    #修正
    #E = np.sqrt((1+z)**3*omegam)
    #H = h*100*E
    #
    rouc = (Qc*(3*H**2)/(8*np.pi*G))/(1/(1+z))**3
    roum = 200*rouc*omegam
    r_200 = Q200*(3*m1/(4*np.pi*200*rouc))**(1/3)
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
    return(Rpc,Sigma,deltaSigma)
#calcu_sigmaz(Rpc=True,m_=True,x=True,z=True)
#单独调用模块时需要对此句修改，引入直接的Rp和质量
##
def calcu_sigma(Rpc,m_,x):
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
  #  global ms_
  #  ms_ = 1.989*10**(30)
  #  ms = ms_  
    global c_
    c_ = 6
    ##调试结果c=3.25~3.75之间比较合适
    c = c_
#下面开始计算
    #对导入的数据做单位转化转为:太阳质量/h
    m1 = h*m_*10**x
    #m1 = m_*10**x
    # LL = len(m1)
    # R = np.linspace(0,100,1500)
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
    #Qc = 3.084*10**(-2)/(1.989*h**4)#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    Q200 = 1#对r200的单位修正（此时对rs也做了修正）
    H = h*100
    #修正
    #E = np.sqrt((1+z)**3*omegam)
    #H = h*100*E
    #
    rouc = Qc*(3*H**2)/(8*np.pi*G)
    roum = 200*rouc*omegam
    r_200 = Q200*(3*m1/(4*np.pi*200*rouc))**(1/3)
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
    return(Rpc,Sigma,deltaSigma)
#calcu_sigma(Rpc=True,m_=True,x=True)   
##
def fig_f(inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma):
#def fig_f(ff):
#inm,Rs,r,rs,r_200,rho_R,sigma,deltasigma=semula_tion(m_,x)
    lm = len(inm)
    plt.figure()
    plt.subplot(121)
    # delta1 = np.zeros(lm,dtype=np.float)
    # delta2 = np.zeros(lm,dtype=np.float)
    for k in range(0,lm):
        plt.loglog(r[k,:],rho_R[k])
        '''
        delta1[k] = r_200[k]
        delta2[k] = rs[k]
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='blue')
        '''
    plt.xlabel(r'$r-Mpc/h$')
    plt.ylabel(r'$\rho(R)-M_\odot/h$')
    plt.grid()
    print('mh_inte=',inm)
    plt.subplot(122)
    #下面引入具体的deltasigma数据的观测值,和所给质量比较
    test_read_m16_ds()
    delta1 = np.zeros(lm,dtype=np.float)
    delta2 = np.zeros(lm,dtype=np.float)
    for k in range(0,lm):
        plt.loglog(Rp[k,:],deltasigma[k,:])
        delta1[k] = r_200[k]
        delta2[k] = rs[k]
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='blue')
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma-hM_\odot/{pc^2}$')
    plt.tight_layout()
    plt.show()
    out=[inm,rho_R,deltasigma,sigma]
    return(out)
#fig_f(inm=True,Rs=True,r=True,rs=True,r_200=True,rho_R=True,sigma=True,deltasigma=True)
#fig_f(ff=True)
##定义一个五个质量最佳预测和观测的对比图象
def fig_ff(Rpc,ds_sim,lmw,r):
    test_read_m16_ds()
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
    return()
#fig_ff(Rpc=True,ds_sim=True,lmw=True,r=True)
#fig_ff(f=True)
##定义一个质量最佳预测和观测的对比图象
def fig_fff1(Rpc,ds_sim,k):
    test_read_m16_ds()
    plt.plot(Rpc[:],ds_sim[k,:],'-*')
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
    return()
#fig_fff1(Rpc=True,ds_sim=True,lmw=True,r=True)
#fig_fff1(f=True)
def fig_fff2(Rpc,ds_sim,k,t):
    test_read_m16_ds()
    plt.plot(Rpc[:],ds_sim[k,t,:],'-*')
    nn = ds_sim[k,t,:].size#求样本大小
    x_mean = np.mean(ds_sim[k,t,:])#求算术平均值
    x_std = np.std(ds_sim[k,t,:])#求标准偏差
    x_se = x_std/np.sqrt(nn)#求标准误差
    dof = nn-1#自由度计算
    alpha = 1.0-0.95
    conf_region = st.ppf(1-alpha/2.,dof)*x_std*\
    np.sqrt(1.+1./nn)#设置置信区间
    plt.errorbar(Rpc[:],ds_sim[k,t,:],yerr=x_std,fmt='-',linewidth=0.5)
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma-hM_\odot/{pc^2}$')   
    return()
#fig_fff2(Rpc=True,ds_sim=True,lmw=True,r=True)
#fig_fff2(f=True)
##
def fit_datar(y):
    h = 0.673
    #先找出观测值对应的rp
    rsa,dssa,ds_errsa = test_read_m16_ds()
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    a = np.shape(dssa)
    print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    z_r = 0.230
    #z_r = 0.105
    a_r = 1/(1+z_r)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2])
   # rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
   # rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    print(rp)
    b = len(rp)
    m,m_dex,lmw,lml = dolad_data(m=True,hz=True)
    '''
    for k in range(0,lmw):
        m_ = m[k,:]
        x = m_dex[k]
      # return(semula_tion(m_,x))
      # semula_tion(m_,x) #单独运行该句需要把画图的模块并入semula_tion模块
        inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma=semula_tion(m_,x)
        fig_f(inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma)
    '''
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simr = np.zeros((lmw,lml,b),dtype=np.float)
    for k in range(0,lmw):
        for t in range(0,lml):
            m_ = m[k,t]
            x = m_dex[k]
            #Rpc = rp
            #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)
            #ds_simr[k,t,:] = deltaSigma
            #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
            Rpc = rp
            #计算模型在对应的投射距离上的预测信号取值
            #对预测信号做相应的变化，把共动的转为物理的
            Rpc,Sigma,deltaSigma = calcu_sigmaz(Rpc,m_,x,z_r)
            ds_simr[k,t,:] = deltaSigma    
    yy = np.shape(ds_simr)
    print(yy)#输出查看ds_sim的维度，即模型预测下的透镜信号    
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    fitr = np.zeros((lmw,2),dtype=np.int)
    delta_r = np.zeros((lmw,lml),dtype=np.float)
    for k in range(0,lmw):
        for t in range(0,lml):
            d_r = 0
            for n in range(0,yy[2]):
                d_r = d_r+((ds_simr[k,t,n]-dssa[0,n])/ds_errsa[0,n])**2
            delta_r[k,t] = d_r
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    for k in range(0,lmw):
        #先把np的定义数组转为列表，再寻找索引号，tolist()表示内置函数调用
        aa = delta_r[k,:].tolist()
        xa = aa.index(min(delta_r[k,:]))
        fitr[k,0] = k
        fitr[k,1] = xa
    print(fitr)
    #下面做图比较几个最佳预测值与观测的对比情况
    '''
    Rpc = rp
    plt.subplot(121)
    fig_ff(Rpc,ds_simr,lmw,fitr)
    plt.title('R-galaxy')
    plt.subplot(122)
    plt.plot(m_dex,np.log10(delta_r))
    plt.title('R-galaxy')
    plt.show() 
    '''
    #比较提取出的五个最小值的最小值为最后结果
    deltar = np.zeros(lmw,dtype=np.float)
    for k in range(0,lmw):
        deltar[k] = delta_r[fitr[k,0],fitr[k,1]]
    #下面用bestr和bestb记录最后比较结果
    bestfr = np.zeros(2,dtype=np.float)
    aa = deltar.tolist()
    xa = aa.index(min(deltar))
    bestfr = fitr[xa,:]  
    bestr = m[bestfr[0],bestfr[1]]
    best_mr = bestr
    dexr1 = m_dex[bestfr[0]]
    #下面两句是为与文献对比做的修正
    dexr = m_dex[bestfr[0]]+np.log10(bestr*h)
    print('mr=',best_mr) #此句质量计算正常
    print('positionr=',bestfr)
    print('dexr=',dexr1)
    print('co_dexr=',dexr)  
    print('corr_mr=',10**dexr)
    return rp,best_mr,dexr1,a,b,yy,dexr,a_r 
#fit_datar(y=True)
def fit_datab(y):
    h = 0.673
    #先找出观测值对应的rp
    rsa,dssa,ds_errsa = test_read_m16_ds()
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    a = np.shape(dssa)
    print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    z_b = 0.246
    #z_b = 0.124
    a_b = 1/(1+z_b)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]) 
    #rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
    #rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    print(rp)
    b = len(rp)
    m,m_dex,lmw,lml = dolad_data(m=True,hz=True)
    '''
    for k in range(0,lmw):
        m_ = m[k,:]
        x = m_dex[k]
      # return(semula_tion(m_,x))
      # semula_tion(m_,x) #单独运行该句需要把画图的模块并入semula_tion模块
        inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma=semula_tion(m_,x)
        fig_f(inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma)
    '''
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simb = np.zeros((lmw,lml,b),dtype=np.float)
    for k in range(0,lmw):
        for t in range(0,lml):
            m_ = m[k,t]
            x = m_dex[k]
            #Rpc = rp
            #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)
            #ds_simb[k,t,:] = deltaSigma
            #对比文献，把物理距离转为共动距离
            Rpc = rp
            #计算模型在对应的投射距离上的预测信号取值
            #预测信号也做修正
            Rpc,Sigma,deltaSigma = calcu_sigmaz(Rpc,m_,x,z_b)
            ds_simb[k,t,:] = deltaSigma
    yy = np.shape(ds_simb)
    print(yy)#输出查看ds_sim的维度，即模型预测下的透镜信号    
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    fitb = np.zeros((lmw,2),dtype=np.int)
    delta_b = np.zeros((lmw,lml),dtype=np.float)
    for k in range(0,lmw):
        for t in range(0,lml):
            d_b = 0
            for n in range(0,yy[2]):
                d_b = d_b+((ds_simb[k,t,n]-dssa[1,n])/ds_errsa[1,n])**2
            delta_b[k,t] = d_b
    #print(delta_b)
    #下面这段求每个质量级对应的方差最小值
    for k in range(0,lmw):
        #先把np的定义数组转为列表，再寻找索引号，tolist()表示内置函数调用
        bb = delta_b[k,:].tolist()
        xb = bb.index(min(delta_b[k,:]))
        fitb[k,0] = k
        fitb[k,1] = xb
    print(fitb)
    #下面做图比较几个最佳预测值与观测的对比情况
    '''
    Rpc = rp
    plt.subplot(121)
    fig_ff(Rpc,ds_simb,lmw,fitb)
    plt.title('B-galaxy')
    plt.subplot(122)
    plt.plot(m_dex,np.log10(delta_b))
    plt.title('B-galaxy')
    plt.show() 
    '''
    #比较提取出的五个最小值的最小值为最后结果
    deltab = np.zeros(lmw,dtype=np.float)
    for k in range(0,lmw):
        deltab[k] = delta_b[fitb[k,0],fitb[k,1]]
    #下面用bestr和bestb记录最后比较结果
    bestfb = np.zeros(2,dtype=np.float)
    bb = deltab.tolist()
    xb = bb.index(min(deltab))
    bestfb = fitb[xb,:]    
    bestb = m[bestfb[0],bestfb[1]]
    best_mb = bestb
    dexb1 = m_dex[bestfb[0]]
    #下面是与文献对比做的修正
    dexb = m_dex[bestfb[0]]+np.log10(bestb*h)
    print('mb=',best_mb) #此句质量计算正常
    print('positionb=',bestfb)
    print('dexb=',dexb1)
    print('co_dexb=',dexb) 
    print('corr_mb=',10**dexb)
    return rp,best_mb,dexb1,a,b,yy,dexb,a_b 
fit_datab(y=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''