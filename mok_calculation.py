##该脚本用于把模型计算和模拟数据对接，理论上应该获得透镜信号，并且透镜信号分布应该有两个序列
###对于该本份脚本的调用要求计算是基于共动坐标系，z=0,输入质量序列即输出透镜信号序列
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
import numpy as np
import matplotlib.pyplot as plt
#引入scipy库，考虑对预测数据做误差棒函数
#from scipy.stats import t as st
def semula_tion(m_,x,z):
    #考虑共动坐标分析，令z=0
    #z = 0
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
    R = np.linspace(0,10,1000)
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
        rouc = (Qc*(3*H**2)/(8*np.pi*G))/(1/(1+z))**3
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
    return inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma
#semula_tion(m_=True,x=True)
def f_plot(m,z=0):
    m_ = 1
    x = m
    z = z
    inm,Rp,r,rs,r_200,rho_R,sigma,deltasigma = semula_tion(m_,x,z)
    LL = len(x)
    for k in range(0,LL):
        x1 = Rp[k,:]
        y1 = deltasigma[k,:]
        #plt.plot(x1,y1,'-',label=x[k])
        plt.loglog(x1,y1,'*')
    #plt.legend(loc=3)
    plt.grid()
    #plt.xlabel(r'$lg(\frac{R}{rs})$')
    plt.xlabel('R-Mpc/h')
    plt.ylabel(r'$\lg(\Delta\Sigma(\frac{R}{rs}))-M_sMpc^{-2}$')
    plt.show()
    for k in range(0,LL):
        x2 = Rp[k,:]
        y2 = rho_R[k,:]
        #plt.plot(x1,y1,'-',label=x[k])
        plt.loglog(x2,y2,'-')
    #plt.legend(loc=3)
    plt.grid()
    #plt.xlabel(r'$lg(\frac{R}{rs})$')
    plt.xlabel('r-Mpc/h')
    plt.ylabel(r'$\lg(\rho(\frac{R}{rs}))-h^2M_sMpc^{-3}$')
    plt.show()    
    return 
#f_plot(m=True,z=True)