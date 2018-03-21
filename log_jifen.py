import os.path
import numpy as np
import matplotlib.pyplot as plt
#section1:数据导入
def dolad_data(m,hz):
    _mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/'
    fname = os.path.join(_mh_path,'data_m.txt')
    m = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)  
    print('m=',m)
    fname = os.path.join(_mh_path,'data_z.txt')
    hz = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)
    print('hz=',hz)
    plt.plot(m,m,'r',label=r'$load Succesful$')
    plt.legend()
    plt.show()
    return(m,hz)
#dolad_data(m=True,hz=True)
#sction2:数据处理
def semula_tion(omegam,h):
    m,hz = dolad_data(m=True,hz=True)
    m_ = m
    #考虑共动坐标分析，令z=0
    hz[0] = 0
    #
    global omega_m
    omega_m = 0.315
    omegam = omega_m
#a flat CDM model,omegak=0;omegalamd=0
    global h_
    h_ = 0.70
    h = h_
    global G_
    G_ = 6.67*10**(-11)
    G = G_
    '''
    global ms_
    ms_ = 1.989*10**(30)
    ms = ms_  
    '''
    global c_
    c_ = 2.5
    c = c_
#下面开始计算
    LL = len(m)
    R = np.linspace(0,100,10000)
    L = len(R)
    Rs = np.zeros((LL,L),dtype=np.float)
    g_x = np.zeros((LL,L),dtype=np.float)
    deltasegma = np.zeros((LL,L),dtype=np.float)
    segma = np.zeros((LL,L),dtype=np.float)
    rs =  np.zeros(LL,dtype=np.float)
    r = np.zeros((LL,L),dtype=np.float)
    r_200 =  np.zeros(LL,dtype=np.float)
    rou_0 = np.zeros(LL,dtype=np.float)
    #检测质量
    inm = np.zeros(LL,dtype=np.float)
    #加入投射距离上的密度变化
    rou_R = np.zeros((LL,L),dtype=np.float)
    #对导入的数据做单位转化转为:太阳质量/h
    m = 0.7*10**m_
    #单位修正因子
    Qc = 0.1412#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    Q200 = 0.7#对r200的单位修正（此时对rs也做了修正）
    #
    for n in range(0,LL):
       # E = np.sqrt(omegam*(1+hz[0]))
       # H = h*100*E
        H = h*100
        rouc = Qc*(3*H**2)/(8*np.pi*G)
        roum = 200*rouc*omegam
        r_200[n] = Q200*(3*m[n]/(4*np.pi*roum))**(1/3)
        rs[n] = r_200[n]/c
        rou_0[n] = m[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rs[n]**3)
        for t in range(0,L):
            Rs[n,t] = R[t]*rs[n]
            r[n,t] = R[t]*rs[n]
            #加入对投射方向密度的计算
            rou_R[n,t] = rou_0[n]*rs[n]**3/(r[n,t]*(rs[n]+r[n,t])**2)
            #密度单位转化
            ##检测积分质量，并换位太阳质量单位
            inm[n] = (4*np.pi*rou_0[n]*rs[n]**3*(np.log(1+\
               r_200[n]/rs[n])-r_200[n]/(rs[n]+r_200[n])))/0.7
            ##
            #引入中间函数
            f0 = Rs[n,t]/rs[n]#这是考虑把R参数化的情况
            if Rs[n,t]<rs[n]:
               f1 = np.arctanh(np.sqrt((1-f0)/(1+f0)))
               f2 = np.log(f0/2)
               f4 = f0**2*np.sqrt(1-f0**2)
               f5 = (f0**2-1)*np.sqrt(1-f0**2)
               g_x[n,t] = 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
               delta_segma = rs[n]*rou_0[n]*g_x[n,t]
               deltasegma[n,t] = delta_segma
               segma[n,t] = (2*rs[n]*rou_0[n]/(f0**2-1))*(1-2*f1/(np.sqrt(1-f0**2)))
            elif Rs[n,t]==rs[n]:
                 g_x[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs[n]*rou_0[n]*g_x[n,t]
                 deltasegma[n,t] = delta_segma
                 segma[n,t] = 2*rs[n]*rou_0[n]/3
            else:
                 f1 = np.arctan(np.sqrt((f0-1)/(f0+1)))
                 f2 = np.log(f0/2)
                 f4 = f0**2*np.sqrt(f0**2-1)
                 f5 = (f0**2-1)**(3/2)
                 g_x[n,t]= 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
                 delta_segma = rs[n]*rou_0[n]*g_x[n,t]
                 deltasegma[n,t] = delta_segma
                 segma[n,t] = 2*rs[n]*rou_0[n]*(1-2*f1/np.sqrt(f0**2-1))/(f0**2-1)
    plt.figure()
    plt.subplot(131)
    delta1 = np.zeros(LL,dtype=np.float)
    delta2 = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):
        plt.loglog(r[k,:],rou_R[k])
        delta1[k] = r_200[k]
        delta2[k] = 0
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=5,color='blue')
    plt.xlabel(r'$r-Mpc/h$')
    plt.ylabel(r'$\rho(R)-M_\odot/h$')
    plt.grid()
    print('mh=',inm)
    #下面的模块做正确性检查，理论上入股从deltasegma的积分可以得到mh 
    ##对密度rou空间积分检测
    Mm = np.zeros((LL,L),dtype=np.float)
    max_m = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):
        dx = r[k,1]-r[k,0]
       # Mm[k,0] = 4*np.pi*(rou_R[k,0]*r[k,0]**2)*dx
        for t in range(0,L):
            if r[k,t]<=c*rs[k] and t>0:
               Mm[k,t] = Mm[k,t-1]+4*np.pi*(rou_R[k,t]*r[k,t]**2)*dx
        max_m[k] = np.max(Mm[k,:])
    print('max_m=',max_m)
    print('mh=',max_m/0.7)
    #print(r_200)
    plt.subplot(1,3,2)
    delta1 = np.zeros(LL,dtype=np.float)
    delta2 = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):
        plt.loglog(r[k,:],Mm[k,:]/0.7)
        delta1[k] = r_200[k]
        delta2[k] = 0
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
    plt.xlabel(r'$r-Mpc/h$')
    plt.ylabel(r'$mh-M_\odot/h$')
    plt.grid()
    plt.subplot(1,3,3)
    delta1 = np.zeros(LL,dtype=np.float)
    delta2 = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):
        plt.loglog(Rs[k,:],segma[k,:])
        delta1[k] = r_200[k]
        delta2[k] = 0
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
    plt.xlabel(r'$r-Mpc/h$')
    plt.ylabel(r'$\Sigma-M_\odot-h/{Mpc^2}$')
    plt.grid()
    plt.tight_layout()
    plt.show()
    #下面做对数空间积分
    '''
    N=10**(-6)
    s_bin = np.linspace(N,100,10000)
    l_s = len(s_bin)
    r_200g = np.zeros(LL,dtype=np.float)
    rsg = np.zeros(LL,dtype=np.float)
    rou_0g = np.zeros(LL,dtype=np.float)
    Rsg = np.zeros((LL,l_s),dtype=np.float)
    Rsg = np.zeros((LL,l_s),dtype=np.float)
    rou_Rg = np.zeros((LL,l_s),dtype=np.float)
    rg = np.zeros((LL,l_s),dtype=np.float)
    g_xg = np.zeros((LL,l_s),dtype=np.float)
    deltasegmag = np.zeros((LL,l_s),dtype=np.float)
    segmag = np.zeros((LL,l_s),dtype=np.float)
    for n in range(0,LL):
       # E = np.sqrt(omegam*(1+hz[0]))
       # H = h*100*E
        H = h*100
        roucg = Qc*(3*H**2)/(8*np.pi*G)
        roumg = 200*roucg*omegam
        r_200g[n] = Q200*(3*m[n]/(4*np.pi*roumg))**(1/3)
        rsg[n] = r_200g[n]/c
        rou_0g[n] = m[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rsg[n]**3)
        for t in range(0,l_s):
            Rsg[n,t] = s_bin[t]*rs[n]
            rg[n,t] = s_bin[t]*rs[n]
            #加入对投射方向密度的计算
            rou_Rg[n,t] = rou_0g[n]*rsg[n]**3/(rg[n,t]*(rsg[n]+rg[n,t])**2)
            #引入中间函数
            f0g = Rsg[n,t]/rsg[n]#这是考虑把R参数化的情况
            if Rsg[n,t]<rsg[n]:
               f1g = np.arctanh(np.sqrt((1-f0g)/(1+f0g)))
               f2g = np.log(f0g/2)
               f4g = f0g**2*np.sqrt(1-f0g**2)
               f5g = (f0g**2-1)*np.sqrt(1-f0g**2)
               g_xg[n,t] = 8*f1g/f4g+4*f2g*f0g**(-2)-2/(f0g**2-1)+4*f1g/f5g
               delta_segma = rsg[n]*rou_0g[n]*g_xg[n,t]
               deltasegmag[n,t] = delta_segma
               segmag[n,t] = (2*rsg[n]*rou_0g[n]/(f0g**2-1))*(1-2*f1g/(np.sqrt(1-f0g**2)))
            elif Rsg[n,t]==rsg[n]:
                 g_xg[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs[n]*rou_0[n]*g_xg[n,t]
                 deltasegmag[n,t] = delta_segma
                 segmag[n,t] = 2*rsg[n]*rou_0g[n]/3
            else:
                 f1g = np.arctan(np.sqrt((f0g-1)/(f0g+1)))
                 f2g = np.log(f0g/2)
                 f4g = f0g**2*np.sqrt(f0g**2-1)
                 f5g = (f0g**2-1)**(3/2)
                 g_xg[n,t]= 8*f1g/f4g+4*f2g*f0g**(-2)-2/(f0g**2-1)+4*f1g/f5g
                 delta_segma = rsg[n]*rou_0g[n]*g_xg[n,t]
                 deltasegmag[n,t] = delta_segma
                 segmag[n,t] = 2*rsg[n]*rou_0g[n]*(1-2*f1g/np.sqrt(f0g**2-1))/(f0g**2-1)
'''
##先做区间划分，再在次基础上积分

##
semula_tion(omegam=True,h=True)