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
    R = np.linspace(0,100,1000)
    L = len(R)
    Rs = np.zeros((LL,L),dtype=np.float)
    g_x = np.zeros((LL,L),dtype=np.float)
    deltasegma = np.zeros((LL,L),dtype=np.float)
    rs =  np.zeros(LL,dtype=np.float)
    r_200 =  np.zeros(LL,dtype=np.float)
    #加入投射距离上的密度变化
    rouR = np.zeros((LL,L),dtype=np.float)
    #
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
        rou_0 = m[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rs[n]**3)

        for t in range(0,L):
            Rs[n,t] = R[t]*rs[n]
            rou_R = rou_0/((Rs[n,t]*(rs[n]+Rs[n,t])**2)/rs[n]**3)
            #把rou的参数也无量纲化
            rouR[n,t] = rou_R
            #
            f0 = Rs[n,t]/rs[n]
            if Rs[n,t]<rs[n]:
               f1 = np.arctanh(np.sqrt((1-f0)/(1+f0)))
               f2 = np.log(f0/2)
               f4 = f0**2*np.sqrt(1-f0**2)
               f5 = (f0**2-1)*np.sqrt(1-f0**2)
               g_x[n,t] = 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
               delta_segma = rs[n]*rou_0*g_x[n,t]
             #  deltasegma[n,t] = np.log10(delta_segma*Q0)
               deltasegma[n,t] = delta_segma
            elif Rs[n,t]==rs[n]:
                 g_x[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs[n]*rou_0*g_x[n,t]
             #  deltasegma[n,t] = np.log10(delta_segma*Q0)
                 deltasegma[n,t] = delta_segma
            else:
                 f1 = np.arctan(np.sqrt((f0-1)/(f0+1)))
                 f2 = np.log(f0/2)
                 f4 = f0**2*np.sqrt(f0**2-1)
                 f5 = (f0**2-1)**(3/2)
                 g_x[n,t]= 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
                 delta_segma = rs[n]*rou_0*g_x[n,t]
              #   deltasegma[n,t] = np.log10(delta_segma*Q0)
                 deltasegma[n,t] = delta_segma
    #下面的模块做正确性检查，理论上入股从deltasegma的积分可以得到mh    
    #先找出最接近r200的点，然后从曲线开始点积分到该点
    s_mm = np.zeros((LL,L-1),dtype=float)
    smm = np.zeros((LL,L-1),dtype=float)
    smm1 = np.zeros((LL,L-1),dtype=float)
    Rs1 = Rs[:,1:L]#这句表示对RS这个数组，每行都只取第二个到最后一个
    Delta_sigma = deltasegma[:,1:L]
    for k in range(0,LL):
        s_mm[k,0] = Delta_sigma[k,0]*(Rs1[k,0]/rs[k])
        for t in range(0,L):
               if Rs[k,t]<=c*r_200[k] and t>0:
                  s_mm[k,t] = s_mm[k,t-1]+Delta_sigma[k,t]*((Rs1[k,t]/rs[k])-\
                      (Rs1[k,t-1]/rs[k]))
                  smm[k,t] = s_mm[k,t]*rs[k]**2*2*np.pi      
    plt.figure()
    plt.subplot(1,2,1)
    for d in range(0,LL):
        plt.loglog(Rs1[d,:],smm[d,:])
        plt.grid()
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$m_h$')
    smm1 = smm*0.7
    plt.subplot(1,2,2)
    for d in range(0,LL):
        plt.loglog(Rs1[d,:],smm1[d,:])
        plt.grid()
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$m_h$')
    plt.tight_layout()
    plt.show()
    #检测结果发现从面密度回到质量估计时质量估计值偏大了一个数量级,返回修正因子，参数正常
    #下满把结果可视化
    plt.figure()
    fig,axes = plt.subplots(nrows=2,ncols=2)
    #设置子图的长宽高间隔
    # fig.tight_layout(pad=1.08,h_pad=None,w_pad=None,rect=None)
    fig.tight_layout(pad=2.0,w_pad=2.0,h_pad=2.0)
    #
    ax1 = plt.subplot(221)

    for k in range(0,LL):
        ax1.loglog(Rs[k,:],rouR[k])  
    ax1.set_xlabel(r'$R-Mpc/h$')
    ax1.set_ylabel(r'$\rho(R)-M_\odot/h$')
    #ax1.set_title(r'$\rho(R)-R$',fontsize=10)
    plt.grid()
    '''
    for k in range(0,LL):
        ax1.loglog(Rs[k,:]/rs[k],rouR[k])  
    ax1.set_xlabel(r'$\frac{R}{rs}$')
    ax1.set_ylabel(r'$\rho(R)-M_\odot/h$')
    #ax1.set_title(r'$\rho(R)-R$',fontsize=10)
    plt.grid()
    '''
    ax2 = plt.subplot(223)
    '''
    for k in range(0,LL):
        x0 = Rs[k,:]/rs[k]
        y0 = deltasegma[k,:]
        ax2.loglog(x0,y0,'-')
    #ax2.set_title(r'$\Delta\Sigma(\frac{R}{rs})-\frac{R}{rs}$',fontsize=10)
    plt.grid()
    ax2.set_xlabel(r'$\frac{R}{rs}$')
    ax2.set_ylabel(r'$\Delta\Sigma(\frac{R}{rs})-M_{\odot}hMpc^{-2}$')
    '''
    for k in range(0,LL):
        x0 = Rs[k,:]
        y0 = deltasegma[k,:]
        ax2.loglog(x0,y0,'-')
    #ax2.set_title(r'$\Delta\Sigma(\frac{R}{rs})-\frac{R}{rs}$',fontsize=10)
    plt.grid()
    ax2.set_xlabel(r'$R-Mpc/h$')
    ax2.set_ylabel(r'$\Delta\Sigma(\frac{R}{rs})-M_{\odot}hMpc^{-2}$')
#下面针对10^13太阳质量的情况具体分析
    #把halo的聚集程度改为变化的值
    c_v = [3,6,9]
    m_13 = m[2]
    rs13 = np.zeros(LL,dtype=np.float)
    Rs13 = np.zeros((LL,L),dtype=np.float)
    rouR13 = np.zeros((LL,L),dtype=np.float)
    g_x13 = np.zeros((LL,L),dtype=np.float)
    deltasegma13 = np.zeros((LL,L),dtype=np.float)
    MM = len(c_v)
    for n in range(0,MM):
       # E = np.sqrt(omegam*(1+hz[0]))
       # H = h*100*E
        H = h*100
        rouc13 = Qc*(3*H**2)/(8*np.pi*G)
        roum13 = 200*rouc13*omegam
        r_20013 = Q200*(3*m_13/(4*np.pi*roum13))**(1/3)
        rs13[n] = r_20013/c_v[n]
        rou_0_13 = m_13/((np.log(1+c_v[n])-c_v[n]/(1+c_v[n]))*4*np.pi*rs13[n]**3)
        for t in range(0,L):
            Rs13[n,t] = R[t]*rs13[n]
            '''
            #加入对投射方向密度的计算
            rouR13 = rou_0_13*rs13[n]**3/(Rs13[n,t]*(rs13[n]+Rs13[n,t])**2)
            #密度单位转化
            rou_R13[n,t] = rouR13
            '''
            rou_R13 = rou_0_13/((Rs13[n,t]*(rs13[n]+Rs13[n,t])**2)/rs13[n]**3)
            #把rou13参数化
            rouR13[n,t] = rou_R13
            #
            #引入中间函数
            f013 = Rs13[n,t]/rs13[n]
            if Rs13[n,t]<rs13[n]:
               f13 = np.arctanh(np.sqrt((1-f013)/(1+f013)))
               f23 = np.log(f013/2)
               f43 = f013**2*np.sqrt(1-f013**2)
               f53 = (f013**2-1)*np.sqrt(1-f013**2)
               g_x13[n,t] = 8*f13/f43+4*f23*f013**(-2)-2/(f013**2-1)+4*f13/f53
               delta_segma = rs13[n]*rou_0_13*g_x13[n,t]
               deltasegma13[n,t] = delta_segma
            elif Rs13[n,t]==rs13[n]:
                 g_x13[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs13[n]*rou_0_13*g_x13[n,t]
                 deltasegma13[n,t] = delta_segma
            else:
                 f13 = np.arctan(np.sqrt((f013-1)/(f013+1)))
                 f23 = np.log(f013/2)
                 f43 = f013**2*np.sqrt(f013**2-1)
                 f53 = (f013**2-1)**(3/2)
                 g_x13[n,t]= 8*f13/f43+4*f23*f013**(-2)-2/(f013**2-1)+4*f13/f53
                 delta_segma = rs13[n]*rou_0_13*g_x13[n,t]
                 deltasegma13[n,t] = delta_segma
    ax3 = plt.subplot(222)
    for k in range(0,MM):
        plt.loglog(Rs13[k,:],rouR13[k])
    ax3.set_xlabel(r'$R-Mpc/h$')
    ax3.set_ylabel(r'$\rho(R)-M_\odot/h$')
    #ax3.set_title(r'$\rho(R)-R$',fontsize=10)
    plt.grid()
    '''
    for k in range(0,LL):
        # plt.loglog(Rs13[k,:],rouR13[k])
        plt.loglog(Rs13[k,:]/rs13[k],rouR13[k,:])  
    ax3.set_xlabel(r'$\frac{R}{rs}$')
    ax3.set_ylabel(r'$\rho(R)-M_\odot/h$')
    #ax3.set_title(r'$\rho(R)-R$',fontsize=10)
    plt.grid()
    '''
    ax4 = plt.subplot(224)
    '''
    for k in range(0,MM):
        x1 = Rs13[k,:]/rs13[k]
        y1 = deltasegma13[k,:]
        plt.loglog(x1,y1)
    #ax4.set_title(r'$\Delta\Sigma(\frac{R}{rs})-\frac{R}{rs}$',fontsize=10) 
    plt.grid()
    ax4.set_xlabel(r'$\frac{R}{rs}$')
    ax4.set_ylabel(r'$\Delta\Sigma(\frac{R}{rs}))-M_{\odot}hMpc^{-2}$')
    '''
    for k in range(0,LL):
        x1 = Rs13[k,:]
        y1 = deltasegma13[k,:]
        plt.loglog(x1,y1)
    #ax4.set_title(r'$\Delta\Sigma(\frac{R}{rs})-\frac{R}{rs}$',fontsize=10) 
    plt.grid()
    ax4.set_xlabel(r'$R$')
    ax4.set_ylabel(r'$\Delta\Sigma-M_{\odot}hMpc^{-2}$')
plt.show()
#
semula_tion(omegam=True,h=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''