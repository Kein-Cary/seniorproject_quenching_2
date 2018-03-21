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
    ##
    #先找出最接近r200的点，然后从曲线开始点积分到该点
    s_mm = np.zeros((LL,L-1),dtype=np.float)
    smm = np.zeros((LL,L-1),dtype=np.float)
    smm1 = np.zeros((LL,L-1),dtype=np.float)
    Rs1 = Rs[:,1:L]#这句表示对RS这个数组，每行都只取第二个到最后一个
    sigma = segma[:,1:L]
    max_d = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):
        dx = (Rs1[k,1]/rs[k])-(Rs1[k,0]/rs[k])
        s_mm[k,0] = sigma[k,0]*(Rs1[k,0]/rs[k])*dx
        for t in range(0,L):
               if Rs[k,t]<=c*rs[k] and t>0:
                  s_mm[k,t] = s_mm[k,t-1]+sigma[k,t]*(Rs1[k,t]/rs[k])*dx
                  smm[k,t] = s_mm[k,t]*rs[k]**2*2*np.pi    
        max_d[k] = np.max(smm[k,:])
    plt.figure()
    plt.subplot(2,2,1)
    for d in range(0,LL):
        plt.loglog(Rs1[d,:],smm[d,:])
        plt.grid()
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$m_h$')
    print('smm=',max_d)
    smm1 = smm
    plt.subplot(2,2,2)
    delta1 = np.zeros(LL,dtype=np.float)
    delta2 = np.zeros(LL,dtype=np.float)
    for d in range(0,LL):
        plt.loglog(Rs1[d,:],smm1[d,:])
        plt.grid()
        delta1[k] = r_200[k]
        delta2[k] = 0
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$m_h$')
    #plt.title('D-i')#直接积分
    #检测结果发现从面密度回到质量估计时质量估计值偏大了一个数量级,返回修正因子，参数正常  
    ##下面是对结果做插值和积分，再回去检测质量是否相等
    N = 10**5
    Rsnew = np.zeros((LL,N),dtype=np.float)
    segma_p = np.zeros((LL,N),dtype=np.float)
    for k in range(0,LL):
        Rsnew[k,:] = np.linspace(Rs1[k,0],Rs1[k,-1],N)
        segma_p[k,:] = np.interp(Rsnew[k,:],Rs1[k,:],sigma[k,:])
    plt.subplot(2,2,3)
    delta1 = np.zeros(LL,dtype=np.float)
    delta2 = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):       
        x3 = Rsnew[k,:]
        y3 = segma_p[k,:]
        plt.loglog(x3,y3)
        delta1[k] = r_200[k]
        delta2[k] = 0
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
        plt.grid()
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma(\frac{R}{rs})-M_{\odot}hMpc^{-2}$')
    ##插值完成
    ##下面做辛普森积分
    smm2 = np.zeros((LL,N),dtype=np.float)
    smm3 = np.zeros((LL,N),dtype=np.float)
    dx = np.zeros(LL,dtype=np.float)
    Rsnew1 = Rsnew
    segma1 = segma_p
    max_s = np.zeros(LL,dtype=np.float)
    for k in range(0,LL):
        dx[k] = (Rsnew[k,1]-Rsnew[k,0])/rs[k]
        smm2[k,0] = (segma1[k,0]*(Rsnew[k,0]/rs[k])+4*segma1[k,1]\
            *(Rsnew[k,1]/rs[k])+segma1[k,2]*(Rsnew[k,t]/rs[k]))*(dx[k]/6)
        # for t in range(0,N):
        for t in range(0,N):
            if Rsnew1[k,t]<=c*rs[k] and t>0:
               smm2[k,t] = smm2[k,t-1]+(segma1[k,t-1]*(Rsnew[k,t]/rs[k])\
                   +4*segma1[k,t]*(Rsnew[k,t]/rs[k])+\
                   segma1[k,t+1]*(Rsnew[k,t]/rs[k]))*(dx[k]/6)
               smm3[k,t] = smm2[k,t]*rs[k]**2*2*np.pi 
        max_s[k] = np.max(smm3[k,:])
    print('smm3=',max_s)
    plt.subplot(2,2,4)
    delta1 = np.zeros(LL,dtype=np.float)
    delta2 = np.zeros(LL,dtype=np.float)
    for d in range(0,LL):
        plt.loglog(Rsnew1[d,:],smm3[d,:])
        delta1[k] = r_200[k]
        delta2[k] = 0
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='blue')
        plt.grid()
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$m_h$')
    plt.tight_layout()
    #plt.title('S-i')#辛普森积分
    plt.show() 
    #下满把结果可视化
    plt.figure()
    fig,axes = plt.subplots(nrows=2,ncols=2)
    #设置子图的长宽高间隔
    # fig.tight_layout(pad=1.08,h_pad=None,w_pad=None,rect=None)
    fig.tight_layout(pad=2.0,w_pad=2.0,h_pad=2.0)
    #
    ax1 = plt.subplot(221)
    for k in range(0,LL):
        ax1.loglog(r[k,:],rou_R[k])  
    ax1.set_xlabel(r'$r-Mpc/h$')
    ax1.set_ylabel(r'$\rho(R)-M_\odot/h$')
    #ax1.set_title(r'$\rho(R)-R$',fontsize=10)
    plt.grid()
    '''
    for k in range(0,LL):
        ax1.loglog(r[k,:]/rs[k],rouR[k])  
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
    r13 = np.zeros((LL,L),dtype=np.float)
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
            r13[n,t] = R[t]*rs13[n]
            rou_R13 = rou_0_13/((r13[n,t]*(rs13[n]+r13[n,t])**2)/rs13[n]**3)
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
        plt.loglog(r13[k,:],rouR13[k])
    ax3.set_xlabel(r'$r-Mpc/h$')
    ax3.set_ylabel(r'$\rho(R)-M_\odot/h$')
    #ax3.set_title(r'$\rho(R)-R$',fontsize=10)
    plt.grid()
    '''
    for k in range(0,LL):
        # plt.loglog(Rs13[k,:],rouR13[k])
        plt.loglog(rs13[k,:]/rs13[k],rouR13[k,:])  
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