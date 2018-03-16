import os.path
import numpy as np
import matplotlib.pyplot as plt
#section1:数据导入
def dolad_data(m,hz):
    '''
    mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/data_mh.txt/'
    '''
    _mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/'
    fname = os.path.join(_mh_path,'data_m.txt')
    m = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)  
    '''
    mr = m('mr')
    mb = m('mb')
    print('mr=',mr)
    print('mb=',mb)
    '''
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
    global ms_
    ms_ = 1.989*10**(30)
    ms = ms_
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
    #加入投射距离上的密度变化
    rou_R = np.zeros((LL,L),dtype=np.float)
    #
    for n in range(0,LL):
        E = np.sqrt(omegam*(1+hz[n]))
        H = h*100*E
        rouc = (3*H**2)/(8*np.pi*G*10**(-9))
        roum = 200*rouc*omegam
        r_200 = (3*10**m[n]*ms/(4*np.pi*roum))**(1/3)
        rs[n] = r_200/c
        rou_0 = 10**m[n]*ms/((np.log(1+c)-c/(1+c))*4*np.pi*rs[n]**3)

        for t in range(0,L):
            Rs[n,t] = R[t]*rs[n]
            #加入对投射方向密度的计算
            rou_R[n,t] = rou_0*rs[n]**3/(Rs[n,t]*(rs[n]+Rs[n,t])**2)
            #
            #引入中间函数
            f0 = Rs[n,t]/rs[n]
            if Rs[n,t]<rs[n]:
               f1 = np.arctanh(np.sqrt((1-f0)/(1+f0)))
               f2 = np.log(f0/2)
               f4 = f0**2*np.sqrt(1-f0**2)
               f5 = (f0**2-1)*np.sqrt(1-f0**2)
               g_x[n,t] = 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
               delta_segma = rs[n]*rou_0*g_x[n,t]
               deltasegma[n,t] = np.log10(delta_segma/ms)
            elif Rs[n,t]==rs[n]:
                 g_x[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs[n]*rou_0*g_x[n,t]
                 deltasegma[n,t] = np.log10(delta_segma/ms)
            else:
                 f1 = np.arctan(np.sqrt((f0-1)/(f0+1)))
                 f2 = np.log(f0/2)
                 f4 = f0**2*np.sqrt(f0**2-1)
                 f5 = (f0**2-1)**(3/2)
                 g_x[n,t]= 8*f1/f4+4*f2*f0**(-2)-2/(f0**2-1)+4*f1/f5
                 delta_segma = rs[n]*rou_0*g_x[n,t]
                 deltasegma[n,t] = np.log10(delta_segma/ms)
    plt.figure()
    for k in range(0,LL):
        x = np.log10(Rs[k,:]/rs[k])
        y = deltasegma[k,:]
        plt.plot(x,y,'-')
        plt.grid()
    plt.xlabel(r'$lg(\frac{R}{rs})$')
    plt.ylabel(r'$\lg(\Delta\Sigma(\frac{R}{rs}))-M_sMpc^{-2}$')
    plt.hold()
    plt.show()
    
    plt.figure()
    plt.loglog(Rs[1,:]/ms,rou_R[1]/ms)
    plt.xlabel(r'$R$')
    plt.ylabel(r'$\rho(r)$')
    plt.grid()
semula_tion(omegam=True,h=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''