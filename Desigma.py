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
    print('m=',m)
    fname = os.path.join(_mh_path,'data_z.txt')
    hz = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)
    print('hz=',hz)
    plt.plot(m,hz,'r',label=r'$load Succesful$')
    plt.legend()
    plt.show()
    return(m,hz)
#dolad_data(m=True,hz=True)
#sction2:数据处理
def semula_tion(omegam,h):
    m,hz = dolad_data(m=True,hz=True)
    hz[0] = 0
    #共动坐标下分析2
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
    for n in range(0,LL):
        #E = np.sqrt(omegam*(1+hz[n]))
        H = h*100
        rouc = (3*H**2)/(8*np.pi*G*10**(-9))
        roum = rouc*omegam
        r_200 = (3*10**m[n]*ms/(4*np.pi*roum))**(1/3)
        rs[n] = r_200/c
        rou_0 = ms*10**m[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rs[n]**3)
        for t in range(0,L):
            Rs[n,t] = R[t]*rs[n]
            if Rs[n,t]<rs[n]:
               g_x[n,t] = 8*np.arctanh(np.sqrt((1-Rs[n,t]/rs[n])/(1+Rs[n,t]/rs[n])))/(\
               (Rs[n,t]/rs[n])**2*np.sqrt(1-(Rs[n,t]/rs[n])**2))\
               +4*np.log((Rs[n,t]/rs[n])/2)/(Rs[n,t]/rs[n])**2\
               -2/((Rs[n,t]/rs[n])**2-1)\
               +4*np.arctanh(np.sqrt((1-Rs[n,t]/rs[n])/(1+Rs[n,t]/rs[n])))/((\
               (Rs[n,t]/rs[n])**2-1)*np.sqrt(1-(Rs[n,t]/rs[n])**2))
               delta_segma = rs[n]*rou_0*g_x[n,t]
               deltasegma[n,t] = delta_segma/ms
            elif Rs[n,t]==rs[n]:
                 g_x[n,t] = 10/3+4*np.log(1/2)
                 delta_segma = rs[n]*rou_0*g_x[n,t]
                 deltasegma[n,t] = delta_segma/ms
            else:
                g_x[n,t] = 8*np.arctan(np.sqrt((Rs[n,t]/rs[n]-1)/(Rs[n,t]/rs[n]+1)))\
                /((Rs[n,t]/rs[n])**2*np.sqrt((Rs[n,t]/rs[n])**2-1))\
                +4*np.log((Rs[n,t]/rs[n])/2)/(Rs[n,t]/rs[n])**2\
                -2/((Rs[n,t]/rs[n])**2-1)\
                +4*np.arctan(np.sqrt((Rs[n,t]/rs[n]-1)/(Rs[n,t]/rs[n]+1)))/(\
                (Rs[n,t]/rs[n])**2-1)**(3/2)
                delta_segma = rs[n]*rou_0*g_x[n,t]
                deltasegma[n,t] = delta_segma/ms            
    plt.figure()
    for k in range(0,LL):
        x = Rs[k,:]/rs[k]
        y = deltasegma[k,:]
        plt.loglog(x,y,'-')
        plt.grid()
        plt.xlabel(r"$lg(R/rs)$")
        plt.ylabel(r"$lgDeltasigma[Msun*Mpc^-2]$")
    plt.hold()
    plt.show()

    plt.figure()
    plt.loglog(x,y)
    plt.xlabel(r"$lg(R/rs)$")
    plt.ylabel(r"$lgDeltasigma[Msun*Mpc^-2]$")
semula_tion(omegam=True,h=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''