import os.path
import numpy as np
import matplotlib.pyplot as plt
def dolad_data(m,hz):
#mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/data_mh.txt/'
    _mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/'
    fname = os.path.join(_mh_path,'data_m1.txt')
    m = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)    
    mr = m[0,:]
    mb = m[1,:]
    print('mr=',mr)
    print('mb=',mb)
    print('m=',m)
    fname = os.path.join(_mh_path,'data_z.txt')
    hz = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)
    print('hz=',hz)
    plt.plot(m[:,1],m[:,2],label='Doload-successfully')
    plt.legend()
    plt.show()
    return(m,hz)
dolad_data(m=True,hz=True)

def semula_tion(omegam,h):
    m,hz = dolad_data(m=True,hz=True)
    mr = m[0,:]
    mb = m[1,:]
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
    LL = len(mr)
    R = np.linspace(0,100,1000)
    L = len(R)
    Rsr = np.zeros((LL,L),dtype=np.float)
    Rsb = np.zeros((LL,L),dtype=np.float)
    g_xr = np.zeros((LL,L),dtype=np.float)
    g_xb = np.zeros((LL,L),dtype=np.float)
    deltasegmar = np.zeros((LL,L),dtype=np.float)
    deltasegmab = np.zeros((LL,L),dtype=np.float)

    rsr =  np.zeros(LL,dtype=np.float)
    rsb =  np.zeros(LL,dtype=np.float)
    rou_0r = np.zeros(LL,dtype=np.float)
    rou_0b = np.zeros(LL,dtype=np.float)
    for n in range(0,LL):
        #E = np.sqrt(omegam*(1+hz[n]))
        H = h*100
        rouc = (3*H**2)/(8*np.pi*G*10**(-9))
        roum = 200*rouc*omegam
        
        r_200r = (3*10**mr[n]*ms/(4*np.pi*roum))**(1/3)
        r_200b = (3*10**mb[n]*ms/(4*np.pi*roum))**(1/3)
        
        rsr[n] = r_200r/c
        rsb[n] = r_200b/c
        
        rou_0r[n] = ms*10**mr[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rsr[n]**3)
        rou_0b[n] = ms*10**mb[n]/((np.log(1+c)-c/(1+c))*4*np.pi*rsb[n]**3)
        
        for t in range(0,L):
            Rsr[n,t] = R[t]*rsr[n]
            if Rsr[n,t]<rsr[n]:
               g_xr[n,t] = 8*np.arctanh(np.sqrt((1-Rsr[n,t]/rsr[n])/(1+Rsr[n,t]/rsr[n])))/(\
               (Rsr[n,t]/rsr[n])**2*np.sqrt(1-(Rsr[n,t]/rsr[n])**2))\
               +4*np.log((Rsr[n,t]/rsr[n])/2)/(Rsr[n,t]/rsr[n])**2\
               -2/((Rsr[n,t]/rsr[n])**2-1)\
               +4*np.arctanh(np.sqrt((1-Rsr[n,t]/rsr[n])/(1+Rsr[n,t]/rsr[n])))/((\
               (Rsr[n,t]/rsr[n])**2-1)*np.sqrt(1-(Rsr[n,t]/rsr[n])**2))
               delta_segmar = rsr[n]*rou_0r[n]*g_xr[n,t]
               deltasegmar[n,t] = delta_segmar/ms
            elif Rsr[n,t]==rsr[n]:
                 g_xr[n,t] = 10/3+4*np.log(1/2)
                 delta_segmar = rsr[n]*rou_0r[n]*g_xr[n,t]
                 deltasegmar[n,t] = delta_segmar/ms
            else:
                g_xr[n,t] = 8*np.arctan(np.sqrt((Rsr[n,t]/rsr[n]-1)/(Rsr[n,t]/rsr[n]+1)))\
                /((Rsr[n,t]/rsr[n])**2*np.sqrt((Rsr[n,t]/rsr[n])**2-1))\
                +4*np.log((Rsr[n,t]/rsr[n])/2)/(Rsr[n,t]/rsr[n])**2\
                -2/((Rsr[n,t]/rsr[n])**2-1)\
                +4*np.arctan(np.sqrt((Rsr[n,t]/rsr[n]-1)/(Rsr[n,t]/rsr[n]+1)))/(\
                (Rsr[n,t]/rsr[n])**2-1)**(3/2)
                delta_segmar = rsr[n]*rou_0r[n]*g_xr[n,t]
                deltasegmar[n,t] = delta_segmar/ms
        t=0;        
        for t in range(0,L):
            Rsb[n,t] = R[t]*rsb[n]
            if Rsb[n,t]<rsb[n]:
               g_xb[n,t] = 8*np.arctanh(np.sqrt((1-Rsb[n,t]/rsb[n])/(1+Rsb[n,t]/rsb[n])))/(\
               (Rsb[n,t]/rsb[n])**2*np.sqrt(1-(Rsb[n,t]/rsb[n])**2))\
               +4*np.log((Rsb[n,t]/rsb[n])/2)/(Rsb[n,t]/rsb[n])**2\
               -2/((Rsb[n,t]/rsb[n])**2-1)\
               +4*np.arctanh(np.sqrt((1-Rsb[n,t]/rsb[n])/(1+Rsb[n,t]/rsb[n])))/((\
               (Rsb[n,t]/rsb[n])**2-1)*np.sqrt(1-(Rsb[n,t]/rsb[n])**2))
               delta_segmab = rsb[n]*rou_0b[n]*g_xb[n,t]
               deltasegmab[n,t] = delta_segmab/ms
            elif Rsb[n,t]==rsb[n]:
                 g_xb[n,t] = 10/3+4*np.log(1/2)
                 delta_segmab = rsb[n]*rou_0b[n]*g_xb[n,t]
                 deltasegmab[n,t] = delta_segmab/ms
            else:
                g_xb[n,t] = 8*np.arctan(np.sqrt((Rsb[n,t]/rsb[n]-1)/(Rsb[n,t]/rsb[n]+1)))\
                /((Rsb[n,t]/rsb[n])**2*np.sqrt((Rsb[n,t]/rsb[n])**2-1))\
                +4*np.log((Rsb[n,t]/rsb[n])/2)/(Rsb[n,t]/rsb[n])**2\
                -2/((Rsb[n,t]/rsb[n])**2-1)\
                +4*np.arctan(np.sqrt((Rsb[n,t]/rsb[n]-1)/(Rsb[n,t]/rsb[n]+1)))/(\
                (Rsb[n,t]/rsb[n])**2-1)**(3/2)
                delta_segmab = rsb[n]*rou_0b[n]*g_xb[n,t]
                deltasegmab[n,t] = delta_segmab/ms         
                
    plt.figure()
    for k in range(0,LL):
        x1 = Rsr[k,:]/rsr[k]
        y1 = deltasegmar[k,:]
        plt.loglog(x1,y1,'-')
        x2 = Rsb[k,:]/rsb[k]
        y2 = deltasegmab[k,:]
        #plt.loglog(x2,y2,'-')
        plt.grid()
    plt.xlabel(r'$\frac{R}{rs}$')
    plt.ylabel(r'$\Delta\Sigma(\frac{R}{rs})-M_sMpc^{-2}$')
    plt.hold()
    plt.show()
    plt.figure()
    plt.loglog(x1,y1,'r*')
    plt.loglog(x2,y2,'b*')
    plt.grid()
    plt.xlabel(r'$\frac{R}{rs}$')
    plt.ylabel(r'$\Delta\Sigma(\frac{R}{rs})-M_sMpc^{-2}$')
  
semula_tion(omegam=True,h=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''