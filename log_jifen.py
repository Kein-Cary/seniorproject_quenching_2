import sys
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
from Desigma_checking import  dolad_data
from Desigma_checking import  constant_f
import numpy as np
import matplotlib.pyplot as plt
def rou_R(R):
    m,hz = dolad_data(m=True,hz=True)
    print('input mh=',m)
    Q200,Qc,c,G,h,omegam,hz,m_=constant_f(t1=True,t2=True)
    m = 0.7*10**m_
    #对导入的数据做单位转化转为:太阳质量/h
    lm = len(m)
    # N = 10**5
    # R = 0
    # R_bin = np.linspace(R,1000,N)
    R_bin = R
    lr = len(R_bin)
    rs = np.zeros(lm,dtype=np.float)
    r = np.zeros((lm,lr),dtype=np.float)
    r200 =  np.zeros(lm,dtype=np.float)
    rou0 = np.zeros(lm,dtype=np.float)
    integm = np.zeros(lm,dtype=np.float)
    rouR = np.zeros((lm,lr),dtype=np.float)
    H = 100*h
    for k in range(0,lm):
        rouc = Qc*(3*H**2)/(8*np.pi*G)
        roum = 200*rouc*omegam
        r200[k] = Q200*(3*m[k]/(4*np.pi*roum))**(1/3)
        rs[k] = r200[k]/c
        rou0[k] = m[k]/((np.log(1+c)-c/(1+c))*4*np.pi*rs[k]**3)
        for t in range(0,lr):
            r[k,t] = R_bin[t]*rs[k]
            rouR[k,t] = rou0[k]*rs[k]**3/(r[k,t]*(rs[k]+r[k,t])**2)
        integm[k] = (4*np.pi*rou0[k]*rs[k]**3*(np.log(1+\
               r200[k]/rs[k])-r200[k]/(rs[k]+r200[k])))/0.7
    plt.figure()
    delta1 = np.zeros(lm,dtype=np.float)
    delta2 = np.zeros(lm,dtype=np.float)
    for k in range(0,lm):
        plt.loglog(r[k,:],rouR[k])
        delta1[k] = r200[k]
        delta2[k] = rs[k]
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
    plt.xlabel(r'$r-Mpc/h$')
    plt.ylabel(r'$\rho(R)-M_\odot-h/{Mpc^3}$')
    plt.grid()
    plt.show()
    print('mh=',integm)
    return(rouR,r,rs,R,lm,lr,r200,c)#需要调用几个就返回几个数组的值
#rou_R(R=True)    
def rou2(x):
    a=-6
    b=2
    M=10**5
    x=np.linspace(a,b,M)
    lR=10**x
    rouR,r,rs,R,lm,lr,r200,c = rou_R(lR)
    y = rouR
    int_m = np.zeros((lm,lr),dtype=np.float)
    max_m = np.zeros(lm,dtype=np.float)
    for k in range(0,lm):
        dx = (b-a)/M
        #int_m[k,0] = 4*np.pi*np.log(10)*lR[0]**3*dx*y[k,1]
        for t in range(0,lr):
            if x[t]<=np.log10(r200[k]/rs[k]) and t>0:
               int_m[k,t] = int_m[k,t-1]+\
               rs[k]**3*4.0*np.pi*np.log(10)*lR[t]**3.0*dx*(y[k,t]+y[k,t-1])/2.0
        max_m[k] = np.max(int_m[k,:])
    plt.figure()
    delta1 = np.zeros(lm,dtype=np.float)
    delta2 = np.zeros(lm,dtype=np.float)
    for n in range(0,lm):
        plt.plot(x[:],np.log10(int_m[n,:]/0.7))
        delta1[k] = np.log10(r200[n])
        delta2[k] = np.log10(rs[n])
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
    plt.grid()
    plt.xlabel(r'$log10(r/rs)$')
    plt.ylabel(r'$log10(\frac{mh}{m_\odot})$')
    plt.show()
    print('mh=',max_m/0.7)    
    return(max_m)
rou2(x=True)
