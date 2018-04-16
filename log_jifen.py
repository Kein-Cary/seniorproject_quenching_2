import sys
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
from Desigma_checking import  dolad_data
from Desigma_checking import  constant_f
from _colossus_ import profile_check
#在上述路径下导入的两个自己编辑的脚本，并且是导入脚本里面的某个模块
#导入文件中已包含单位变换
import numpy as np
import matplotlib.pyplot as plt
def rou_R(R):
    m,hz = dolad_data(m=True,hz=True)
    print('input mh=',m)
    Q200,Qc,c,G,h,omegam,hz,m_=constant_f(t1=True,t2=True)
    m = h*10**m_
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
        rouc = (Qc*(3*H**2)/(8*np.pi*G))
        #上句表示添加红移后的修正（z=3.5）
        #rouc = Qc*(3*H**2)/(8*np.pi*G)
        roum = 200*rouc*omegam
        r200[k] = Q200*(3*m[k]*omegam/(4*np.pi*roum))**(1/3)
        rs[k] = r200[k]/c
        rou0[k] = m[k]/((np.log(1+c)-c/(1+c))*4*np.pi*rs[k]**3)
        for t in range(0,lr):
            r[k,t] = R_bin[t]*rs[k]
            rouR[k,t] = rou0[k]*rs[k]**3/(r[k,t]*(rs[k]+r[k,t])**2)
        integm[k] = (4*np.pi*rou0[k]*rs[k]**3*(np.log(1+\
               r200[k]/rs[k])-r200[k]/(rs[k]+r200[k])))/h
    #integm表示直接从积分的解析函数出发，检查积分质量是多少，理论上该值应严格等于输入质量
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
    plt.ylabel(r'$\rho(R)-M_\odot-h^2/{Mpc^3}$')
    profile_check(c=True)
    plt.grid()
    plt.savefig('25.png',dpi=600)
    plt.show()
    print('mh=',integm)
    return(rouR,r,rs,R,lm,lr,r200,c)#需要调用几个就返回几个数组的值
#rou_R(R=True)    
def rou2(x):
    a=-6
    b=2
    h=0.673
    M=1e4
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
            #直接的对数空间积分变换计算，因为存在计算误差，直接积分的质量并不能严格等于输入
            #但是两者的误差要越小越好
        max_m[k] = np.max(int_m[k,:])#求出每一次积分的最后结果，作为检测质量，和输入比较
    plt.figure()
    #下面两个变量是为了做垂直标度线取得狄拉克函数
    delta1 = np.zeros(lm,dtype=np.float)
    delta2 = np.zeros(lm,dtype=np.float)
    for n in range(0,lm):
        plt.plot(x[:],np.log10(int_m[n,:]/h))
        delta1[k] = np.log10(r200[n])
        delta2[k] = np.log10(rs[n])
        plt.axvline(delta1[k],ls='--',linewidth=0.5,color='red')
        plt.axvline(delta2[k],ls='--',linewidth=0.5,color='black')
        #axvline表示快速做图函数命令，做水平线的见脚本Desigma_comoving.(一开始的数据导入测试)
    plt.grid()
    plt.xlabel(r'$log10(r/rs)$')
    plt.ylabel(r'$log10(\frac{mh}{m_\odot})$')
    plt.show()
    print('mh=',max_m/h)    
    return(max_m)
rou2(x=True)
##待补充：尝试用函数调用实现上诉过程