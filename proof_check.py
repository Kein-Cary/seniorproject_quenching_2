import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
from read_m16 import test_read_m16_ds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t as st
import seaborn as sns
#画误差棒函数须引入scipy库
#导入的质量区间为11~15dex-Msolar
def input_mass_bin(v):
    N = 48
    mr = np.linspace(13.3,13.4,N)
    mb = np.linspace(12.6,13.0,N)
    #m = np.linspace(12,13.5,N)
    return mr,mb,N
    #return m,N
#input_mass_bin(v=True)
def calcu_sigmaz(Rpc,m_,x,z):
    global omega_m
    omega_m = 0.315
    omegam = omega_m
#a flat CDM model,omegak=0;omegalamd=0
    global h_
    h_ = 0.7
    h = h_
    global G_
    G_ = 6.67*10**(-11)
    G = G_
    #global ms_
    #ms_ = 1.989*10**(30)
    #ms = ms_  
    global c_
    c_ = 6
    ##调试结果c=3.25~3.75之间比较合适
    c = c_
    #下面开始计算
    #对导入的数据做单位转化转为:太阳质量/h
    m1 = h*m_*10**x
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
    #Qc = 3.084*10**(-2)/(1.989)#对rouc的单位修正（此时对roum、rou0的也完成了修正）
    #rhom = 52700242579.0
    #m1 = m_*10**x
    Q200 = 1#对r200的单位修正（此时对rs也做了修正）
    H = h*100
    ##修正可以选择两处，对于计算过程涉及的物理量修正，或者对于常数修正
    rhoc = (Qc*(3*H**2)/(8*np.pi*G))/(1/(1+z))**3
    rhom = 200*rhoc*omegam
    #rhom=52700242579.0*200/h**2
    #考虑所有的红移为0.1
    r_200 = Q200*(3*m1/(4*np.pi*rhom))**(1/3)
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
    return(Rpc,Sigma,deltaSigma,rs)
#calcu_sigmaz(Rpc=True,m_=True,x=True,z=True)
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
    plt.plot(Rpc[:],ds_sim[k,:],'r--')
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
#fig_fff1(f=True)
def fig_fff2(Rpc,ds_sim,k,t):
    test_read_m16_ds()
    plt.plot(Rpc[:],ds_sim[k,t,:],'-',)
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
#fig_fff2(f=True)
def fit_datar(y):
#def fie_datar(mass_bin):
    h = 0.7
    #先找出观测值对应的rp
    rsa,dssa,ds_errsa = test_read_m16_ds()
    #rsa,dssa,ds_errsa = test_read_m16_ds(mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    a = np.shape(dssa)
    print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    z_r = 0.1
    #z_r = 0.105
    a_r = 1/(1+z_r)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=5])
   # rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
   # rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    print(rp)
    b = len(rp)
    mr,mb,N = input_mass_bin(v=True)
    #m,N = input_mass_bin(v=True)
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simr = np.zeros((N,b),dtype=np.float)
    #rr = np.zeros(N,dtype=np.float)
    for t in range(0,N):
        m_ = 1
        x = mr[t]
        #x = m[t]
        #Rpc = rp
        #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)    
        #ds_simr[k,t,:] = deltaSigma
        #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
        #计算模型在对应的投射距离上的预测信号取值
        Rpc = rp
        Rpc,Sigma,deltaSigma,rs = calcu_sigmaz(Rpc,m_,x,z_r)
        #对预测信号做相应的变化，把共动的转为物理的
        ds_simr[t,:] = deltaSigma
        #rr[t] = rs
        #fig_fff1(rp,ds_simr,t)
    #plt.title('Red')
    #plt.legend(m[:],bbox_to_anchor=(1,1))
    #plt.legend(m[:])   
    #plt.savefig('Fit-red2.png',dpi=600)
    #plt.savefig('compare_1_rd.png',dpi=600)
    #plt.show()
    yy = np.shape(ds_simr)
    #print('rr=',rr)#输出查看ds_sim的维度，即模型预测下的透镜信号    
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    delta_r = np.zeros(N,dtype=np.float)
    for t in range(0,N):
        d_r = 0
        for n in range(0,yy[1]):
            d_r = d_r+((ds_simr[t,n]-dssa[0,n])/ds_errsa[0,n])**2
        delta_r[t] = d_r
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    deltar = delta_r.tolist()
    xr = deltar.index(min(deltar))
    bestfmr = mr[xr]
    #bestfmr = m[xr]
    best_mr = bestfmr+np.log10(h)
    #作图对比最佳情况
    '''
    plt.figure()
    k = xr
    fig_fff1(rp,ds_simr,k)
    plt.title('Red')
    #plt.savefig('fit_r.png',dpi=600)
    plt.show()
    '''
    print('x^2=',min(deltar))
    print('co_mr=',best_mr)
    print('mr=',bestfmr)
    print('positionr=',xr)
    return rp,best_mr,delta_r
#fit_datar(y=True)
def fit_datab(y):
#def fit_datab(mass_bin):
    h = 0.7
    #先找出观测值对应的rp
    rsa,dssa,ds_errsa = test_read_m16_ds()
    #rsa,dssa,ds_errsa = test_read_m16_ds(mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    a = np.shape(dssa)
    print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    z_b = 0.1
    #z_b = 0.124
    a_b = 1/(1+z_b)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=5]) 
    # rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
    # rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    print(rp)
    b = len(rp)
    mr,mb,N = input_mass_bin(v=True)
    #m,N = input_mass_bin(v=True)
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simb = np.zeros((N,b),dtype=np.float)
    #rb = np.zeros(N,dtype=np.float)
    for t in range(0,N):
        m_ = 1
        x = mb[t]
        #x = m[t]
        #Rpc = rp
        #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)    
        #ds_simr[k,t,:] = deltaSigma
        #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
        #计算模型在对应的投射距离上的预测信号取值
        Rpc = rp
        Rpc,Sigma,deltaSigma,rs = calcu_sigmaz(Rpc,m_,x,z_b)
        #对预测信号做相应的变化，把共动的转为物理的
        ds_simb[t,:] = deltaSigma
        #rb[t] = rs
        #fig_fff1(rp,ds_simb,t)
    #plt.title('Blue')
    #plt.legend(m[:],bbox_to_anchor=(1,1))   
    #plt.legend(m[:])   
    #plt.savefig('Fit-blue2.png',dpi=600)
    #plt.savefig('compare_2_bl.png',dpi=600)
    #plt.show()
    yy = np.shape(ds_simb)
    #print('rr=',rr)#输出查看ds_sim的维度，即模型预测下的透镜信号    
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    delta_b = np.zeros(N,dtype=np.float)
    for t in range(0,N):
        d_b = 0
        for n in range(0,yy[1]):
            d_b = d_b+((ds_simb[t,n]-dssa[1,n])/ds_errsa[1,n])**2
        delta_b[t] = d_b
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    deltab = delta_b.tolist()
    xb = deltab.index(min(deltab))
    bestfmb = mb[xb]
    #bestfmb = m[xb]
    best_mb = bestfmb+np.log10(h)
    #作图对比最佳情况
    '''
    plt.figure()
    k = xb
    fig_fff1(rp,ds_simb,k)
    plt.title('Blue')
    #plt.savefig('fit_r.png',dpi=600)
    plt.show()
    '''
    print('x^2=',min(deltab))
    print('co_mb=',best_mb)
    print('mb=',bestfmb)
    print('positionr=',xb)
    return rp,best_mb,delta_b
#fit_datab(y=True)
#下面做图比较最佳预测值与观测的对比情况
def fig_(T):
    mr,mb,N = input_mass_bin(v=True)
    rp,best_mr,delta_r = fit_datar(y=True)
    rp,best_mb,delta_b = fit_datab(y=True)
    #rp,best_mr,delta_r = fit_datar(mass_bin)
    #rp,best_mb,delta_b = fit_datab(mass_bin)
    plt.subplot(121)
    plt.plot(mr,np.log10(delta_r),'r')
    plt.title('R-galaxy')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.grid()
    plt.subplot(122)
    plt.plot(mb,np.log10(delta_b),'b')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.title('B-galaxy')
    plt.grid()
    plt.tight_layout()
    plt.show()  
    print('mr=',best_mr)
    print('mb=',best_mb)
    ##下面把x^2转化为相应的概率分布，并让mh的划分区间服从这个分布
    rsa,dssa,ds_errsa = test_read_m16_ds()
    pr = np.ones(N,dtype=np.float)
    pb = np.ones(N,dtype=np.float)
    ds_errsar = ds_errsa[0,0:len(rp)]
    ds_errsab = ds_errsa[1,0:len(rp)]
    for k in range(0,N):
        ss=1
        qq=1
        for t in range(0,len(rp)):
            #ss = ss*(1/(np.sqrt(np.pi*2)*ds_errsar[t]))*np.exp((-1/2)*d_d_r[k])
            #qq = qq*(1/(np.sqrt(np.pi*2)*ds_errsab[t]))*np.exp((-1/2)*d_d_b[k])
            ss = ss*(1/(np.sqrt(np.pi*2)*ds_errsar[t]))*np.exp((-1/2)*delta_r[k])
            qq = qq*(1/(np.sqrt(np.pi*2)*ds_errsab[t]))*np.exp((-1/2)*delta_b[k])
        pr[k]=ss
        pb[k]=qq
    ##第一次归一化，把概率密度函数归一到0~1之间
    plt.figure()
    pr = (pr-np.min(pr))/(np.max(pr)-np.min(pr))
    pb = (pb-np.min(pb))/(np.max(pb)-np.min(pb))
    plt.plot(mr,pr,'r')
    plt.title('Probability distribution')
    plt.xlabel(r'$log(M_h/M_\odot)$')
    plt.ylabel(r'$dP(M_h|M_\ast)/dM_h$')
    plt.show()
    plt.plot(mb,pb,'b')
    plt.title('Probability distribution')
    plt.xlabel(r'$log(M_h/M_\odot)$')
    plt.ylabel(r'$dP(M_h|M_\ast)/dM_h$')
    plt.show()  
    #print('r=',delta_r)
    #print('b=',delta_b)
    ##第二次归一化，目的是让所有概率求和为1
    fr=np.zeros(N,dtype=np.float)
    fb=np.zeros(N,dtype=np.float)
    ss=0
    qq=0
    Fr=np.zeros(N,dtype=np.float)
    Fb=np.zeros(N,dtype=np.float)
    for k in range(0,N):
        ss=ss+pr[k]*(mr[1]-mr[0])
        fr[k]=ss
    for k in range(0,N):
        qq=qq+pb[k]*(mb[1]-mb[0])
        fb[k]=qq
    Ar = np.max(fr)
    Ab = np.max(fb)
    ss=0
    qq=0
    for k in range(0,N):
        ss=ss+pr[k]*(mr[1]-mr[0])/Ar
        Fr[k]=ss
    for k in range(0,N):
        qq=qq+pb[k]*(mb[1]-mb[0])/Ab
        Fb[k]=qq   
    plt.figure()
    plt.plot(mr,Fr,'r')
    plt.show()
    plt.plot(mb,Fb,'b')
    plt.show()
    vr = np.array([mr[k] for k in range(0,len(mr)) if Fr[k]>=0.16 and Fr[k]<=0.84] )
    vb = np.array([mb[k] for k in range(0,len(mb)) if Fb[k]>=0.16 and Fb[k]<=0.84] )
    ur = [vr[0],vr[-1]]
    ub = [vb[0],vb[-1]]
    print(ur)
    print(ub)
    ##直接把x^2规范化，求errorbar的大小
    D_chi_r = delta_r-np.min(delta_r)
    D_chi_b = delta_b-np.min(delta_b)
    plt.plot(mr,D_chi_r,'r')
    plt.grid()
    plt.show()
    plt.plot(mb,D_chi_b,'b')
    plt.grid()
    plt.show()
    sigma_mr = np.array([mr[k] for k in range(0,len(mr)) if abs(D_chi_r[k]-1)<=0.15])
    sigma_mb = np.array([mb[k] for k in range(0,len(mb)) if abs(D_chi_b[k]-1)<=0.1])
    print('sigma_r=',sigma_mr)
    print('sigma_b=',sigma_mb)
    return mr,mb,N,best_mr,best_mb
fig_(T=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''