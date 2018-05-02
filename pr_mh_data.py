import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
from read_m16_1 import test_read_m16_ds_1
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.stats import t as st
import seaborn as sns
#from scipy.stats import chisquare as chi
#画误差棒函数须引入scipy库
#导入的质量区间为11~15dex-Msolar
def input_mass_bin(v):
    N = 4
    #m = np.linspace(11,15,21)
    m = np.array([np.linspace(11,12.75,101),np.linspace(11,12.75,101),\
                  np.linspace(11.75,13,201),np.linspace(12.25,13.75,201)])
    #针对不同质量区间设置拟合质量范围
    return m,N
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
def fig_0(Rpc,ds_sim,lmw,r,mass_bin):
    test_read_m16_ds_1(mass_bin=mass_bin)
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
#fig_0(Rpc=True,ds_sim=True,lmw=True,r=True)
#fig_0(f=True)
##定义一个质量最佳预测和观测的对比图象
def fig_0f1(Rpc,ds_sim,k,mass_bin):
    test_read_m16_ds_1(mass_bin=mass_bin)
    plt.plot(Rpc[:],ds_sim[k,:],'--')
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
#fig_0f1(f=True)
def fig_0f2(Rpc,ds_sim,k,t,mass_bin):
    test_read_m16_ds_1(mass_bin=mass_bin)
    plt.plot(Rpc[:],ds_sim[k,t,:],'--',)
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
#fig_0f2(f=True)
#def fit_datar(y):
def fit_datar_1(mass_bin,T):
    h = 0.7
    #先找出观测值对应的rp
    #rsa,dssa,ds_errsa = test_read_m16_ds_1()
    rsa,dssa,ds_errsa = test_read_m16_ds_1(mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    a = np.shape(dssa)
    #print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    z_r = 0.1
    #z_r = 0.105
    a_r = 1/(1+z_r)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=5])
    #rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    b = len(rp)
    m,N = input_mass_bin(v=True)
    mr = m[T]
    #print('mr=',mr)
    lm_bin = len(mr)   
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simr = np.zeros((lm_bin,b),dtype=np.float)
    #rr = np.zeros(lm_bin,dtype=np.float)
    for t in range(0,lm_bin):
        m_ = 1
        x = mr[t]
        #Rpc = rp
        #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)    
        #ds_simr[k,t,:] = deltaSigma
        #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
        #计算模型在对应的投射距离上的预测信号取值
        Rpc = rp
        Rpc,Sigma,deltaSigma,rs = calcu_sigmaz(Rpc,m_,x,z_r)
        #对预测信号做相应的变化，把共动的转为物理的
        ds_simr[t,:] = deltaSigma
        #下面部分作图对比
        #rr[t] = rs
        #fig_0f1(rp,ds_simr,t,mass_bin)
    #plt.title('Red')
    #plt.legend(m[:],bbox_to_anchor=(1,1))
    #plt.legend(m[:])   
    #plt.savefig('Fit-red2.png',dpi=600)
    #plt.savefig('compare_1_rd.png',dpi=600)
    #plt.show()
    yy = np.shape(ds_simr)
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    delta_r = np.zeros(lm_bin,dtype=np.float)
    for t in range(0,lm_bin):
        d_r = 0
        for n in range(0,yy[1]):
            d_r = d_r+((ds_simr[t,n]-dssa[0,n])/ds_errsa[0,n])**2
        delta_r[t] = d_r
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    deltar = delta_r.tolist()
    xr = deltar.index(min(deltar))
    bestfmr = mr[xr]
    best_mr = bestfmr+np.log10(h)
    #作图对比最佳情况
    '''
    plt.figure()
    k = xr
    fig_0f1(rp,ds_simr,k,mass_bin)
    plt.title('Red')
    #plt.savefig('fit_r.png',dpi=600)
    plt.show()
    '''
    #print('x^2=',min(deltar))
    #print('mr=',best_mr)
    #print('positionr=',xr)
    return rp,best_mr,delta_r,bestfmr
#fit_datar_1(y=True)
#def fit_datab_1(y):
def fit_datab_1(mass_bin,T):
    h = 0.7
    #先找出观测值对应的rp
    #rsa,dssa,ds_errsa = test_read_m16_ds_1()
    rsa,dssa,ds_errsa = test_read_m16_ds_1(mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    a = np.shape(dssa)
    #print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    z_b = 0.1
    #z_b = 0.124
    a_b = 1/(1+z_b)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=5]) 
    # rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
    # rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    b = len(rp)
    m,N = input_mass_bin(v=True)
    mb = m[T]
    #print('mb=',mb)
    lm_bin = len(mb) 
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simb = np.zeros((lm_bin,b),dtype=np.float)
    #rb = np.zeros(lm_bin,dtype=np.float)
    for t in range(0,lm_bin):
        m_ = 1
        x = mb[t]
        #Rpc = rp
        #Rpc,Sigma,deltaSigma = calcu_sigma(Rpc,m_,x)    
        #ds_simr[k,t,:] = deltaSigma
        #对比文献，加入尺度因子修正如下,把物理的距离转为共动的距离
        #计算模型在对应的投射距离上的预测信号取值
        Rpc = rp
        Rpc,Sigma,deltaSigma,rs = calcu_sigmaz(Rpc,m_,x,z_b)
        #对预测信号做相应的变化，把共动的转为物理的
        ds_simb[t,:] = deltaSigma
        ##下面部分作图对比
        #rb[t] = rs
        #fig_0f1(rp,ds_simb,t,mass_bin)
    #plt.title('Blue')
    #plt.legend(m[:],bbox_to_anchor=(1,1))   
    #plt.legend(m[:])   
    #plt.savefig('Fit-blue2.png',dpi=600)
    #plt.savefig('compare_2_bl.png',dpi=600)
    #plt.show()
    yy = np.shape(ds_simb)
    #比较观测的sigma和预测的Sigma,比较结果用fitr和fitb记录比较结果
    delta_b = np.zeros(lm_bin,dtype=np.float)
    for t in range(0,lm_bin):
        d_b = 0
        for n in range(0,yy[1]):
            d_b = d_b+((ds_simb[t,n]-dssa[1,n])/ds_errsa[1,n])**2
        delta_b[t] = d_b
    #print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    deltab = delta_b.tolist()
    xb = deltab.index(min(deltab))
    bestfmb = mb[xb]
    best_mb = bestfmb+np.log10(h)
    #作图对比最佳情况
    '''
    plt.figure()
    k = xb
    fig_0f1(rp,ds_simb,k,mass_bin)
    plt.title('Blue')
    #plt.savefig('fit_r.png',dpi=600)
    plt.show()
    '''
    #print('x^2=',min(deltab))
    #print('mb=',best_mb)
    #print('positionr=',xb)
    return rp,best_mb,delta_b,bestfmb
#fit_datab_1(y=True)
#下面做图比较最佳预测值与观测的对比情况
def fig_1(tt):
    m,N = input_mass_bin(v=True)
    rp,best_mr,delta_r,bestfmr = fit_datar_1(y=True)
    rp,best_mb,delta_b,bestfmb = fit_datab_1(y=True)
    #rp,best_mr,delta_r = fit_datar(mass_bin)
    #rp,best_mb,delta_b = fit_datab(mass_bin)
    plt.subplot(121)
    plt.plot(m,np.log10(delta_r),'r')
    plt.title('R-galaxy')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.grid()
    plt.subplot(122)
    plt.plot(m,np.log10(delta_b),'b')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.title('B-galaxy')
    plt.grid()
    plt.tight_layout()
    plt.show()  
    print('mr=',best_mr)
    print('mb=',best_mb)
    return m,N,best_mr,best_mb
#fig_(tt=True)
def fig_mass(pp):
    m_bin = ['10.0_10.4','10.4_10.7','10.7_11.0','11.0_15.0']
    mbin = [10.2,10.55,10.9,13.0]
    #质量区间
    x_m = np.logspace(1,1.18,4)
    ##画图质量区间
    m,N = input_mass_bin(v=True)
    #print(m)
    #该句导入拟合质量
    Pr_mh = np.zeros((len(m_bin),2),dtype=np.float)
    #保存以Msolar/h为单位的最佳质量
    b_m_r = np.zeros(len(m_bin),dtype=np.float)
    b_m_b = np.zeros(len(m_bin),dtype=np.float)
    #分别保存以Msolar为单位的最佳拟合质量
    x_std = np.zeros((len(m_bin),2),dtype=np.float)    
    #Pr_mh的第一行存red的质量，第二行存blue的质量,x_std也一样
    ##计算误差棒并存储的数组
    err_bar_r = np.zeros((len(m_bin),2),dtype=np.float)
    err_bar_b = np.zeros((len(m_bin),2),dtype=np.float)
    ##为了X^2检测，需要记录并提取每一个m*区间的概率分布（Ps:不是概率密度分布）
    Dp_r = np.array([np.zeros(len(m[1]),dtype=np.float),np.zeros(len(m[2]),dtype=np.float),\
                  np.zeros(len(m[2]),dtype=np.float),np.zeros(len(m[3]),dtype=np.float)])
    Dp_b = np.array([np.zeros(len(m[1]),dtype=np.float),np.zeros(len(m[2]),dtype=np.float),\
                  np.zeros(len(m[2]),dtype=np.float),np.zeros(len(m[3]),dtype=np.float)])
    ##记录最小的x^2的值，用于x^2检测
    c_xs_r = np.zeros(len(m_bin),dtype=np.float)
    c_xs_b = np.zeros(len(m_bin),dtype=np.float)
    ##尝试直接从chi-square出发，求出一倍sigma的误差棒函数
    D_chi_r = np.array([np.zeros(len(m[1]),dtype=np.float),np.zeros(len(m[2]),dtype=np.float),\
                  np.zeros(len(m[2]),dtype=np.float),np.zeros(len(m[3]),dtype=np.float)])
    D_chi_b = np.array([np.zeros(len(m[1]),dtype=np.float),np.zeros(len(m[2]),dtype=np.float),\
                  np.zeros(len(m[2]),dtype=np.float),np.zeros(len(m[3]),dtype=np.float)])  
    for k in range(0,len(m_bin)):
        rp,best_mr,delta_r,bestfmr = fit_datar_1(m_bin[k],k)
        rp,best_mb,delta_b,bestfmb = fit_datab_1(m_bin[k],k)
        Pr_mh[k,0] = best_mr
        Pr_mh[k,1] = best_mb
        b_m_r[k] = bestfmr
        b_m_b[k] = bestfmb
        c_xs_r[k] = np.min(delta_r)
        c_xs_b[k] = np.min(delta_b)
        mr = m[k]
        mb = m[k]
        ##不考虑插值时直接带入mr,mb计算，求出误差棒
        ##考虑插值时，先对最佳拟合质量附近考虑插值,后续代码相应符号(mr_,mb_,or chir,chib)表示以插值部分量做计算
        mr_ = np.linspace(b_m_r[k]-0.2,b_m_r[k]+0.2,len(m[k]))
        mb_ = np.linspace(b_m_b[k]-0.2,b_m_b[k]+0.2,len(m[k]))
        chir = np.interp(mr_,mr,delta_r)
        chib = np.interp(mb_,mb,delta_b)
        ##chir,chib表示即将带入计算概率分布的X^2.
        ##下面几句直接从x^2求解1倍sigma对应的mh,为此需要先做如下处理，1-sigma的取值见后面
        dchir = delta_r-np.min(delta_r)
        dchib = delta_b-np.min(delta_b)
        D_chi_r[k] = dchir
        D_chi_b[k] = dchib      
        ##下面把x^2转化为相应的概率分布，并让mh的划分区间服从这个分布
        rsa,dssa,ds_errsa = test_read_m16_ds_1(mass_bin=m_bin[k])
        pr = np.ones(len(m[k]),dtype=np.float)
        pb = np.ones(len(m[k]),dtype=np.float)
        pr_ = np.ones(len(m[k]),dtype=np.float)
        pb_ = np.ones(len(m[k]),dtype=np.float)
        ds_errsar = ds_errsa[0,0:len(rp)]
        ds_errsab = ds_errsa[1,0:len(rp)]
        ##假设x^2的数据分布服从高斯分布
        for q in range(0,len(mr)):
        ##未考虑插值的处理
            ss=1
            for t in range(0,len(rp)):
                ss = ss*(1/(np.sqrt(np.pi*2)*ds_errsar[t]))*np.exp((-1/2)*delta_r[q])
            pr[q]=ss
        for q in range(0,len(mr_)):
        ##考虑插值的计算处理
            ss=1
            for t in range(0,len(rp)):
                ss = ss*(1/(np.sqrt(np.pi*2)*ds_errsar[t]))*np.exp((-1/2)*chir[q])
            pr_[q]=ss
        for q in range(0,len(mb)):
        ##未考虑插值的处理
            qq=1
            for t in range(0,len(rp)):
                qq = qq*(1/(np.sqrt(np.pi*2)*ds_errsab[t]))*np.exp((-1/2)*delta_b[q])
            pb[q]=qq
        for q in range(0,len(mb_)):
        ##考虑插值的计算处理
            qq=1
            for t in range(0,len(rp)):
                qq = qq*(1/(np.sqrt(np.pi*2)*ds_errsab[t]))*np.exp((-1/2)*chib[q])
            pb_[q]=qq
        ##第一次归一化，把概率密度函数归一到0~1之间
        pr = (pr-np.min(pr))/(np.max(pr)-np.min(pr))
        pb = (pb-np.min(pb))/(np.max(pb)-np.min(pb))
        pr_ = (pr_-np.min(pr_))/(np.max(pr_)-np.min(pr_))
        pb_ = (pb_-np.min(pb_))/(np.max(pb_)-np.min(pb_))
        '''
        plt.figure()
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
        plt.figure()
        plt.plot(mr_,pr_,'r')
        plt.title('Probability distribution')
        plt.xlabel(r'$log(M_h/M_\odot)$')
        plt.ylabel(r'$dP(M_h|M_\ast)/dM_h$')
        plt.show()
        plt.plot(mb_,pb_,'b')
        plt.title('Probability distribution')
        plt.xlabel(r'$log(M_h/M_\odot)$')
        plt.ylabel(r'$dP(M_h|M_\ast)/dM_h$')
        plt.show()
        '''
        #print('r=',delta_r)
        #print('b=',delta_b)
        ##第二次归一化，目的是让所有概率求和为1
        ##未考虑插值的处理
        fr=np.zeros(len(mr),dtype=np.float)
        fb=np.zeros(len(mb),dtype=np.float)
        ##考虑插值的处理
        fr_=np.zeros(len(mr_),dtype=np.float)
        fb_=np.zeros(len(mb_),dtype=np.float)
        ss=0
        qq=0
        ##未考虑插值的处理
        Fr=np.zeros(len(mr),dtype=np.float)
        Fb=np.zeros(len(mb),dtype=np.float)
        ##考虑插值的处理
        Fr_=np.zeros(len(mr_),dtype=np.float)
        Fb_=np.zeros(len(mb_),dtype=np.float)       
        ##在不考虑插值时调用该部分
        for q in range(0,len(mr)):
            ss=ss+pr[q]*(mr[1]-mr[0])
            fr[q]=ss
        ##考虑插值的处理
        for q in range(0,len(mr_)):
            ss=ss+pr[q]*(mr_[1]-mr_[0])
            fr_[q]=ss
        ##在不考虑插值时调用该部分
        for q in range(0,len(mb)):
            qq=qq+pb[q]*(mb[1]-mb[0])
            fb[q]=qq
        ##考虑插值的处理
        for q in range(0,len(mb_)):
            qq=qq+pb[q]*(mb_[1]-mb_[0])
            fb_[q]=qq
        Ar = np.max(fr)
        Ab = np.max(fb)
        Ar_ = np.max(fr)
        Ab_ = np.max(fb)
        ss=0
        qq=0
        ##在不考虑插值时调用该部分
        for q in range(0,len(mr)):
            ss=ss+pr[q]*(mr[1]-mr[0])/Ar
            Fr[q]=ss
        ##考虑插值时调用该部分
        for q in range(0,len(mr_)):
            ss=ss+pr[q]*(mr_[1]-mr_[0])/Ar
            Fr_[q]=ss
        ##在不考虑插值时调用该部分
        for q in range(0,len(mb)):
            qq=qq+pb[q]*(mb[1]-mb[0])/Ab
            Fb[q]=qq 
        ##考虑插值时调用该部分
        for q in range(0,len(mb_)):
            qq=qq+pb[q]*(mb_[1]-mb_[0])/Ab
            Fb_[q]=qq
        #plt.figure()
        #plt.plot(mr,Fr,'r')
        #plt.show()
        #plt.plot(mb,Fb,'b')
        #plt.show()
        #plt.figure()
        #plt.plot(mr_,Fr_,'r')
        #plt.show()
        #plt.plot(mb_,Fb_,'b')
        #plt.show()
        vr = np.interp(0.16,Fr,mr)
        ur = np.interp(0.84,Fr,mr)
        err_bar_r[k,:] = [vr-bestfmr,ur-bestfmr]
        vb = np.interp(0.16,Fb,mb)
        ub = np.interp(0.84,Fb,mb)
        err_bar_b[k,:] = [vb-bestfmb,ub-bestfmb]
        ##记录每一次概率分布
        Dp_r[k] = pr
        Dp_b[k] = pb
    #print('phy_mr=',Pr_mh[:,0])
    #print('phy_mb=',Pr_mh[:,1])
    print('err_mr=',err_bar_r)
    print('err_mb=',err_bar_b) 
    #scatter函数参数设置
    color_list = ['r','b','g']
    bar_list = ['s','^','v']
    ##直接从概率分布得到误差棒
    plt.figure()
    line1,caps1,bars1=plt.errorbar(mbin,Pr_mh[:,0],yerr=[err_bar_r[:,0],err_bar_r[:,1]],fmt="rs-",linewidth=1,
                                elinewidth=0.5,ecolor='k',capsize=1,capthick=0.5,label='red')
    line3,caps3,bars3=plt.errorbar(mbin,Pr_mh[:,1],yerr=[err_bar_b[:,0],err_bar_b[:,1]],fmt="b^-",linewidth=1,
                                elinewidth=0.5,ecolor='k',capsize=1,capthick=0.5,label='blue')
    plt.fill_between(mbin,Pr_mh[:,0]+err_bar_r[:,0],Pr_mh[:,0]+err_bar_r[:,1],
                         facecolor=color_list[0],alpha=.20)
    plt.fill_between(mbin,Pr_mh[:,1]+err_bar_b[:,0],Pr_mh[:,1]+err_bar_b[:,1],
                         facecolor=color_list[1],alpha=.20)
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$log{\langle M_{200} \rangle/[M_\odot h^{-1}]}$')    
    plt.legend(loc=2)
    plt.title(r'$Standard Deviation-1\sigma_{p}$')
    plt.axis([10,13.5,11,13.5])
    #plt.savefig('Standard_deviation-1.png',dpi=600)
    plt.show()
    ##比较X^2差值的变化
    ###(该部分同时说明怎么样循环生成排列子图)
    plt.figure()
    for k in range(0,2*len(m_bin)):
        plt.subplot(241+k)
        if k<=3:
            plt.plot(m[k],D_chi_r[k],'r')
            plt.title(r'$\Delta\chi^2_{red}$')
            plt.axis([b_m_r[k]+1,b_m_r[k]-1,0,1])
        else:
            plt.plot(m[k-4],D_chi_b[k-4],'b')
            plt.title(r'$\Delta\chi^2_{blue}$')
            plt.axis([b_m_b[k-4]+1,b_m_b[k-4]-1,0,1])
        plt.grid()
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0,hspace=0)
    #plt.savefig('delta_x^2',dpi=600)
    plt.show()
    ##观察概率分布函数，在大质量端取样性应该越来越好
    plt.figure()
    for k in range(0,2*len(m_bin)):
        plt.subplot(241+k)
        if k<=3:
            plt.plot(m[k-4],Dp_r[k-4],'r')
            plt.title(r'$probability_{red}$')
        else:
            plt.plot(m[k-4],Dp_b[k-4],'b')
            plt.title(r'$Probability_{blue}$')
        plt.grid()
    plt.tight_layout()
    #plt.savefig('probability',dpi=600)
    plt.show()
    ##做出由x^2直接得到的1-sigma的误差棒图像
    ######注意插值函数的要求：在差值区间要求目标函数为单调递增函数
    Dchir = np.zeros((len(m_bin),2),dtype=np.float)
    Dchib = np.zeros((len(m_bin),2),dtype=np.float) 
    for k in range(0,len(m_bin)):
        mr_ = m[k]
        mb_ = m[k]
        ##把相应的x^2的差值分两个区间
        Dchir1_ = D_chi_r[k]
        ix = mr_ < b_m_r[k]
        Dchir_1 = Dchir1_[ix]
        mr1 = mr_[ix]
        Dchir_2 = Dchir1_[~ix]
        mr2 = mr_[~ix]
        ##下面求1-sigma的点
        m_r1 = np.interp(-1.0,-Dchir_1,mr1)
        m_r2 = np.interp(1.0,Dchir_2,mr2) 
        ##把相应的x^2的差值分两个区间
        Dchib1_ = D_chi_b[k]
        ix = mb_ < b_m_b[k]
        Dchib_1 = Dchib1_[ix]
        mb1 = mb_[ix]
        Dchib_2 = Dchib1_[~ix]
        mb2 = mb_[~ix]
        ##下面求1-sigma的点
        m_b1 = np.interp(-1.0,-Dchib_1,mb1)
        m_b2 = np.interp(1.0,Dchib_2,mb2)
        ##插值后比较
        Dchir[k,:] = np.array([m_r1-b_m_r[k],m_r2-b_m_r[k]])
        Dchib[k,:] = np.array([m_b1-b_m_b[k],m_b2-b_m_b[k]])
    print(Dchir)
    print(Dchib)
    plt.figure()
    line1,caps1,bars1=plt.errorbar(mbin,Pr_mh[:,0],yerr=[err_bar_r[:,0],err_bar_r[:,1]],fmt="rs-",linewidth=1,
                                elinewidth=0.5,ecolor='k',capsize=1,capthick=0.5,label='red')
    line3,caps3,bars3=plt.errorbar(mbin,Pr_mh[:,1],yerr=[err_bar_b[:,0],err_bar_b[:,1]],fmt="b^-",linewidth=1,
                                elinewidth=0.5,ecolor='k',capsize=1,capthick=0.5,label='blue')
    plt.fill_between(mbin,Pr_mh[:,0]+err_bar_r[:,0],Pr_mh[:,0]+err_bar_r[:,1],
                         facecolor=color_list[0],alpha=.20)
    plt.fill_between(mbin,Pr_mh[:,1]+err_bar_b[:,0],Pr_mh[:,1]+err_bar_b[:,1],
                         facecolor=color_list[1],alpha=.20)
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$log{\langle M_{200} \rangle/[M_\odot h^{-1}]}$')    
    plt.legend(loc=4)
    plt.title(r'$Standard Deviation-1\sigma_{\chi^2}$')
    plt.axis([10,13.5,11,14])
    #plt.savefig('Standard_deviation.png',dpi=600)
    plt.show()
    ##下面部分做x^2检测
    print('min_x^2_r=',c_xs_r)    
    print('min_x^2_b=',c_xs_b)
    return Pr_mh
fig_mass(pp=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''