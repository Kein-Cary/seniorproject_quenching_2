####该脚本考虑实际红移，观察Mh的分布
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
#件的路径具体到文件夹
import os.path
from read_m16_1 import test_read_m16_ds_1
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.stats import t as st
#import seaborn as sns
#from scipy.stats import chisquare as chi
#画误差棒函数须引入scipy库
#导入的质量区间为11~15dex-Msolar
def input_mass_bin(v):
    _mh_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/'
    fname = os.path.join(_mh_path,'z_eff.txt')
    z = np.loadtxt(fname,delimiter=',',dtype=np.float,unpack=True)  
    z = np.array(z)
    N = 4
    m = np.array([np.linspace(11,12.75,1001),np.linspace(11,12.75,1001),\
                  np.linspace(11.75,13,2001),np.linspace(12.25,13.75,2001)])
    #针对不同质量区间设置拟合质量范围
    return m,N,z
#input_mass_bin(v=True)
def calcu_sigmaz(Rpc,m_,x,z):
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
    return Rpc,Sigma,deltaSigma,rs
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
    return x_mean,x_se,conf_region
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
    return x_mean,x_se,conf_region
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
    conf_region = st.ppf(1-alpha/2.,dof)*x_std*np.sqrt(1.+1./nn)#设置置信区间
    plt.errorbar(Rpc[:],ds_sim[k,t,:],yerr=x_std,fmt='-',linewidth=0.5)
    plt.xlabel(r'$R-Mpc/h$')
    plt.ylabel(r'$\Delta\Sigma-hM_\odot/{pc^2}$')   
    return x_mean,x_se,conf_region
#fig_0f2(f=True)
#def fit_datar(y):
def fit_datar_1(mass_bin,z_int,T):
    h = 0.673
    #先找出观测值对应的rp
    #rsa,dssa,ds_errsa = test_read_m16_ds_1()
    rsa,dssa,ds_errsa = test_read_m16_ds_1(mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    #a = np.shape(dssa)
    #print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    #z_r = 0.1
    #a_r = 1/(1+z_r)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=5])
   # rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
   # rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    #print(rp)
    b = len(rp)
    m,N,z = input_mass_bin(v=True)
    mr = m[T]
    #print('mr=',mr)
    lm_bin = len(mr) 
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simr = np.zeros((lm_bin,b),dtype=np.float)
    #rr = np.zeros(lm_bin,dtype=np.float)
    #该句用于下面相应作图部分
    for t in range(0,lm_bin):
        m_ = 1
        x = mr[t]
        z_r = z_int
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
    #print('rr=',rr)#输出查看ds_sim的维度，即模型预测下的透镜信号    
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
def fit_datab_1(mass_bin,z_int,T):
    h = 0.673
    #先找出观测值对应的rp
    #rsa,dssa,ds_errsa = test_read_m16_ds_1()
    rsa,dssa,ds_errsa = test_read_m16_ds_1(mass_bin=mass_bin)
    # print('ds=',dssa)
    # print('ds_err=',ds_errsa)
    #print('rp=',rsa)  
    #a = np.shape(dssa)
    #print('size a=',a)
    ##注意到观测数值选取的是物理坐标，需要靠里尺度银子修正，修正如下
    #z_b = 0.1
    #a_b = 1/(1+z_b)
    ##此时对于预测的R也要做修正
    rp = np.array([rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=5]) 
    # rp = [rsa[0,k] for k in range(0,len(rsa[0,:])) if rsa[0,k]*h<=2]
    # rp = rsa[0,:]
    #该句直接对数组筛选，计算2Mpc以内（保证NFW模型适用的情况下）信号
    #print(rp)
    b = len(rp)
    m,N,z = input_mass_bin(v=True)
    mb = m[T]
    #print('mr=',mb)
    lm_bin = len(mb) 
    #下面做两组数据的方差比较,注意不能对观测数据进行插值
    #比较方法，找出观测的sigma数值对应的Rp,再根据模型计算此时的模型数值Sigma（该步以完成）    
    ds_simb = np.zeros((lm_bin,b),dtype=np.float)
    #rb = np.zeros(lm_bin,dtype=np.float)
    #该句对应下面作图部分
    for t in range(0,lm_bin):
        m_ = 1
        x = mb[t]
        z_b = z_int
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
    #print('rr=',rr)#输出查看ds_sim的维度，即模型预测下的透镜信号    
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
    m,N,z = input_mass_bin(v=True)
    #调用时需要修改下面两句函数的关键字
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
    #作图取点
    m,N,z = input_mass_bin(v=True)
    z_use = np.zeros((len(m_bin),2),dtype=np.float)
    ##计算误差棒并存储的数组
    err_bar_r = np.zeros((len(m_bin),2),dtype=np.float)
    err_bar_b = np.zeros((len(m_bin),2),dtype=np.float)
    for k in range(0,len(m_bin)):
        if k==len(m_bin)-1:
            z_use[k,:] = z[:,-1]
        else:
            z_use[k,:] = z[:,k]
    print(z_use)
    #print(m_bin)
    #该句导入拟合质量
    Pr_mh = np.zeros((len(m_bin),2),dtype=np.float)
    b_m_r = np.zeros(len(m_bin),dtype=np.float)
    b_m_b = np.zeros(len(m_bin),dtype=np.float)
    #Pr_mh的第一行存red的质量，第二行存blue的质量,x_std也一样
    for k in range(0,len(m_bin)):
        rp,best_mr,delta_r,bestfmr = fit_datar_1(m_bin[k],z_use[k,0],k)
        rp,best_mb,delta_b,bestfmb = fit_datab_1(m_bin[k],z_use[k,1],k)
        Pr_mh[k,0] = best_mr
        Pr_mh[k,1] = best_mb
        b_m_r[k] = bestfmr
        b_m_b[k] = bestfmb
        ##下面把x^2转化为相应的概率分布，并让mh的划分区间服从这个分布
        mr = m[k]
        mb = m[k]
        rsa,dssa,ds_errsa = test_read_m16_ds_1(mass_bin=m_bin[k])
        pr = np.ones(len(m[k]),dtype=np.float)
        pb = np.ones(len(m[k]),dtype=np.float)
        ds_errsar = ds_errsa[0,0:len(rp)]
        ds_errsab = ds_errsa[1,0:len(rp)]
        for q in range(0,len(mr)):
            ss=1
            for t in range(0,len(rp)):
                ss = ss*(1/(np.sqrt(np.pi*2)*ds_errsar[t]))*np.exp((-1/2)*delta_r[q])
            pr[q]=ss
        for q in range(0,len(mb)):
            qq=1
            for t in range(0,len(rp)):
                qq = qq*(1/(np.sqrt(np.pi*2)*ds_errsab[t]))*np.exp((-1/2)*delta_b[q])
            pb[q]=qq
        ##第一次归一化，把概率密度函数归一到0~1之间
        pr = (pr-np.min(pr))/(np.max(pr)-np.min(pr))
        pb = (pb-np.min(pb))/(np.max(pb)-np.min(pb))
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
        '''
        #print('r=',delta_r)
        #print('b=',delta_b)
        ##第二次归一化，目的是让所有概率求和为1
        fr=np.zeros(len(mr),dtype=np.float)
        fb=np.zeros(len(mb),dtype=np.float)
        ss=0
        qq=0
        Fr=np.zeros(len(mr),dtype=np.float)
        Fb=np.zeros(len(mb),dtype=np.float)
        for q in range(0,len(mr)):
            ss=ss+pr[q]*(mr[1]-mr[0])
            fr[q]=ss
        for q in range(0,len(mb)):
            qq=qq+pb[q]*(mb[1]-mb[0])
            fb[q]=qq
        Ar = np.max(fr)
        Ab = np.max(fb)
        ss=0
        qq=0
        for q in range(0,len(mr)):
            ss=ss+pr[q]*(mr[1]-mr[0])/Ar
            Fr[q]=ss
        for q in range(0,len(mb)):
            qq=qq+pb[q]*(mb[1]-mb[0])/Ab
            Fb[q]=qq   
        #plt.figure()
        #plt.plot(mr,Fr,'r')
        #plt.show()
        #plt.plot(mb,Fb,'b')
        #plt.show()
        vr = np.interp(0.16,Fr,mr)
        ur = np.interp(0.84,Fr,mr)
        err_bar_r[k,:] = [vr-bestfmr,ur-bestfmr]
        vb = np.interp(0.16,Fb,mb)
        ub = np.interp(0.84,Fb,mb)
        err_bar_b[k,:] = [vb-bestfmb,ub-bestfmb]
    ##对应for循环前面的plt.figure
    #print('phy_mr=',Pr_mh[:,0])
    #print('phy_mb=',Pr_mh[:,1])
    #print('errbar_r=',err_bar_r)
    #print('errbar_b=',err_bar_b)
    #####注意errorbar作图命令
    plt.figure()
    color_list = ['r','b','g']
    line1,caps1,bars1=plt.errorbar(mbin,Pr_mh[:,0],yerr=[err_bar_r[:,0],err_bar_r[:,1]],fmt="ks-",linewidth=1,
                                elinewidth=0.5,ecolor='k',capsize=1,capthick=0.5,label='red')
    line3,caps3,bars3=plt.errorbar(mbin,Pr_mh[:,1],yerr=[err_bar_b[:,0],err_bar_b[:,1]],fmt="k^-",linewidth=1,
                                elinewidth=0.5,ecolor='k',capsize=1,capthick=0.5,label='blue')
    plt.fill_between(mbin,Pr_mh[:,0]+err_bar_r[:,0],Pr_mh[:,0]+err_bar_r[:,1],
                         facecolor=color_list[0],alpha=.20)
    plt.fill_between(mbin,Pr_mh[:,1]+err_bar_b[:,0],Pr_mh[:,1]+err_bar_b[:,1],
                         facecolor=color_list[1],alpha=.20)
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$log{\langle M_{200} \rangle/[M_\odot h^{-1}]}$')    
    plt.legend(loc=2)
    plt.title(r'$Standard Deviation-1\sigma_{p}$')
    plt.axis([10,14,11,14])
    #plt.savefig('Standard_deviation.png',dpi=600)
    plt.show()
    return Pr_mh
fig_mass(pp=True)
'''
if __name__ == "__main__":
    dolad_data(m=True,hz=True)
    pass
'''