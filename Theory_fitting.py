####这部分脚本文件主要对Theory_calculation里面得到的曲线的参数调节，以得到最有解
####MCMC调用部分
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
##件的路径具体到文件夹
###库函数调用
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats as st
import matplotlib.gridspec as gridspec
import pandas as pa
from scipy import interpolate as sinter
from scipy.optimize import minimize
###mock_相关脚本调用，主要用于导入数据
from Theory_calculation import Theory_fun3
def func_fred(mh_q,miu_mh):
    _halo,ms_use,N_ms_Mh,p_joint,p_Mh_ms,p_ms_Mh,P_Joint3,P_Joint3_1,p_ms_Mh1,p_Mh_ms1,p_joint1\
    = Theory_fun3(ff3=True)
##考虑暗晕质量是主要的quenching的原因：f_red_mh,mh_q表示quenching的临界质量
    dms = (10**ms_use[-1]-10**ms_use[0])/len(ms_use)
    dmh = (10**_halo[-1]-10**_halo[0])/len(_halo)
    f_red_mh = 1-np.exp(-(10**_halo/10**mh_q)**miu_mh)
    S4 = np.zeros(len(_halo),dtype=np.float)
    for k in range(len(ms_use)):
        S4[k] = np.sum(f_red_mh*p_joint[:,k]*dmh)  
    tot_f_red_mh = np.sum(S4*dms)
    #for red sequence,the joint distribution as flow 
    p_red_msmh2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        p_red_msmh2[:,k] = p_joint[:,k]*f_red_mh/tot_f_red_mh
    #求解横行质量的边界分布
    p_red_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        p_red_ms2[k] = np.sum(p_red_msmh2[:,k]*dmh)
    #the condition distribution:p_red(Mh|m*) as red_p_Mh_ms
    red_p_Mh_ms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        red_p_Mh_ms2[:,k] = p_red_msmh2[:,k]/p_red_ms2[k]
    red_p_Mh_ms2[np.isnan(red_p_Mh_ms2)]=0
    red_p_Mh_ms2[np.isinf(red_p_Mh_ms2)]=0
    ###对条件概率做归一化
    red_pMhms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)   
    for k in range(len(ms_use)):
        s = np.sum(red_p_Mh_ms2[:,k]*dmh)
        red_pMhms2[:,k] = red_p_Mh_ms2[:,k]/s
    #############
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    red_Mh_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        red_Mh_ms2[k] = np.sum(red_p_Mh_ms2[:,k]*10**_halo*dmh)/(np.sum(red_p_Mh_ms2[:,k]*dmh))
    ###下面求errorbar
    '''
    red_Mh_err2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):      
        mh_err = _halo
        p_m = red_p_Mh_ms2[:,k]
        mh = np.sum(dMh*(10**mh_err-red_Mh_ms2[k])**2*p_m)/np.sum(p_m*dMh)
        mh = np.sqrt(mh)
        red_Mh_err2[k] = np.log10(mh)
    '''
    red_Mh_err2 = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = red_p_Mh_ms2[:,k]
        F_m3 = np.zeros(len(mh_err),dtype=np.float)
        F_m3[0] = 0
        for t in range(len(_halo)):
            F_m3[t] = F_m3[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err5 = np.interp(0.1585,F_m3,mh_err)-np.log10(red_Mh_ms2[k])
        va_err6 = np.interp(0.8415,F_m3,mh_err)-np.log10(red_Mh_ms2[k])
        red_Mh_err2[k,:] = np.array([va_err5,va_err6]) 
    ####此时对blue sequence的求解
    S5 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        S5[k] = np.sum(p_joint[:,k]*dmh)
    P_tot2 = np.sum(S5*dms)
    #求解该情况下联合分布概率密度
    f_blue_mh = 1-f_red_mh
    p_blue_msmh2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        p_blue_msmh2[:,k] = f_blue_mh*p_joint[:,k]/(P_tot2 - tot_f_red_mh)
    #求解恒星质量的边界分布
    p_blue_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        p_blue_ms2[k] = np.sum(p_blue_msmh2[:,k]*dmh)
    #the condition distribution:p_blue(Mh|m*) as blue_p_Mh_ms
    blue_p_Mh_ms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        blue_p_Mh_ms2[:,k] = p_blue_msmh2[:,k]/p_blue_ms2[k]
    blue_p_Mh_ms2[np.isnan(blue_p_Mh_ms2)]=0
    blue_p_Mh_ms2[np.isinf(blue_p_Mh_ms2)]=0
    #对条件概率归一化
    blue_pMhms2 = np.zeros((len(_halo),len(ms_use)),dtype=np.float)
    for k in range(len(ms_use)):
        s = np.sum(blue_p_Mh_ms2[:,k]*dmh)
        blue_pMhms2[:,k] = blue_p_Mh_ms2[:,k]/s
    #########
    ###下面求各个恒星质量区间的理论上的暗晕质量<Mh|M*>
    blue_Mh_ms2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):
        blue_Mh_ms2[k] = np.sum(blue_p_Mh_ms2[:,k]*10**_halo*dmh)/(np.sum(blue_p_Mh_ms2[:,k]*dmh))
    ###下面求errorbar
    '''
    blue_Mh_err2 = np.zeros(len(ms_use),dtype=np.float)
    for k in range(len(ms_use)):      
        mh_err = _halo
        p_m = blue_p_Mh_ms2[:,k]
        mh = np.sum(dMh*(10**mh_err-blue_Mh_ms2[k])**2*p_m)/np.sum(p_m*dMh)
        mh = np.sqrt(mh)
        blue_Mh_err2[k] = np.log10(mh)
    '''
    blue_Mh_err2 = np.zeros((len(ms_use),2),dtype=np.float)
    for k in range(len(ms_use)):
        mh_err = _halo
        p_m = blue_p_Mh_ms2[:,k]
        F_m4 = np.zeros(len(mh_err),dtype=np.float)
        F_m4[0] = 0
        for t in range(len(_halo)):
            F_m4[t] = F_m4[t-1]+p_m[t]*dmh/np.sum(p_m*dmh)
        va_err7 = np.interp(0.1585,F_m4,mh_err)-np.log10(blue_Mh_ms2[k])
        va_err8 = np.interp(0.8415,F_m4,mh_err)-np.log10(blue_Mh_ms2[k])
        blue_Mh_err2[k,:] = np.array([va_err7,va_err8]) 
    return red_Mh_ms2,red_Mh_err2,blue_Mh_ms2,blue_Mh_err2,_halo,ms_use
#func_fred(mh_q,miu_mh)
def fit_data(t):
    t=t
    if t==1:
        ####导入M16观测数据
        mh_r = np.array([12.17,12.14,12.50,12.89,13.25,13.63,14.05])
        mh_r_err = np.array([[0.19,0.12,0.04,0.04,0.03,0.03,0.05],
                              [-0.24,-0.14,-0.05,-0.04,-0.03,-0.03,-0.05]])
        mhrerr = (1/2)*(np.abs(mh_r_err[0])+np.abs(mh_r_err[1]))##把上下误差做一个平均处理
        mh_r_err = mhrerr
        ms_r = np.array([10.28,10.58,10.86,11.10,11.29,11.48,11.68])
        mh_b = np.array([11.80,11.73,12.15,12.61,12.69,12.79,12.79])
        mh_b_err = np.array([[0.16,0.13,0.08,0.10,0.19,0.43,0.58],
                              [-0.20,-0.17,-0.10,-0.11,-0.25,-1.01,-2.23]])
        mhberr = (1/2)*(np.abs(mh_b_err[0])+np.abs(mh_b_err[1]))##把上下误差做一个平均处理
        mh_b_err = mhberr
        ms_b = np.array([10.24,10.56,10.85,11.10,11.28,11.47,11.68] )
        ####对M16的数据,还要注意把恒星质量从Msolar的单位转化为Msolar/h^-2
        ####C_m16的结果对应部分
        h = 0.72
        delta_value = np.log10(h)
        ms_r = ms_r+2*delta_value 
        ms_b = ms_b+2*delta_value
    elif t==0:
        ####导入模拟的观测数据(数据点比较多情况)
        mh_r= np.array([11.84144843,11.84144843,11.89535823,11.95580005,12.02537811,\
              12.10329057,12.19181432,12.29147487,12.40331193,12.5262711,\
              12.66457476,12.80943267,12.96171603,13.12992385,13.30603297,\
              13.49122876,13.65855888,13.85375991,14.01142053,14.05900916,\
              14.18654906])
        ms_r = np.array([9.55516636,9.66545281,9.77573927,9.88602573,9.99631218,\
              10.10659864,10.2168851,10.32717155,10.43745801,10.54774447,\
              10.65803092,10.76831738,10.87860384,10.98889029,11.09917675,\
              11.20946321,11.31974967,11.43003612,11.54032258,11.65060904,11.76089549])
        mh_r_err= np.array([0.01,0.51726065,0.513776,0.51061946,0.50786479,0.50636558,\
              0.5054495,0.505326,0.5058986,0.50688928,0.50698652,0.5107636,\
              0.51497867,0.51649112,0.51241142,0.49964074,0.48506296,0.47290132,\
              0.45835034,0.48461951,0.04812004])
        mh_b= np.array([11.56549616,11.56549616,11.61073136,11.66002609,11.71386246,\
              11.77294388,11.83778009,11.90955185,11.98354181,12.06735616,\
              12.15251732,12.24396081,12.33427442,12.4449957,12.54540254,\
              12.66468538,12.82667239,13.01481881,13.08768458,13.21836804,\
              13.27014179])
        ms_b= np.array([9.54956774,9.64870165,9.74783556,9.84696947,9.94610338,\
              10.0452373,10.14437121,10.24350512,10.34263903,10.44177294,\
              10.54090685,10.64004076,10.73917467,10.83830858,10.93744249,\
              11.0365764,11.13571031,11.23484422,11.33397813,11.43311204,11.53224595])
        mh_b_err= np.array([0.01,0.31073456,0.31014416,0.31049053,0.31156874,0.31375724,\
              0.3163361,0.3207385,0.32540602,0.32897889,0.33508065,0.34308683,\
              0.34839135,0.3588951,0.36576389,0.39988588,0.40469186,0.34897021,\
              0.34346791,0.22070814,0.28744449]) 
    else:
        ####导入模拟的观测数据(数据点比较少情况)
        mh_r= np.array([11.93213772,11.93213772,12.12434067,12.38838582,12.73441978,\
               13.15024239,13.62083619,14.05767997])
        ms_r = np.array([9.6447741,9.93427605,10.223778,10.51327995,10.8027819,\
                 11.09228385,11.3817858,11.67128775])
        mh_r_err = np.array([0.01,0.51185669,0.50611527,0.50602658,0.5087693,0.51648376,\
                 0.48655984,0.42089061])
        mh_b = np.array([11.64123409,11.64123409,11.7885786,11.97499485,12.19929409,\
               12.45631101,12.75363835,13.02485297])
        ms_b = np.array([9.63011405,9.89034056,10.15056708,10.41079359,10.67102011,\
             10.93124662,11.19147314,11.45169965])
        mh_b_err = np.array([0.01,0.31022394,0.31428295,0.32420174,0.33966039,0.36085605,\
                 0.4146244,0.333523])
    return mh_r,ms_r,mh_r_err,mh_b,ms_b,mh_b_err
#fit_data(t=1)
###################
def fit_best(dd):
    mh_r,ms_r,mh_r_err,mh_b,ms_b,mh_b_err = fit_data(t=1)
    ##给定需要优化的参数范围
    mh_q = np.linspace(11.0,15.5,11)
    miu_mh = np.linspace(0,3.0,11)
    chi_v = np.zeros((len(mh_q),len(miu_mh)),dtype=np.float)
    for k in range(len(mh_q)):
        for t in range(len(miu_mh)):
            red_Mh_ms2,red_Mh_err2,blue_Mh_ms2,blue_Mh_err2,_halo,ms_use = func_fred(mh_q[k],miu_mh[t])
            x1 = ms_use
            y1 = np.log10(red_Mh_ms2)
            x2 = ms_use
            y2 = np.log10(blue_Mh_ms2)
            f1 = sinter.interpolate.interp1d(x1,y1)
            f2 = sinter.interpolate.interp1d(x2,y2)
            T_mh_r = f1(ms_r)
            T_mh_b = f2(ms_b)
            chi_1 = 0
            chi_2 = 0
            for m in range(len(mh_r)):  
                chi1 = chi_1+((T_mh_r[m]-mh_r[m])/(mh_r_err[m]))**2
            for n in range(len(mh_b)):
                chi2 = chi_2+((T_mh_b[n]-mh_b[n])/(mh_b_err[n]))**2
            chi_v[k,t] = chi1+chi2  
    return chi_v,mh_q,miu_mh  
#fit_best(dd=True)
def fig_fit_best(d):
    chi_v,mh_q,miu_mh = fit_best(dd=True)
    plt.pcolormesh(mh_q,miu_mh,chi_v.T,cmap='rainbow',vmin=1e-1,vmax=np.max(chi_v)+1,alpha=1,
            norm = mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel(r'$logM_h [M_\odot h^{-1}]$')
    plt.ylabel(r'$\mu_{M_h}$')
    #plt.savefig('Parameter_analysis_m16',dpi=600)
    #plt.savefig('Parameter_analysis_t0',dpi=600)
    #plt.savefig('Parameter_analysis_t2',dpi=600)
    #plt.savefig('C_m16_Parameter_analysis_m16',dpi=600)
    plt.show()
    return 
fig_fit_best(d=True)
######################
def youhua(x):
    #mh_q = 12.25
    #miu_mh = 0.425
    mh_q = x[0]
    miu_mh = x[1]
    mh_r,ms_r,mh_r_err,mh_b,ms_b,mh_b_err = fit_data(t=2)
    red_Mh_ms2,red_Mh_err2,blue_Mh_ms2,blue_Mh_err2,_halo,ms_use = func_fred(mh_q,miu_mh)
    x1 = ms_use
    y1 = np.log10(red_Mh_ms2)
    x2 = ms_use
    y2 = np.log10(blue_Mh_ms2)
    f1 = sinter.interpolate.interp1d(x1,y1)
    f2 = sinter.interpolate.interp1d(x2,y2)
    T_mh_r = f1(ms_r)
    T_mh_b = f2(ms_b)
    chi_1 = 0
    chi_2 = 0
    for k in range(len(mh_r)):
        chi1 = chi_1+((T_mh_r[k]-mh_r[k])/(mh_r_err[k]))**2
    for t in range(len(mh_b)):
        chi2 = chi_2+((T_mh_b[t]-mh_b[t])/(mh_b_err[t]))**2
    chi_v = chi1+chi2 
    return chi_v
#youhua(x=True) 
#youhua(mh_q=True,miu_mh=Truem)
def youhua_fit(f):
    mh_q = np.array(np.linspace(11.0,15.5,10))
    miu_mh = np.array(np.linspace(0,3.0,11))  
    pre1 = minimize(youhua,x0=np.array([mh_q[0],miu_mh[0]]), method='Powell',tol=1e-5)
    #pre1 = minimize(youhua,x0=np.array([mh_q[0],miu_mh[0]]),method='L-BFGS-B',tol=1e-5)
    ##下面两句表示可以根据调用函数的关键字返回自己需要的值
    print('pre1=',pre1)
    return pre1
#youhua_fit(f=True)