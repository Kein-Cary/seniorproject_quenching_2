##这部分脚本针对观测数据做预测优化
##主要变量导入：dolad_data.py
import sys
# sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
from read_m16 import test_read_m16_ds
# from Desigma_checking import constant_f
# from log_jifen import rou_R
# from log_jifen import rou2
# from dolad_data import dolad_data
# from dolad_data import semula_tion
from dolad_data import fit_data
from dolad_data import calcu_sigma
# from dolad_data import fig_ff
##根据做图需要调整该句（作图模块在最后）
import numpy as np
# import matplotlib.pyplot as plt
##作图时需要取消这句注释（作图模块在最后）
from scipy.optimize import minimize
#Section 1:把初步结果的最佳质量做细化拓展
def extend_m(v):
    rp,best_mr,best_mb,dexr,dexb,a,b,yy = fit_data(y=True)
    #实际上主要的影响因素来自指数部分
    #mr = np.linspace(best_mr-2,best_mr+6,1000)
    #mb = np.linspace(best_mb,best_mb+8,1000)
    mr = best_mr
    mb = best_mb
    d_exr = np.linspace(dexr-1,dexr+1,1000)
    d_exb = np.linspace(dexb-1,dexb+1,1000)
    #d_exr = dexr
    #d_exb = dexb
   # print('mr=',best_mr)
   # print('mb=',best_mb)
   # print('dexr=',d_exr)
   # print('dexb=',d_exr)   
    return(rp,mr,mb,d_exr,d_exb,b)
#extend_m(v=True)
def va_da(p):
    mr,mb,rp,d_exr,d_exb,b = extend_m(v=True)
    return(mr,mb,rp,d_exr,d_exb,b)
#va_da(p=True)
#Section 2：针对拓展细化的质量区间，继续预测和比较
##分别对红星系和蓝星系求解
def c_fit_datar(x0r):
    # (mr,d_exr) = x0r
    rp,mr,mb,d_exr,d_exb,b = va_da(p=True)
    d_exr = x0r
    #lmr = len(mr)#质量数组的长度
    ldr = len(d_exr)#指数数组的长度
    #dssimr = np.zeros((lmr,ldr,b),dtype=np.float)
    dssimr = np.zeros((ldr,b),dtype=np.float)
    #for k in range(0,lmr):
    for t in range(0,ldr):
        #m_ = mr[k]
        m_ = mr
        xr = d_exr[t]
        cRp = rp
        #计算模型在对应的投射距离上的预测信号取值
        c_Rp,c_Sigmar,c_deltaSigmar = calcu_sigma(cRp,m_,xr)           
        #c_表示该变量做继续比较量
        dssimr[t,:] = c_deltaSigmar           
    size_ds1 = np.shape(dssimr)   
    print(size_ds1)
    delta_r = np.zeros(ldr,dtype=np.float)
    rsa,dssa,ds_errsa = test_read_m16_ds()
    # for k in range(0,lmr):
    for t in range(0,ldr):
        d_r = 0
        for n in range(0,b):
            d_r = d_r+((dssimr[t,n]-dssa[0,n])/ds_errsa[0,n])**2
        delta_r[t] = d_r
    # print(delta_r)
    #下面用bestr和bestb记录最后比较结果
    '''
    #下面求取最小值及索引号
    aa = delta_r.tolist()
    xa = aa.index(min(delta_r))
    deltar = min(delta_r) 
    delta_r = deltar
    bestfr = d_exr[xa]    
    dexr = bestfr
    best_mr = mr*10**dexr
    print(delta_r)
    print('mr=',best_mr)
    print('dexr=',bestfr)
    '''
    return delta_r
#c_fit_datar(mr=True,mb=True,d_exr=True,d_exb=True)
#c_fit_datar(p=True)
##
def c_fit_datab(x0b):
    # (mb,d_exb) = x0b
    rp,mr,mb,d_exr,d_exb,b = va_da(p=True)
    d_exb = x0b
    # lmb = len(mb)#质量数组的长度
    ldb = len(d_exb)
    dssimb = np.zeros((ldb,b),dtype=np.float)
    # for k in range(0,lmb):
    for t in range(0,ldb):
        #m_ = mb[k]
        m_ = mb
        xb = d_exb[t]
        cRp = rp
        #计算模型在对应的投射距离上的预测信号取值
        c_Rp,c_Sigmab,c_deltaSigmab = calcu_sigma(cRp,m_,xb)            
        #c_表示该变量做继续比较量
        dssimb[t,:] = c_deltaSigmab              
    size_ds2 = np.shape(dssimb)   
    print(size_ds2)
    delta_b = np.zeros(ldb,dtype=np.float)
    rsa,dssa,ds_errsa = test_read_m16_ds()
    # for k in range(0,lmb):
    for t in range(0,ldb):
        d_b = 0
        for n in range(0,b):
            d_b = d_b+((dssimb[t,n]-dssa[1,n])/ds_errsa[1,n])**2
        delta_b[t] = d_b
    # print(delta_b)
    #下面用bestr和bestb记录最后比较结果
    #下面求取最小值及索引号
    '''
    bb = delta_b.tolist()
    xb = bb.index(min(delta_b))
    deltab = min(delta_b)  
    delta_b = deltab
    bestfb = d_exb[xb]    
    dexb = bestfb
    best_mb = mr*10**dexb
    print(delta_b)
    print('mb=',best_mb)
    print('dexb=',bestfb)
    '''
    return delta_b
#c_fit_datab(mr=True,mb=True,d_exr=True,d_exb=True)
#c_fit_datab(p=True)
#Section 3:下面设置循环调用比较，逐步优化模块 
def my_fun(d_exr,d_exb):
    mr,mb,rp,d_exr,d_exb,b = va_da(p=True)
    x0r = d_exr
    x0b = d_exb
    #x0r = mr
    #x0b = mb
    pre1 = minimize(c_fit_datar,x0=x0r[0], method='Powell',tol=1e-5)
    pre2 = minimize(c_fit_datab,x0=x0b[0], method='Powell',tol=1e-5)
    #pre1 = minimize(c_fit_datar,x0=x0r[0], method='Nelder-Mead',tol=1e-5)
    #pre2 = minimize(c_fit_datab,x0=x0b[0], method='Nelder-Mead',tol=1e-5)
    ##下面两句表示可以根据调用函数的关键字返回自己需要的值
    print('pre1=',pre1.fun)
    print('pre2=',pre2.fun)
    return pre1,pre2
my_fun(d_exr=True,d_exb=True)
#下面做图比较最佳预测值与观测的对比情况
