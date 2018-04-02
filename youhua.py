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
from dolad_data import fig_ff
##根据做图需要调整该句（作图模块在最后）
import numpy as np
import matplotlib.pyplot as plt
##作图时需要取消这句注释（作图模块在最后）
# from scipy.optimize import fmin 
#Section 1:把初步结果的最佳质量做细化拓展
def extend_m(v):
    rp,best_mr,best_mb,dexr,dexb,a,b,yy = fit_data(y=True)
    mr = np.linspace(best_mr-0.5,best_mr+1,100)
    mb = np.linspace(best_mb-0.5,best_mb+1,100)
    # mr = best_mr
    # mb = best_mb
    d_exr = np.linspace(dexr-2,dexr+1,100)
    d_exb = np.linspace(dexb-2,dexb+1,100)
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
def c_fit_datar(u):
    #(mr,mb,d_exr,d_exb) = arr
    rp,mr,mb,d_exr,d_exb,b = va_da(p=True)
    lmr = len(mr)#质量数组的长度
    ldr = len(d_exr)#指数数组的长度
    dssimr = np.zeros((lmr,ldr,b),dtype=np.float)
    for k in range(0,lmr):
        for t in range(0,ldr):
            m_ = mr[k]
            xr = d_exr[t]
            cRp = rp
            #计算模型在对应的投射距离上的预测信号取值
            c_Rp,c_Sigmar,c_deltaSigmar = calcu_sigma(cRp,m_,xr)           
            #c_表示该变量做继续比较量
            dssimr[k,t,:] = c_deltaSigmar            
    size_ds1 = np.shape(dssimr)   
    print(size_ds1)
    fitr = np.zeros((lmr,2),dtype=np.int)
    delta_rr = np.zeros((lmr,ldr),dtype=np.float)
    rsa,dssa,ds_errsa = test_read_m16_ds()
    for k in range(0,lmr):
        for t in range(0,ldr):
            d_r = 0
            for n in range(0,size_ds1[2]):
                d_r = d_r+((dssimr[k,t,n]-dssa[0,n])/ds_errsa[0,n])**2
            delta_rr[k,t] = d_r
    # print(delta_r)
    #下面这段求每个质量级对应的方差最小值
    for k in range(0,lmr):
        aa = delta_rr[k,:].tolist()
        #先把np的定义数组转为列表，再寻找索引号，tolist()表示内置函数调用
        xa = aa.index(min(delta_rr[k,:]))
        fitr[k,0] = k
        fitr[k,1] = xa
    # print(fitr)
    #比较提取出的五个最小值的最小值为最后结果
    deltar = np.zeros(lmr,dtype=np.float)
    xrr = np.zeros((lmr,2),dtype=np.float)
    for k in range(0,lmr):
        deltar[k] = delta_rr[fitr[k,0],fitr[k,1]]
        xrr[k] = [mr[fitr[k,0]],d_exr[fitr[k,1]]]
    #下面用bestr和bestb记录最后比较结果
    bestfr = np.zeros(2,dtype=np.float)
    #下面求取最小值及索引号
    aa = deltar.tolist()
    xa = aa.index(min(deltar))
    delta_r = min(deltar) 
    bestfr = fitr[xa,:]
    print(bestfr)    
    bestr = mr[bestfr[0]]
    d_ex_r = d_exr[bestfr[1]]
    dexr = d_ex_r
    best_mr = bestr
    print(dexr)
    print(best_mr)
    #作图对比最佳情况
    plt.figure()
    fig_ff(rp,dssimr,lmr,fitr)
    plt.title('Red')
    plt.show()
   # print('mr=',best_mr*10**d_ex_r)
   # return rp,best_mr,dexr,a,delta_r,a,b,yy
    return delta_r,lmr,dssimr,fitr,delta_rr,xrr
#c_fit_datar(mr=True,mb=True,d_exr=True,d_exb=True)
#c_fit_datar(u=True)
##
def c_fit_datab(v):
    #(mr,mb,d_exr,d_exb) = arr
    rp,mr,mb,d_exr,d_exb,b = va_da(p=True)
    lmb = len(mb)#质量数组的长度
    ldb = len(d_exb)
    dssimb = np.zeros((lmb,ldb,b),dtype=np.float)
    for k in range(0,lmb):
        for t in range(0,ldb):
            m_ = mb[k]
            xb = d_exb[t]
            cRp = rp
            #计算模型在对应的投射距离上的预测信号取值
            c_Rp,c_Sigmab,c_deltaSigmab = calcu_sigma(cRp,m_,xb)            
            #c_表示该变量做继续比较量
            dssimb[k,t,:] = c_deltaSigmab              
    size_ds2 = np.shape(dssimb)   
    print(size_ds2)
    fitb = np.zeros((lmb,2),dtype=np.int)
    delta_bb = np.zeros((lmb,ldb),dtype=np.float)
    rsa,dssa,ds_errsa = test_read_m16_ds()
    for k in range(0,lmb):
        for t in range(0,ldb):
            d_b = 0
            for n in range(0,size_ds2[2]):
                d_b = d_b+((dssimb[k,t,n]-dssa[1,n])/ds_errsa[1,n])**2
            delta_bb[k,t] = d_b
    # print(delta_b)
    #下面这段求每个质量级对应的方差最小值
    for k in range(0,lmb):
        bb = delta_bb[k,:].tolist()       
        #先把np的定义数组转为列表，再寻找索引号，tolist()表示内置函数调用
        xb = bb.index(min(delta_bb[k,:]))
        fitb[k,0] = k
        fitb[k,1] = xb
    # print(fitr)
    # print(fitb)
    #比较提取出的五个最小值的最小值为最后结果
    xbb = np.zeros((lmb,2),dtype=np.float)
    deltab = np.zeros(lmb,dtype=np.float)
    for k in range(0,lmb):
        deltab[k] = delta_bb[fitb[k,0],fitb[k,1]]
        xbb[k] = [mb[fitb[k,0]],d_exb[fitb[k,1]]]
    #下面用bestr和bestb记录最后比较结果
    bestfb = np.zeros(2,dtype=np.float)
    #下面求取最小值及索引号
    bb = deltab.tolist()
    xb = bb.index(min(deltab))
    delta_b = min(deltab)  
    bestfb = fitb[xb,:]
    print(bestfb)    
    bestb = mb[bestfb[0]]
    d_ex_b = d_exb[bestfb[1]]
    dexb = d_ex_b
    best_mb = bestb
    print(dexb)
    print(best_mb)
    #作图对比最佳情况
    plt.figure()
    fig_ff(rp,dssimb,lmb,fitb)
    plt.title('blue')
    plt.show()
   # print('mb=',best_mb*10**dexb)
   # return rp,best_mb,,dexb,a,b,yy,delta_b 
    return delta_b,lmb,dssimb,fitb,delta_bb,xbb
#c_fit_datab(mr=True,mb=True,d_exr=True,d_exb=True)
#c_fit_datab(v=True)
#Section 3:下面设置循环调用比较，逐步优化模块 
'''
def my_fun(mr,mb,d_exr,d_exb):
    mr=3
    mb = 1
    d_exr = 14
    d_exb = 14
    arr = [mr,mb,d_exr,d_exb]
    x0 = arr
    X_pre = fmin(c_fit_datar,x0)
    Y_pre = fmin(c_fit_datab,x0)
    print('r=',X_pre)
    print('b=',Y_pre)
    return()
my_fun(mr=True,mb=True,d_exr=True,d_exb=True)
'''
#下面做图比较最佳预测值与观测的对比情况
def fig_(T):
    rp,mr,mb,d_exr,d_exb,b = va_da(p=True)
    # arr = [mr,mb,d_exr,d_exb]
    delta_r,lmr,dssimr,fitr,delta_rr,xrr = c_fit_datar(u=True)
    delta_b,lmb,dssimb,fitb,delta_bb,xbb = c_fit_datab(v=True)
    cRp = rp
    plt.subplot(121)
    fig_ff(cRp,dssimr,lmr,fitr)  
    plt.title('R-galaxy')
    plt.subplot(122)
    fig_ff(cRp,dssimb,lmb,fitb) 
    plt.title('B-galaxy')
    plt.tight_layout()
    plt.show()
    plt.subplot(121)
    for k in range(0,lmr):
        plt.plot(d_exr,np.log10(delta_rr[k,:]))
    plt.title('R-galaxy')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.grid()
    plt.subplot(122)
    for k in range(0,lmb):
        plt.plot(d_exb,np.log10(delta_bb[k,:]))
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.title('B-galaxy')
    plt.grid()
    #plt.tight_layout(pad=2.0,w_pad=2.0,h_pad=2.0)
    plt.tight_layout()
    plt.show()
    xr = np.mean(xrr[:,1])
    xb = np.mean(xbb[:,1])
    print('mr_bar=',xr)
    print('mb_bar=',xb)
    #从图上对比，最佳拟合质量应该在10^10M-solar
fig_(T=True)
