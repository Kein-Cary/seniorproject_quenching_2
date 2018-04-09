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
from dolad_data import fit_datar
from dolad_data import fit_datab
#from dolad_data import calcu_sigma
from dolad_data import calcu_sigmaz
from dolad_data import fig_fff1
##根据做图需要调整该句（作图模块在最后）
import numpy as np
import matplotlib.pyplot as plt
##作图时需要取消这句注释（作图模块在最后）
#from scipy.optimize import minimize
#Section 1:把初步结果的最佳质量做细化拓展
def extend_m(v):
    #rp,best_mr,best_mb,dexr1,dexb1,a,b,yy,dexr,dexb = fit_data(y=True)
    #修正
    rp,best_mr,dexr1,a,b,yy,dexr,a_r = fit_datar(y=True)
    rp,best_mb,dexb1,a,b,yy,dexb,a_b = fit_datab(y=True)    
    #实际上主要的影响因素来自指数部分
    #mr = best_mr
    #mb = best_mb
    mr = 1
    mb = 1
    print('mr=',mr)
    print('mb=',mb)
    d_exr = np.linspace(dexr1-2,dexr1+2,400)
    d_exb = np.linspace(dexb1-2,dexb1+2,400)
    #d_exr = dexr
    #d_exb = dexb
   # print('mr=',best_mr)
   # print('mb=',best_mb)
   # print('dexr=',d_exr)
   # print('dexb=',d_exr)   
    return(rp,mr,mb,d_exr,d_exb,b,a_r,a_b)
#extend_m(v=True)
def va_da(p):
    mr,mb,rp,d_exr,d_exb,b,a_r,a_b = extend_m(v=True)
    return(mr,mb,rp,d_exr,d_exb,b,a_r,a_b)
#va_da(p=True)
#Section 2：针对拓展细化的质量区间，继续预测和比较
##分别对红星系和蓝星系求解
def c_fit_datar(x0r):
    # (mr,d_exr) = x0r
    rp,mr,mb,d_exr,d_exb,b,a_r,a_b = va_da(p=True)
    #d_exr = x0r
    #lmr = len(mr)#质量数组的长度
    ldr = len(d_exr)#指数数组的长度
    #dssimr = np.zeros((lmr,ldr,b),dtype=np.float)
    dssimr = np.zeros((ldr,b),dtype=np.float)
    #for k in range(0,lmr):
    for t in range(0,ldr):
        #m_ = mr[k]
        m = mr
        xr = d_exr[t]
        #cRp = rp
        #dssimr[t,:] = c_deltaSigmar
        #c_Rp,c_Sigmar,c_deltaSigmar = calcu_sigma(cRp,m,xr)   
        #修正
        cRp = rp
        z_r = 0.105
        #计算模型在对应的投射距离上的预测信号取值        
        #c_表示该变量做继续比较量
        #修正
        c_Rp,c_Sigmar,c_deltaSigmar = calcu_sigmaz(cRp,m,xr,z_r)
        dssimr[t,:] = c_deltaSigmar
    size_ds1 = np.shape(dssimr)   
    print('size_r=',size_ds1)
    delta_r = np.zeros(ldr,dtype=np.float)
    rsa,dssa,ds_errsa = test_read_m16_ds()
    # for k in range(0,lmr):
    for t in range(0,ldr):
        d_r = 0
        for n in range(0,b):
            d_r = d_r+((dssimr[t,n]-dssa[0,n])/ds_errsa[0,n])**2
        delta_r[t] = d_r
    #print(delta_r)
    #下面用bestr和bestb记录最后比较结果
    #下面求取最小值及索引号
    aa = delta_r.tolist()
    xa = aa.index(min(delta_r))
    deltar = min(delta_r) 
    #delta_r = deltar
    bestfr = d_exr[xa]    
    dexr = bestfr
    best_mr = mr*10**dexr
    #对比修正
    h = 0.673
    dexr11 = bestfr+np.log10(mr*h)
    #作图对比最佳情况
    plt.figure()
    fig_fff1(rp,dssimr,xa)
    plt.title('Red')
    plt.show()
    print('corr_mr=',10**dexr11)
    print('co_dexr=',dexr11)
    print('x^2=',deltar)
    print('mr=',best_mr)
    print('dexr=',bestfr)
    return delta_r,rp,ldr
#c_fit_datar(mr=True,mb=True,d_exr=True,d_exb=True)
#c_fit_datar(x0r=True)
##
def c_fit_datab(x0b):
    # (mb,d_exb) = x0b
    rp,mr,mb,d_exr,d_exb,b,a_r,a_b = va_da(p=True)
    #d_exb = x0b
    # lmb = len(mb)#质量数组的长度
    ldb = len(d_exb)
    dssimb = np.zeros((ldb,b),dtype=np.float)
    # for k in range(0,lmb):
    for t in range(0,ldb):
        #m_ = mb[k]
        m = mb
        xb = d_exb[t]
        #cRp = rp
        #计算模型在对应的投射距离上的预测信号取值
        #c_Rp,c_Sigmab,c_deltaSigmab = calcu_sigma(cRp,m,xb)
        #dssimb[t,:] = c_deltaSigmab
        #修正
        cRp = rp  
        z_b = 0.124        
        #c_表示该变量做继续比较量
        #修正
        c_Rp,c_Sigmab,c_deltaSigmab = calcu_sigmaz(cRp,m,xb,z_b)
        dssimb[t,:] = c_deltaSigmab 
    size_ds2 = np.shape(dssimb)   
    print('size_b=',size_ds2)
    delta_b = np.zeros(ldb,dtype=np.float)
    rsa,dssa,ds_errsa = test_read_m16_ds()
    # for k in range(0,lmb):
    for t in range(0,ldb):
        d_b = 0
        for n in range(0,b):
            d_b = d_b+((dssimb[t,n]-dssa[1,n])/ds_errsa[1,n])**2
        delta_b[t] = d_b
    #print(delta_b)
    #下面用bestr和bestb记录最后比较结果
    #下面求取最小值及索引号
    bb = delta_b.tolist()
    xb = bb.index(min(delta_b))
    deltab = min(delta_b)  
    #delta_b = deltab
    bestfb = d_exb[xb]    
    dexb = bestfb
    best_mb = mb*10**dexb
    #对比修正
    h = 0.673
    dexb11 = bestfb+np.log10(mb*h)
    #作图对比最佳情况
    plt.figure()
    fig_fff1(rp,dssimb,xb)
    plt.title('Blue')
    plt.show()
    print('corr_mb =',10**dexb11)
    print('co_dexb=',dexb11)
    print('x^2=',deltab)
    print('mb=',best_mb)
    print('dexb=',bestfb)
    return delta_b,rp,ldb
#c_fit_datab(mr=True,mb=True,d_exr=True,d_exb=True)
#c_fit_datab(x0b=True)
#下面做图比较最佳预测值与观测的对比情况
def fig_(T):
    rp,mr,mb,d_exr,d_exb,b,a_r,a_b = va_da(p=True)
    delta_r,ldr,rp = c_fit_datar(x0r=True)
    delta_b,ldb,rp = c_fit_datab(x0b=True)
    plt.subplot(121)
    plt.plot(d_exr,np.log10(delta_r),'r-')
    plt.title('R-galaxy')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.grid()
    plt.subplot(122)
    plt.plot(d_exb,np.log10(delta_b),'b-')
    plt.xlabel(r'$log(\frac{M_h}{M_\odot})$')
    plt.ylabel(r'$log(\chi^2)$')
    plt.title('B-galaxy')
    plt.grid()
    #plt.tight_layout(pad=2.0,w_pad=2.0,h_pad=2.0)
    plt.tight_layout()
    #plt.savefig('x^2.png',dpi=600)
    plt.show()
    print('consist=',min(delta_r)/min(delta_b))   
fig_(T=True)
