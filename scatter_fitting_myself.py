###这个脚本写入散点图曲线拟合，暂时考虑能够对mock的数据进行拟合即可
#section1:先尝试小样本拟合
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
##件的路径具体到文件夹
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats as st
import matplotlib.gridspec as gridspec
import pandas as pa
from mock_data_reshow import read_mock_hmf
from Mock_SceondFigure import devi_data
def mock_data(tt):
    mockfile = 'D:/Python1/pydocument/seniorproject_quenching2/practice/iHODcatalog_bolshoi.h5'
    Mh_arr, dndlnMh_arr, use_data = read_mock_hmf(mockfile, mmin=1.e9, mmax=1.e16, nmbin=101, h=0.701, rcube=250.0)
    return use_data
#mock_data(tt=True)
def show_data(vv):
    use_data = mock_data(tt=True)
    N = np.array(use_data['b'].shape)
    print('size_con=',N)
    ###查看数据量：529308*8~=4000000
    ###把主晕取出来,并把主晕对应的物理量也取出来
    ix = use_data['d']>0
    _halos = use_data['d']
    ##取出主晕
    main_halo = np.array(_halos[ix])
    ##下面把与主晕有关的物理量取出来
    g_color = use_data['c']
    ##取出主晕下星系颜色
    gcolor = np.array(g_color[ix])
    ###尝试读取数据的处理
    M_star = use_data['e']
    ##取出恒星质量
    mstar = np.array(M_star[ix])
    ###求出中央星系的比例
    n = np.array(main_halo.shape)
    print('size_mainhalo=',n)
    frac_ = n[0]/N[0]
    print('f_c=',frac_)
    con_c = use_data['b']
    con = np.array(con_c[ix])
####第一部分作图,尝试用grid.spec完成子图分布
####该部分说明具体子图排列，用划分坐标轴的做法
    plt.figure()
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0,:])
    ax1.hist(main_halo,100, histtype="stepfilled")
    plt.xlabel(r'$Mh[M_\odot/h]$')
    plt.title('Main halo')
    plt.yscale('log')
    #plt.show()
    ax2 = plt.subplot(gs[1:,0])
    xv = ax2.hist(gcolor,100, histtype="stepfilled")
    plt.xlabel('g-r')
    #plt.show()
    #print(xv)
    ax3 = plt.subplot(gs[1:,1])
    nu = xv[0]
    nu = (nu-np.min(nu))/(np.max(nu)-np.min(nu))
    nv = xv[1]
    ax3.plot(nv[0:-1],nu)
    #plt.show()
    ##取出参数C
    ax4 = plt.subplot(gs[1:,2])
    yv = ax4.hist(con,100, histtype="stepfilled")
    plt.xlabel('concentration')
    plt.yscale('log')
    #plt.show()
    ax5 = plt.subplot(gs[1:,3])
    mu = yv[0]
    mu = (mu-np.min(mu))/(np.max(mu)-np.min(mu))
    mv = yv[1]
    ax5.plot(mv[0:-1],mu)
    #plt.show()
    plt.tight_layout()
    #plt.savefig('Main_halo',dpi=600)
    plt.show()

####作图与拟合,尝试做出几个主要物理量的联合分布和概率函数    
###对con,gcolor两个颜色矩阵做设置
    color1 = np.array((con-np.mean(con))/(np.std(con)/np.sqrt(N)))
    ###颜色1和星系的聚集程度对应
    color2 = np.array(gcolor)
    ###颜色2和星系的红化程度对应
####简单拟合几个量的概率密度曲线
    gs1 = gridspec.GridSpec(4,4)
    plt.subplot(gs1[:,0])
    sns.set(color_codes=True)
    sns.distplot(mstar)
    #sns.kdeplot(mstar)
    plt.xlabel(r'$M_\ast$')
    #plt.show()
    plt.subplot(gs1[:,1])
    sns.set(color_codes=True)
    sns.distplot(main_halo)
    #sns.kdeplot(main_halo)
    plt.xlabel(r'$M_h$')
    #plt.show()
    plt.subplot(gs1[:,2])
    sns.set(color_codes=True)
    sns.distplot(gcolor)
    #sns.kdeplot(gcolor)
    plt.xlabel(r'$g-r$')
    #plt.show()  
    plt.subplot(gs1[:,3])
    sns.set(color_codes=True)
    sns.distplot(con)
    #sns.kdeplot(con)
    plt.xlabel(r'$con$')
    #plt.show()
    plt.title('Dstribution_sight')
    plt.tight_layout()
    #plt.savefig('Distribution_details',dpi=600)
    #plt.savefig('Distribution_kdeplot',dpi=600)
    plt.show()
####观察量量变量之间的变化关系
    ###下面绘制主晕的几个主要相关两之间的联合分布阵列
    data_list = np.array([main_halo, mstar, gcolor, con])
    #print(data_list.T)
    test_data=pa.DataFrame(data_list.T)
    test_data.columns=['logMh','logMs','g-r','con']
    #print(test_data.shape) 
    #raise
    #g = sns.pairplot(test_data)
    ###这句会运行比较慢
    g = sns.PairGrid(test_data,vars=['logMh','logMs','g-r','con'],
                     diag_sharey=False, size=2.5, aspect=1)
    g.map_diag(sns.distplot)
    #g.map_diag(sns.kdeplot)
    '''
    g.map_offdiag(sns.regplot, x_estimator=np.std, x_bins=100, x_ci='ci', ci=95,
                  scatter=True, fit_reg=True, logx=False, truncate=True)
    '''
    g.map_offdiag(plt.scatter,s=0.5, color='gray')
    #g.map_offdiag(sns.residplot)
    #g.map_upper(sns.kdeplot)
    #g.map_lower(plt.scatter,s=0.5, color='gray')
    #g.savefig('distribution',dpi=600)
    return color1, color2
#show_data(vv=True)
###下面一部分书写scatter_fitting函数,自己对数据分区间做出函数图象
def bins_dic(tt):
    use_data = mock_data(tt=True)
    N = np.array(use_data['b'].shape)
    print('size_con=',N)
    ###查看数据量：529308*8~=4000000
    ###把主晕取出来,并把主晕对应的物理量也取出来
    ix = use_data['d']>0
    _halos = use_data['d']
    ##取出主晕
    main_halo = np.array(_halos[ix])
    ##下面把与主晕有关的物理量取出来
    g_color = use_data['c']
    ##取出主晕下星系颜色
    gcolor = np.array(g_color[ix])
    ###尝试读取数据的处理
    M_star = use_data['e']
    ##取出恒星质量
    mstar = np.array(M_star[ix])
    ###求出中央星系的比例
    n = np.array(main_halo.shape)
    print('size_mainhalo=',n)
    frac_ = n[0]/N[0]
    print('f_c=',frac_)
    con_c = use_data['b']
    con = np.array(con_c[ix])
###下面为质量函数
####mh-function
    value_mh = plt.hist(main_halo,100)
    plt.show()
    f_N_mh = np.array(value_mh[0])
    x_N_mh = np.array(value_mh[1])
    print(value_mh[0].size)
    #print(value_mh[1])
    media_mh = np.zeros(value_mh[0].size,dtype=np.float)
    dn_dMh = np.zeros(value_mh[0].size,dtype=np.float)
    for k in range(1,value_mh[0].size):
            media_mh[k] = (x_N_mh[k]+x_N_mh[k-1])/2
            dn_dMh[k] = f_N_mh[k]/(x_N_mh[1]-x_N_mh[0])
    media_mh[0] = media_mh[1]
    dn_dMh[0] = f_N_mh[0]/(x_N_mh[1]-x_N_mh[0])
    plt.plot(media_mh,dn_dMh,label='dn_dlgMh')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.ylabel(r'$dn/dlgM_h$')
    plt.grid()
    plt.title('Mh_fun')
    #plt.savefig('fun_Mh',dpi=600)
    plt.show()
####mstar-function    
    value_ms = plt.hist(mstar,100)
    plt.show()
    f_N_ms = np.array(value_ms[0])
    x_N_ms = np.array(value_ms[1])
    print(value_ms[0].size)
    #print(value_ms[1])
    media_ms = np.zeros(value_ms[0].size,dtype=np.float)
    dn_dms = np.zeros(value_ms[0].size,dtype=np.float)
    for k in range(1,value_ms[0].size):
            media_ms[k] = (x_N_ms[k]+x_N_ms[k-1])/2
            dn_dms[k] = f_N_ms[k]/(x_N_ms[1]-x_N_ms[0])
    media_ms[0] = media_ms[1]
    dn_dms[0] = f_N_ms[0]/(x_N_ms[1]-x_N_ms[0])
    plt.plot(media_ms,dn_dms,label=r'$dn_dlgM_\ast$')
    plt.legend(loc=1) 
    plt.yscale('log')
    plt.xlabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.ylabel(r'$dn/dlgM_\ast$')
    plt.grid()
    plt.title('stellarmass_fun')
    #plt.savefig('fun_Ms',dpi=600)
    plt.show()  
####下面求联合质量分布函数
    vl_joint = plt.hist2d(mstar,main_halo,bins=[100,100],
              range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]]
               ,normed=False,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.show()
    print(vl_joint[0].shape)
    ##第一行表示的是各个划分区间的数目多少
    #print(vl_joint[1])
    ##第二行表示对恒星质量的划分区间
    #print(vl_joint[2])
    ##第三行表示对暗晕质量的划分区间
    N_joint = np.array(vl_joint[0])
    ms_joint = np.array(vl_joint[1])
    mh_joint = np.array(vl_joint[2])
    dN_mh_joint = np.zeros((len(N_joint[:,0]),len(N_joint[0,:])),dtype=np.float)
    dN_ms_joint = np.zeros((len(N_joint[0,:]),len(N_joint[:,0])),dtype=np.float)
    for k in range(0,len(N_joint[:,0])):
        dN_ms_joint[k,:] = N_joint[k,:]/(ms_joint[1]-ms_joint[0])
    for k in range(0,len(N_joint[0,:])):
        dN_mh_joint[:,k] = N_joint[:,k]/(mh_joint[1]-mh_joint[0])
    bin1 = np.linspace(np.min(mstar), np.max(mstar), 100)
    bin2 = np.linspace(np.min(main_halo),np.max(main_halo),100) 
    mstar_ = mstar
    mainhalo = main_halo
    mstar_, mainhalo = np.meshgrid(bin2, bin1)
    plt.pcolormesh(mstar_,mainhalo,dN_ms_joint,cmap='rainbow',vmin=1,vmax=1e5,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar(label=r'$dN(M_\ast,M_h)/dlgM_\ast$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    #plt.savefig('DN',dpi=600)
    plt.show()
    return
#bins_dic(tt=True)
def dis_pro_fun(dd):
    use_data = mock_data(tt=True)
    N = np.array(use_data['b'].shape)
    print('size_con=',N)
    ###查看数据量：529308*8~=4000000
    ###把主晕取出来,并把主晕对应的物理量也取出来
    ix = use_data['d']>0
    _halos = use_data['d']
    ##取出主晕
    main_halo = np.array(_halos[ix])
    ##下面把与主晕有关的物理量取出来
    g_color = use_data['c']
    ##取出主晕下星系颜色
    gcolor = np.array(g_color[ix])
    ###尝试读取数据的处理
    M_star = use_data['e']
    ##取出恒星质量
    mstar = np.array(M_star[ix])
    ###求出中央星系的比例
    n = np.array(main_halo.shape)
    print('size_mainhalo=',n)
    frac_ = n[0]/N[0]
    print('f_c=',frac_)
    con_c = use_data['b']
    con = np.array(con_c[ix])
####不分序列，求解P(Mh|ms)或者P(Ms|Mh)
    NN=101
    MM=11
    bin1 = np.linspace(np.min(mstar), np.max(mstar), NN)
    bin2 = np.linspace(np.min(main_halo),np.max(main_halo),MM) 
    mstar_ = mstar
    mainhalo = main_halo
    mstar_,mainhalo = np.meshgrid(bin1,bin2)
    dN = plt.hist2d(mstar,main_halo,bins=[NN,MM],
            range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.show()
    #print(dN[0])
    #print(dN[1])
    #print(dN[2])
    ms_bin = np.array(dN[1])
    mh_bin = np.array(dN[0])
    p_m = np.array(dN[0])
    print(np.sum(p_m*((np.max(mstar)-np.min(mstar))/100)*((np.max(main_halo)-np.min(main_halo))/100)))
    ####下面求沿着Mh方向的边界分布
    P_ms = np.zeros(NN,dtype=np.float)
    P_Mh = np.zeros(MM,dtype=np.float)
    for k in range(0,NN):
        P_ms[k] = np.sum(p_m[k,:]*((np.max(main_halo)-np.min(main_halo))/NN))
    #####沿着行积分得到的msred的边界概率
    for k in range(0,MM):
        P_Mh[k] = np.sum(p_m[:,k]*((np.max(mstar)-np.min(mstar))/MM))
    ####沿着列方向积分得到的是Mhred的边界概率
    print(np.sum(P_ms*((np.max(mstar)-np.min(mstar))/NN)))
    print(np.sum(P_Mh*((np.max(main_halo)-np.min(main_halo))/MM)))
    '''
    ####下面作图比较，检测求解结果
    plt.plot(P_ms)
    plt.show()
    plt.hist(mstar,100,normed=True)
    plt.show()
    '''
    ####下面求在观测恒星质量前提下，观测到相应暗晕质量的概率,P(Mh|m*)
    P_mr_Mh = np.zeros((NN,MM),dtype=np.float)
    for k in range(0,NN):
        P_mr_Mh[k,:] = p_m[k,:]/P_ms[k]
    ####下面求在给定暗晕质量前提下，观测到相应恒星质量的概率,P(m*|Mh) 
    P_Mh_mr = np.zeros((MM,NN),dtype=np.float)
    for k in range(0,MM):
        P_Mh_mr[k,:] = p_m[:,k]/P_Mh[k]
    x_ms = np.array(np.linspace(np.min(mstar),np.max(mstar),MM))
    x_Mh = np.array(np.linspace(np.min(main_halo),np.max(main_halo),NN))
    plt.plot(x_ms,P_Mh_mr)
    plt.yscale('log')
    #plt.legend(x_ms)
    plt.show()
    plt.plot(x_Mh,P_mr_Mh)
    plt.yscale('log')
    plt.legend(x_ms)
    plt.show()
    return
#dis_pro_fun(dd=True) 
def fun_fit(cc):
    co_out = devi_data(xx=True)
    msred=co_out['a']
    Mhred=co_out['b']
    gcolorred=co_out['c']
    #conred=co_out['d']
    msblue=co_out['e']
    Mhblue=co_out['f']
    gcolorblue=co_out['g']
    #conblue=co_out['h']
####下面对两个序列分别求mh_mstar的关系和拟合
    LL = 4
    WL = 21
    ###颜色矩阵的长度，颜色矩阵的宽度
###for red sequence
    #bin1 = np.linspace(np.min(msred),np.max(msred),LL)
    #bin2 = np.linspace(np.min(gcolorred),np.max(gcolorred),WL)
    bin_meansr = st.binned_statistic_2d(msred,gcolorred,gcolorred,bins=[WL,LL],statistic='mean')
    print(bin_meansr[0].shape)
    mean_color = np.array(bin_meansr[0])
    print(bin_meansr[2])
    edge_color = np.array(bin_meansr[2])
    edge_ms = np.array(bin_meansr[1])
    m_bin_width = (edge_ms[1]-edge_ms[0])
    m_bins_center = edge_ms[1:]-m_bin_width/2
    plt.plot(m_bins_center,mean_color,'g')
    plt.show()
    color_mean = np.zeros((WL,LL),dtype=np.float)
    color_std = np.zeros((WL,LL),dtype=np.float)
    for k in range(0,WL):
        ix = (msred>=edge_ms[k-1]) & (msred<edge_ms[k])
        color = np.array(gcolorred[ix])
        for t in range(0,LL):
            iy = (color>=edge_color[t-1]) & (color<edge_color[t])
            gc = np.array(color[iy])
            color_mean[k,t] = np.mean(gc)
            color_std[k,t] = np.std(gc)
        color_mean[k,0] = color_mean[k,1]
    plt.plot(msred,gcolorred,'m*',alpha=0.02)
    for k in range(0,LL):
        plt.errorbar(m_bins_center,color_mean[:,k],yerr=[color_std[:,k],color_std[:,k]],
                 fmt='r^-',linewidth=1,elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5)
    plt.plot(m_bins_center,mean_color)
    plt.show()
###for blue sequence
    #bin3 = np.linspace(np.min(msred),np.max(msred),LL)
    #bin4 = np.linspace(np.min(gcolorred),np.max(gcolorred),WL)
    bin_meansb = st.binned_statistic_2d(msblue,gcolorblue,gcolorblue,bins=[WL,LL],statistic='mean')
    print(bin_meansb[0].shape)
    b_mean_color = np.array(bin_meansb[0])
    print(bin_meansb[2])
    b_edge_color = np.array(bin_meansb[2])
    b_edge_ms = np.array(bin_meansb[1])
    b_m_bin_width = (b_edge_ms[1]-b_edge_ms[0])
    b_m_bins_center = b_edge_ms[1:]-b_m_bin_width/2
    plt.plot(b_m_bins_center,b_mean_color,'g')
    plt.show()
    b_color_mean = np.zeros((WL,LL),dtype=np.float)
    b_color_std = np.zeros((WL,LL),dtype=np.float)
    for k in range(0,WL):
        ix = (msblue>=b_edge_ms[k-1]) & (msblue<b_edge_ms[k])
        color1 = np.array(gcolorblue[ix])
        for t in range(0,LL):
            iy = (color1>=b_edge_color[t]) & (color1<b_edge_color[t+1])
            gc1 = np.array(color1[iy])
            b_color_mean[k,t] = np.mean(gc1)
            b_color_std[k,t] = np.std(gc1)
        b_color_mean[k,0] = b_color_mean[k,1]
    plt.plot(msblue,gcolorblue,'y*',alpha=0.02)
    for k in range(0,LL):
        plt.errorbar(b_m_bins_center,b_color_mean[:,k],yerr=[b_color_std[:,k],b_color_std[:,k]],
                 fmt='r^-',linewidth=1,elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5)
    plt.plot(b_m_bins_center,b_mean_color)
    plt.show()
######下面·对质量分析
    LM = 4
    WM = 21
    #质量矩阵的长度，质量矩阵的宽度
###for red sequence
    mbin_meansr = st.binned_statistic_2d(msred,Mhred,Mhred,bins=[WM,LM],statistic='mean')
    print(mbin_meansr[0].shape)
    mean_mr = np.array(mbin_meansr[0])
    print(mbin_meansr[2])
    edge_mh = np.array(mbin_meansr[2])
    edge_msr = np.array(mbin_meansr[1])
    mr_bin_width = (edge_msr[1]-edge_msr[0])
    mr_bins_center = edge_msr[1:]-mr_bin_width/2
    plt.plot(mr_bins_center,mean_mr,'g')
    plt.show()
    mr_mean = np.zeros((WM,LM),dtype=np.float)
    mr_std = np.zeros((WM,LM),dtype=np.float)
    for k in range(0,WM):
        ix = (msred>=edge_msr[k-1]) & (msred<edge_msr[k])
        mr = np.array(Mhred[ix])
        for t in range(0,LM):
            iy = (mr>=edge_mh[t-1]) & (mr<edge_mh[t])
            gm = np.array(mr[iy])
            mr_mean[k,t] = np.mean(gm)
            mr_std[k,t] = np.std(gm)
        mr_mean[k,0] = mr_mean[k,1]
    plt.plot(msred,Mhred,'m*',alpha=0.02)
    for k in range(0,LM):
        plt.errorbar(mr_bins_center,mr_mean[:,k],yerr=[mr_std[:,k],mr_std[:,k]],
                 fmt='r^-',linewidth=1,elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5)
    plt.plot(mr_bins_center,mean_mr)
    plt.show()
###for blue sequence
    mbin_meansb = st.binned_statistic_2d(msblue,Mhblue,Mhblue,bins=[WM,LM],statistic='mean')
    print(mbin_meansb[0].shape)
    b_mean_mb = np.array(mbin_meansb[0])
    print(mbin_meansb[2])
    b_edge_mb = np.array(mbin_meansb[2])
    b_edge_msb = np.array(mbin_meansb[1])
    mb_bin_width = (b_edge_msb[1]-b_edge_msb[0])
    mb_bins_center = b_edge_msb[1:]-mb_bin_width/2
    plt.plot(mb_bins_center,b_mean_mb,'g')
    plt.show()
    mb_mean = np.zeros((WM,LM),dtype=np.float)
    mb_std = np.zeros((WM,LM),dtype=np.float)
    for k in range(0,WM):
        ix = (msblue>=b_edge_msb[k-1]) & (msblue<b_edge_msb[k])
        mb = np.array(Mhblue[ix])
        for t in range(0,LM):
            iy = (mb>=b_edge_mb[t-1]) & (mb<b_edge_mb[t])
            gm1 = np.array(mb[iy])
            mb_mean[k,t] = np.mean(gm1)
            mb_std[k,t] = np.std(gm1)
        mb_mean[k,0] = mb_mean[k,1]
    plt.plot(msblue,Mhblue,'y*',alpha=0.02)
    for k in range(0,LL):
        plt.errorbar(mb_bins_center,mb_mean[:,k],yerr=[mb_std[:,k],mb_std[:,k]],
                 fmt='r^-',linewidth=1,elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5)
    plt.plot(mb_bins_center,b_mean_mb)
    plt.show()
    return
#fun_fit(cc=True)
def fun_control(c):
    #show_data(vv=True)
    #bins_dic(tt=True)
    #dis_pro_fun(dd=True) 
    fun_fit(cc=True)
    return
fun_control(c=True)