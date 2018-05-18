# 02 May 2017 00:48:45
import sys
#sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice/_pycache_/session_1')
sys.path.insert(0,'D:/Python1/pydocument/seniorproject_quenching2/practice')
##导入自己编写的脚本，需要加入这两句，一句声明符号应用，然后声明需要引入文-
##件的路径具体到文件夹
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats as st
#from mpl_toolkits.mplot3d import Axes3D as A3D
##需要3d作图时,导入该模块部分
from mock_data_reshow import read_mock_hmf
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
####第一部分作图,尝试用grid.spec完成子图分布
####该部分说明具体子图排列，用划分坐标轴的做法
    import matplotlib.gridspec as gridspec
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
    con_c = use_data['b']
    con = np.array(con_c[ix])
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
    ####预先做一个关于SFR截断断和g-r的截断的计算和分布（阶段是指过了该值星系为red），最后需要得到一致的结果
    SFR = mstar-0.36*mstar-6.4
    ###所有量实在对数坐标系下的关系
    g_r = 0.8*(mstar/10.5)**0.6
    sfr = -0.35*(mstar-10.0)-10.23+mstar
    plt.hist2d(mstar,gcolor,bins=[100,100],
               range=[[np.min(mstar),np.max(mstar)],[np.min(gcolor),np.max(gcolor)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.plot(mstar,g_r,'k-')
    plt.show()
    plt.hist2d(SFR,gcolor,bins=[100,100],
               range=[[np.min(SFR),np.max(SFR)],[np.min(gcolor),np.max(gcolor)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e2,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.plot(SFR,sfr,'k--')
    plt.plot(SFR,g_r,'k-')
    plt.ylabel(r'$g-r$')
    plt.xlabel(r'$logSFR$')
    #plt.savefig('Hist2d-SFR',dpi=600)
    plt.show()
    return main_halo, mstar, gcolor, con, color1, color2
#show_data(vv=True)
#####################
def static_data(uu):
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
    
    import seaborn as sns
    sns.set(style='white')
   # with sns.axes_style("dark"):
   #     sns.jointplot(mstar, main_halo, kind="hex")
    g1 = sns.jointplot(mstar, main_halo, kind="reg", stat_func=None, space=0, color='g')
    g1.set_axis_labels(r'$logM_\ast[M_\odot/h^2]$', r'$logM_h[M_\odot/h]$')  
    #plt.savefig('Ms-Mh-joint',dpi=600)
    plt.show()

    sns.set(style='white')
    g2 = sns.jointplot(mstar, gcolor,kind="reg", stat_func=None, space=0, color='g')
    g2.set_axis_labels(r'$logM_\ast[M_\odot/h^2]$', r'$<g-r>$')  
     #plt.savefig('Ms-g-r-joint',dpi=600)
    plt.show()
    
    sns.set(style='white')
    g4 = sns.jointplot(main_halo, gcolor, kind="reg", stat_func=None, space=0, color='g')
    g4.set_axis_labels(r'$M_h[M_\odot/h]$',r'$<g-r>$') 
    #plt.savefig('Mh-g-r-joint',dpi=600)
    plt.show() 
    
    sns.set(style='white')
    g3 = sns.jointplot(mstar, con, kind="reg", stat_func=None, space=0, color='g')
    g3.set_axis_labels(r'$M_\ast[M_\odot/h^2]$', r'$con$') 
    plt.yscale('log')
    #plt.savefig('Ms-con-joint',dpi=600)
    plt.show()
   
    sns.set(style='white')
    g5 = sns.jointplot(main_halo, con, kind="reg", stat_func=None, space=0, color='g')
    g5.set_axis_labels(r'$M_h[M_\odot/h]$',r'$con$') 
    plt.yscale('log')
    #plt.savefig('Mh-con-joint',dpi=600)
    plt.show() 
#static_data(uu=True)   
#####################
def dis_o_data(kk): 
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
    plt.hist(con,bins=100,normed=True,histtype='stepfilled')
    plt.show()
####尝试用别的统计方式：hexbin
    vll1 = plt.hexbin(mstar,main_halo,gridsize=100,bins=100,xscale='linear',mincnt=0,
               yscale='linear',cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=0.5,
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.show()
    L1 = mpl.collections.PolyCollection.get_array(vll1)
    #print(L1)
    #print(L1.size)
    plt.hist(L1,100)
    plt.yscale('log')
    plt.show()
    print(np.sum(L1))
    
    vll2 = plt.hexbin(mstar,gcolor,gridsize=100,bins=100,xscale='linear',mincnt=0,
               yscale='linear',cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=0.5,
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.show()
    L2 = mpl.collections.PolyCollection.get_array(vll2)
    #print(L2)
    #print(L2.size)
    plt.hist(L2,100)
    plt.yscale('log')
    plt.show()
    print(np.sum(L2))
    
    vll3 = plt.hexbin(main_halo,gcolor,gridsize=100,bins=100,xscale='linear',mincnt=0,
               yscale='linear',cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=0.5,
               norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.show()
    L3 = mpl.collections.PolyCollection.get_array(vll3)
    #print(L3)
    #print(L3.size)
    plt.hist(L3,100)
    plt.yscale('log')
    plt.show()
    print(np.sum(L3))   
####hexbin的返回的计数对所有bins求和与划分的总数不一致（与给定样本的总数不一致，并且类型调用不方便）
    #raise
    ###添加raise表示在这里中断以调整代码或者debug
    return
#dis_o_data(kk=True)
##################
def devi_data(xx):
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
####下面把暗晕分为red和blue两个序列
    g_r = 0.8*(mstar/10.5)**0.6
    ##依照恒星质量带入计算，在g_r之上的为red,blue.
    Mh_red = np.zeros(len(main_halo),dtype=np.float)
    ms_red = np.zeros(len(main_halo),dtype=np.float)
    con_red = np.zeros(len(main_halo),dtype=np.float) 
    gcolor_red = np.zeros(len(main_halo),dtype=np.float) 
    
    Mh_blue = np.zeros(len(main_halo),dtype=np.float)
    ms_blue = np.zeros(len(main_halo),dtype=np.float) 
    con_blue = np.zeros(len(main_halo),dtype=np.float) 
    gcolor_blue = np.zeros(len(main_halo),dtype=np.float) 
    
    for k in range(0,len(main_halo)):
        if gcolor[k]<=g_r[k]:
           Mh_blue[k] = main_halo[k]
           ms_blue[k] = mstar[k]
           con_blue[k] = con[k]
           gcolor_blue[k] = gcolor[k]
        else:
           Mh_red[k] = main_halo[k]
           ms_red[k] = mstar[k]
           con_red[k] = con[k]
           gcolor_red[k] = gcolor[k]
    ix = Mh_blue!=0
    Mhblue = Mh_blue[ix]
    msblue = ms_blue[ix]
    conblue = con_blue[ix]
    gcolorblue = gcolor_blue[ix]    
    plt.hist(conblue,bins=100,normed=True,histtype='stepfilled')
    plt.show()    
    iy = Mh_red!=0
    Mhred = Mh_red[iy]
    msred = ms_red[iy]
    conred = con_red[iy]
    gcolorred = gcolor_red[iy]
    plt.hist(conred,bins=100,normed=True,histtype='stepfilled')
    plt.show()
    co_out={'a':msred, 'b':Mhred, 'c':gcolorred, 'd':conred,
            'e':msblue, 'f':Mhblue, 'g':gcolorblue, 'h':conblue
            }
    return co_out
#devi_data(xx=True)
#####################
def dist_data(tt):
    co_out = devi_data(xx=True)
    msred=co_out['a']
    Mhred=co_out['b']
    gcolorred=co_out['c']
    #conred=co_out['d']
    msblue=co_out['e']
    Mhblue=co_out['f']
    gcolorblue=co_out['g']
    #conblue=co_out['h']
####针对两个序列二维统计分布图   
    value1r = plt.hist2d(msred,gcolorred,bins=[100,100],
               range=[[np.min(msred),np.max(msred)],[np.min(gcolorred),np.max(gcolorred)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    value1r[0][np.isnan(value1r[0])] = 0
    ###norm.color.LogNorm表示在数密度计数上用对数计数
    plt.colorbar()
    plt.xlabel(r'$logM_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$g-r$')
    #plt.savefig('Hist2d-1r',dpi=600)
    plt.show()
    
    value2r = plt.hist2d(msred,Mhred,bins=[100,100],
               range=[[np.min(msred),np.max(msred)],[np.min(Mhred),np.max(Mhred)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    value2r[0][np.isnan(value2r[0])] = 0
    plt.colorbar()
    plt.xlabel(r'$logM_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$logMh[M_\odot/h]$')
    #plt.savefig('Hist2d-2r',dpi=600)
    plt.show()

    value3r = plt.hist2d(Mhred,gcolorred,bins=[100,100],
               range=[[np.min(Mhred),np.max(Mhred)],[np.min(gcolorred),np.max(gcolorred)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    value3r[0][np.isnan(value3r[0])] = 0
    ###这句表示对value3里面nan的数全部都以0来替换
    plt.colorbar()
    plt.xlabel(r'$logMh[M_\odot/h]$')
    plt.ylabel(r'$g-r$')
    #plt.savefig('Hist2d-3r',dpi=600)
    plt.show() 
    ####上面三个hist2d的部分，实际上通过binned的函数来计数每个bins里面的星系出数目，根据需要可以改变统计量
    ####上述三个量：vlth1,vlth2,vlth3(stats.bined的函数调用返回值)里面的第一分两均为100，100的数组，
    ####反映了hist2d里面的x，y,量对应的携同分布
    ###hist2d能给出数密度分布，但是没有反馈值    
    value1b = plt.hist2d(msblue,gcolorblue,bins=[100,100],
               range=[[np.min(msblue),np.max(msblue)],[np.min(gcolorblue),np.max(gcolorblue)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    value1b[0][np.isnan(value1b[0])] = 0
    ###norm.color.LogNorm表示在数密度计数上用对数计数
    plt.colorbar()
    plt.xlabel(r'$logM_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$g-r$')
    #plt.savefig('Hist2d-1b',dpi=600)
    plt.show()
    
    value2b = plt.hist2d(msblue,Mhblue,bins=[100,100],
               range=[[np.min(msblue),np.max(msblue)],[np.min(Mhblue),np.max(Mhblue)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    value2b[0][np.isnan(value2b[0])] = 0
    plt.colorbar()
    plt.xlabel(r'$logM_\ast[M_\odot/h^2]$')
    plt.ylabel(r'$logMh[M_\odot/h]$')
    #plt.savefig('Hist2d-2b',dpi=600)
    plt.show()

    value3b = plt.hist2d(Mhblue,gcolorblue,bins=[100,100],
               range=[[np.min(Mhblue),np.max(Mhblue)],[np.min(gcolorblue),np.max(gcolorblue)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    value3b[0][np.isnan(value3b[0])] = 0
    ###这句表示对value3里面nan的数全部都以0来替换
    plt.colorbar()
    plt.xlabel(r'$logMh[M_\odot/h]$')
    plt.ylabel(r'$g-r$')
    #plt.savefig('Hist2d-3b',dpi=600)
    plt.show() 
####对两个序列的颜色分析
###section1:对颜色部分分析
###for red sequence
    bin_meansr, bin_edgesr, binnumberr = st.binned_statistic(msred,gcolorred,bins=21,statistic='mean',)
    std_red = np.zeros(len(bin_meansr),dtype=np.float)
    sre_red = np.zeros(len(bin_meansr),dtype=np.float)
    mean_red = np.zeros(len(bin_meansr),dtype=np.float)
    bin_widthr = (bin_edgesr[1]-bin_edgesr[0])
    bins_centerr = bin_edgesr[1:]-bin_widthr/2
    for k in range(0,len(bin_meansr)):
        ix = (bin_edgesr[k-1]<=msred) & (msred>bin_edgesr[k])
        gc1 = gcolorred[ix]
        n = gc1.size
        mean_red[k] = np.mean(gc1)
        std_red[k] = np.std(gc1)
        sre_red[k] = std_red[k]/np.sqrt(n)
    mean_red[0] = bin_meansr[0]
    gs1 = mpl.gridspec.GridSpec(2,2)
    bx1 = plt.subplot(gs1[0,0])
    bx1.hist(gcolorred,bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='color_red')
    plt.legend(loc=1)
    #plt.show()
    bx2 = plt.subplot(gs1[1,0])
    bx2.plot(msred,gcolorred,'m*',alpha=0.02, label='*-data_red')
    plt.hlines(bin_meansr, bin_edgesr[:-1], bin_edgesr[1:], color='r', lw=2, label='statics_color_red')
    plt.errorbar(bins_centerr,mean_red,yerr=[std_red,std_red],fmt='r^-',linewidth=1,
                 elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='color_err_red')
    plt.legend(loc=2)
    #plt.show()
###for blue sequence
    bin_meansb, bin_edgesb, binnumberb = st.binned_statistic(msblue,gcolorblue,bins=21,statistic='mean',)
    std_blue = np.zeros(len(bin_meansb),dtype=np.float)
    sre_blue = np.zeros(len(bin_meansb),dtype=np.float)
    mean_blue = np.zeros(len(bin_meansb),dtype=np.float)
    bin_widthb = (bin_edgesb[1]-bin_edgesb[0])
    bins_centerb = bin_edgesb[1:]-bin_widthb/2
    for k in range(0,len(bin_meansb)):
        iy = (bin_edgesb[k-1]<=msblue) & (msblue>bin_edgesb[k])
        gc2 = gcolorblue[iy]
        n = gc2.size
        mean_blue[k] = np.mean(gc2)
        std_blue[k] = np.std(gc2)
        sre_blue[k] = std_blue[k]/np.sqrt(n)
    mean_blue[0] = bin_meansb[0]
    bx3 = plt.subplot(gs1[0,1])
    bx3.hist(gcolorblue,bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='color_blue')
    plt.legend(loc=2)
    #plt.show()
    bx4 = plt.subplot(gs1[1,1])
    bx4.plot(msblue,gcolorblue,'c*',alpha=0.02, label='*-data_blue')
    plt.hlines(bin_meansb, bin_edgesb[:-1], bin_edgesb[1:], color='b', lw=2, label='statics_color_blue')
    plt.errorbar(bins_centerb,mean_blue,yerr=[std_blue,std_blue],fmt='bo-',linewidth=1,
                 elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='color_err_blue')
    plt.legend(loc=4)
    #plt.show()
    plt.tight_layout()
    #plt.savefig('color_analysis',dpi=600)
    plt.show()
    
    plt.figure()
    #plt.hlines(bin_meansr, bin_edgesr[:-1], bin_edgesr[1:], color='r', lw=2, label='statics_color_red')
    plt.errorbar(bins_centerr,mean_red,yerr=[std_red,std_red],fmt='r^-',linewidth=1,
                 elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='color_err_red')
    #plt.hlines(bin_meansb, bin_edgesb[:-1], bin_edgesb[1:], color='b', lw=2, label='statics_color_blue')
    plt.errorbar(bins_centerb,mean_blue,yerr=[std_blue,std_blue],fmt='bo-',linewidth=1,
                 elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='color_err_blue')
    plt.legend(loc=2)
    #plt.savefig('color_comparation',dpi=600)
    plt.show()
###section2:对质量部分分析
###for red sequence
    m_bin_meansr, m_bin_edgesr, m_binnumberr = st.binned_statistic(msred,Mhred,bins=8,statistic='mean',)
    m_std_red = np.zeros(len(m_bin_meansr),dtype=np.float)
    m_sre_red = np.zeros(len(m_bin_meansr),dtype=np.float)
    m_mean_red = np.zeros(len(m_bin_meansr),dtype=np.float)
    m_bin_widthr = (m_bin_edgesr[1]-m_bin_edgesr[0])
    m_bins_centerr = m_bin_edgesr[1:]-m_bin_widthr/2
    for k in range(0,len(m_bin_meansr)):
        ix = (m_bin_edgesr[k-1]<=msred) & (msred>m_bin_edgesr[k])
        gc1 = Mhred[ix]
        n = gc1.size
        m_mean_red[k] = np.mean(gc1)
        m_std_red[k] = np.std(gc1)
        m_sre_red[k] = m_std_red[k]/np.sqrt(n)
    #m_mean_red[0] = m_bin_meansr[0]
    m_mean_red[0] = m_mean_red[1]
    gs2 = mpl.gridspec.GridSpec(2,2)
    cx1 = plt.subplot(gs2[0,0])
    cx1.hist(Mhred,bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='Mh_red')
    plt.legend(loc=1)
    #plt.show()
    cx2 = plt.subplot(gs1[1,0])
    cx2.plot(msred,Mhred,'m*',alpha=0.02, label='*-data_red')
    plt.hlines(m_bin_meansr, m_bin_edgesr[:-1], m_bin_edgesr[1:], color='r', lw=2, label='statics_Mh_red')
    plt.errorbar(m_bins_centerr,m_mean_red,yerr=[m_std_red,m_std_red],fmt='r^-',linewidth=1,
                 elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='Mh_err_red')
    plt.legend(loc=2)
    #plt.show()
###for blue sequence
    m_bin_meansb, m_bin_edgesb, m_binnumberb = st.binned_statistic(msblue,Mhblue,bins=8,statistic='mean',)
    m_std_blue = np.zeros(len(m_bin_meansb),dtype=np.float)
    m_sre_blue = np.zeros(len(m_bin_meansb),dtype=np.float)
    m_mean_blue = np.zeros(len(m_bin_meansb),dtype=np.float)
    m_bin_widthb = (m_bin_edgesb[1]-m_bin_edgesb[0])
    m_bins_centerb = m_bin_edgesb[1:]-m_bin_widthb/2
    for k in range(0,len(m_bin_meansb)):
        iy = (m_bin_edgesb[k-1]<=msblue) & (msblue>m_bin_edgesb[k])
        gc2 = Mhblue[iy]
        n = gc2.size
        m_mean_blue[k] = np.mean(gc2)
        m_std_blue[k] = np.std(gc2)
        m_sre_blue[k] = m_std_blue[k]/np.sqrt(n)
    #m_mean_blue[0] = m_bin_meansb[0]
    m_mean_blue[0] = m_mean_blue[1]
    cx3 = plt.subplot(gs2[0,1])
    cx3.hist(Mhblue,bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='Mh_blue')
    plt.legend(loc=1)
    #plt.show()
    cx4 = plt.subplot(gs2[1,1])
    cx4.plot(msblue,Mhblue,'c*',alpha=0.02, label='*-data_blue')
    plt.hlines(m_bin_meansb, m_bin_edgesb[:-1], m_bin_edgesb[1:], color='b', lw=2, label='statics_Mh_blue')
    plt.errorbar(m_bins_centerb,m_mean_blue,yerr=[m_std_blue,m_std_blue],fmt='bo-',linewidth=1,
                 elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='Mh_err_blue')
    plt.legend(loc=2)
    #plt.show()
    plt.tight_layout()
    #plt.savefig('Mh_analysis',dpi=600)
    plt.show()
    ####注意需要把恒星质量单位转化
    h = 0.72
    delta_bar = np.log10(h)
    plt.figure()
    #plt.hlines(m_bin_meansr, m_bin_edgesr[:-1], m_bin_edgesr[1:], color='r', lw=2, label='statics_Mh_red')
    plt.errorbar(m_bins_centerr-2*delta_bar,m_mean_red,yerr=[m_std_red,m_std_red],fmt='r^-',linewidth=1,
                 elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='Mh_err_red')
    #plt.hlines(m_bin_meansb, m_bin_edgesb[:-1], m_bin_edgesb[1:], color='b', lw=2, label='statics_Mh_blue')
    plt.errorbar(m_bins_centerb-2*delta_bar,m_mean_blue,yerr=[m_std_blue,m_std_blue],fmt='b^-',linewidth=1,
                 elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='Mh_err_blue')
    ####M16数据对比
    data_r = np.array([12.17,12.14,12.50,12.89,13.25,13.63,14.05])
    data_r_err = np.array([[0.19,0.12,0.04,0.04,0.03,0.03,0.05],
                          [-0.24,-0.14,-0.05,-0.04,-0.03,-0.03,-0.05]])
    m_binr = np.array([10.28,10.58,10.86,11.10,11.29,11.48,11.68])
    data_b = np.array([11.80,11.73,12.15,12.61,12.69,12.79,12.79])
    data_b_err = np.array([[0.16,0.13,0.08,0.10,0.19,0.43,0.58],
                          [-0.20,-0.17,-0.10,-0.11,-0.25,-1.01,-2.23]])
    m_binb = np.array([10.24,10.56,10.85,11.10,11.28,11.47,11.68])
    line2,caps2,bars2=plt.errorbar(m_binr,data_r,yerr=abs(data_r_err)[::-1],fmt="ro--",linewidth=1,
                                elinewidth=0.5,ecolor='r',capsize=1,capthick=0.5,label='red(M16)')
    line4,caps4,bars3=plt.errorbar(m_binb,data_b,yerr=abs(data_b_err)[::-1],fmt="bo--",linewidth=1,
                                elinewidth=0.5,ecolor='b',capsize=1,capthick=0.5,label='blue(M16)')
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$log{\langle M_{200} \rangle/[M_\odot h^{-1}]}$')  
    plt.legend(loc=2)
    #plt.savefig('Mh_comparation',dpi=600)
    plt.show()
    #print('mh_r=',m_mean_red)
    #print('mh_b=',m_mean_blue)
    return  
#dist_data(tt=True)
###################
##下面的hist_proba函数尝试从二位的binned_statistic函数调用，得到联合分布
def hist_proba(rr):
    co_out = devi_data(xx=True)
    msred=co_out['a']
    Mhred=co_out['b']
    #gcolorred=co_out['c']
    #conred=co_out['d']
    msblue=co_out['e']
    Mhblue=co_out['f']
    #gcolorblue=co_out['g']
    #conblue=co_out['h']
###for red sequence
    NN = 101
    MM = 101
    rvalue = plt.hist2d(msred,Mhred,bins=[NN,MM],
               range=[[np.min(msred),np.max(msred)],[np.min(Mhred),np.max(Mhred)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.show()
    #print(bvalue[0].shape)
    print(np.sum(rvalue[0]*((np.max(Mhred)-np.min(Mhred))/MM)*((np.max(msred)-np.min(msred))/NN)))
    ####把value的值给P_Mh_ms——联合概率分布
    r_P = np.array(rvalue[0])
    ####下面求沿着Mh方向的边界分布
    P_ms_r = np.zeros(NN,dtype=np.float)
    P_Mh_r = np.zeros(MM,dtype=np.float)
    for k in range(0,NN):
        P_ms_r[k] = np.sum(r_P[k,:]*((np.max(Mhred)-np.min(Mhred))/NN))
    #####沿着行积分得到的msred的边界概率
    for k in range(0,MM):
        P_Mh_r[k] = np.sum(r_P[:,k]*((np.max(msred)-np.min(msred))/MM))
    ####沿着列方向积分得到的是Mhred的边界概率
    print(np.sum(P_ms_r*((np.max(msred)-np.min(msred))/NN)))
    print(np.sum(P_Mh_r*((np.max(Mhred)-np.min(Mhred))/MM)))
    ####下面作图比较，检测求解结果
    plt.plot(P_ms_r)
    plt.show()
    plt.hist(msred,100,normed=True)
    plt.show()
    ####下面求在观测恒星质量前提下，观测到相应暗晕质量的概率,P(Mh|m*)
    rP_mr_Mh = np.zeros((NN,MM),dtype=np.float)
    for k in range(0,NN):
        rP_mr_Mh[k,:] = r_P[k,:]/P_ms_r[k]
    ####下面求在给定暗晕质量前提下，观测到相应恒星质量的概率,P(m*|Mh) 
    rP_Mh_mr = np.zeros((MM,NN),dtype=np.float)
    for k in range(0,MM):
        rP_Mh_mr[k,:] = r_P[:,k]/P_Mh_r[k]
    x_ms_r = np.array(np.linspace(np.min(msred),np.max(msred),MM))
    x_Mh_r = np.array(np.linspace(np.min(Mhred),np.max(Mhred),NN))
    #print(x_ms_r)
    #print(x_Mh_r)
    ####每一列对应一个Mh区间的mstar的情况
    '''
    for k in range(0,5):
        plt.plot(x_ms_r,rP_Mh_mr[:,22+k])
        #plt.xscale('log')
        plt.yscale('log')
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$P(M_\ast|M_h)$')
    plt.title('Red')
    #plt.savefig('P_ms_red',dpi=600)
    plt.show() 
    ####每一行对应一个mstar区间的Mh的情况
    for k in range(0,5):
        plt.plot(x_Mh_r,rP_mr_Mh[22+k,:])
        #plt.xscale('log')
        plt.yscale('log')
    plt.title('Red')
    plt.xlabel(r'$log[M_h/M_\odot]$')
    plt.ylabel(r'$P(M_h|M_\ast)$')
    #plt.savefig('P_mh_red',dpi=600)
    plt.show()
    '''
    plt.plot(x_ms_r,rP_Mh_mr)
    plt.yscale('log')
    plt.show()
    plt.plot(x_Mh_r,rP_mr_Mh)
    plt.yscale('log')
    plt.show()
###for blue sequence
    bvalue = plt.hist2d(msblue,Mhblue,bins=[NN,MM],
               range=[[np.min(msblue),np.max(msblue)],[np.min(Mhblue),np.max(Mhblue)]]
               ,normed=True,cmin=0,cmap='rainbow',vmin=1e-5,vmax=1e3,alpha=1, 
               norm=mpl.colors.LogNorm())
    plt.show()
    #print(value[0].shape)
    print(np.sum(bvalue[0]*((np.max(Mhblue)-np.min(Mhblue))/MM)*((np.max(msblue)-np.min(msblue))/NN)))
    ####把value的值给P_Mh_ms——联合概率分布
    b_P = np.array(bvalue[0])
    ####下面求沿着Mh方向的边界分布
    P_ms_b = np.zeros(NN,dtype=np.float)
    P_Mh_b = np.zeros(MM,dtype=np.float)
    for k in range(0,NN):
        P_ms_b[k] = np.sum(b_P[k,:]*((np.max(Mhblue)-np.min(Mhblue))/NN))
    #####沿着行积分得到的msred的边界概率
    for k in range(0,MM):
        P_Mh_b[k] = np.sum(b_P[:,k]*((np.max(msblue)-np.min(msblue))/MM))
    ####沿着列方向积分得到的是Mhred的边界概率
    print(np.sum(P_ms_b*((np.max(msblue)-np.min(msblue))/NN)))
    print(np.sum(P_Mh_b*((np.max(Mhblue)-np.min(Mhblue))/MM)))
    ####下面作图比较，检测求解结果
    plt.plot(P_ms_b)
    plt.show()
    plt.hist(msblue,100,normed=True)
    plt.show()
    ####下面求在观测恒星质量前提下，观测到相应暗晕质量的概率,P(Mh|m*)
    bP_mr_Mh = np.zeros((NN,MM),dtype=np.float)
    for k in range(0,NN):
        bP_mr_Mh[k,:] = b_P[k,:]/P_ms_b[k]
    ####下面求在给定暗晕质量前提下，观测到相应恒星质量的概率,P(m*|Mh) 
    bP_Mh_mr = np.zeros((MM,NN),dtype=np.float)
    for k in range(0,MM):
        bP_Mh_mr[k,:] = b_P[:,k]/P_Mh_b[k]
    x_ms_b = np.array(np.linspace(np.min(msblue),np.max(msblue),MM))
    x_Mh_b = np.array(np.linspace(np.min(Mhblue),np.max(Mhblue),NN))
    #print(x_ms_b)
    #print(x_Mh_b)
    ####每一列对应一个Mh区间的mstar的情况
    '''
    for k in range(0,5):
        plt.plot(x_ms_b,bP_Mh_mr[:,22+k])
        #plt.xscale('log')
        plt.yscale('log')
    plt.xlabel(r'$log[M_\ast/M_\odot]$')
    plt.ylabel(r'$P(M_\ast|M_h)$')
    plt.title('Blue')
    #plt.savefig('P_ms_blue',dpi=600)
    plt.show() 
    ####每一行对应一个mstar区间的Mh的情况
    for k in range(0,5):
        plt.plot(x_Mh_b,bP_mr_Mh[17+k,:])
        #plt.xscale('log')
        plt.yscale('log')
    plt.title('Blue')
    plt.xlabel(r'$log[M_h/M_\odot]$')
    plt.ylabel(r'$P(M_h|M_\ast)$')
    #plt.savefig('P_mh_blue',dpi=600)
    plt.show()
    '''
    plt.plot(x_ms_b,bP_Mh_mr)
    plt.yscale('log')
    plt.show()
    plt.plot(x_Mh_b,bP_mr_Mh)
    plt.yscale('log')
    plt.show()
####下面部分求red星系的占比系数f_red
###for red sequences
    rf_red_mstar = 1-np.exp(-(msred/10.55)**0.66)
    rf_red_Mh = 1-np.exp(-(Mhred/12.25)**0.42)    
    plt.plot(msred,rf_red_mstar,'r*')
    plt.plot(Mhred,rf_red_Mh,'g*')
    plt.show()
###for blue sequence
    bf_red_mstar = 1-np.exp(-(msblue/10.55)**0.66)
    bf_red_Mh = 1-np.exp(-(Mhblue/12.25)**0.42)
    plt.plot(msblue,bf_red_mstar,'r*')
    plt.plot(Mhblue,bf_red_Mh,'g*')
    plt.show()
    return
#hist_proba(rr=True)
##################
def fred_dist(L):
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
    g_r = 0.8*(mstar/10.5)**0.6
    #print(g_r.shape)
    #print(gcolor.shape)
    fred = np.zeros(len(gcolor),dtype=np.float)
    for k in range(0,len(gcolor)):
        if gcolor[k]>=g_r[k]:
           fred[k] = 1
        else:
           fred[k]  = 0       
    ####下面说明怎么对binned_statics的划分区间赋值
    bin1 = np.linspace(np.min(mstar), np.max(mstar), 100)
    bin2 = np.linspace(np.min(main_halo),np.max(main_halo),100)
    value1 = st.binned_statistic_2d(mstar,main_halo,fred,statistic='mean',bins=[bin1,bin2],
           range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]])
    #value[0][np.isnan(value[0])] = 0
    f_red = np.array(value1[0])
    #print(value[0])
    mstar_ = mstar
    mainhalo = main_halo
    mstar_, mainhalo = np.meshgrid(bin1, bin2)
    plt.pcolormesh(mainhalo,mstar_,f_red.T,cmap='rainbow',vmin=0.1,vmax=0.9,
                   edgecolors=None,alpha=1)
    plt.colorbar(label='f_red')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    #plt.savefig('f_red',dpi=600)
    plt.show()
    value2 = st.binned_statistic_2d(mstar,main_halo,gcolor,statistic='mean',bins=[bin1,bin2],
           range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]])
    mean_color = np.array(value2[0])
    plt.pcolormesh(mainhalo,mstar_,mean_color.T,cmap='rainbow',vmin=0.1,vmax=1.2,
                   edgecolors=None,alpha=1)
    plt.colorbar(label='g-r')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    #plt.savefig('g-r_dist',dpi=600)
    plt.show()
    value3 = st.binned_statistic_2d(mstar,main_halo,con,statistic='mean',bins=[bin1,bin2],
           range=[[np.min(mstar),np.max(mstar)],[np.min(main_halo),np.max(main_halo)]])
    mean_con = np.array(value3[0])
    plt.pcolormesh(mainhalo,mstar_,mean_con.T,cmap='rainbow',vmin=1,vmax=40,
                   edgecolors=None,alpha=1,norm=mpl.colors.LogNorm())
    plt.colorbar(label='con')
    plt.xlabel(r'$lgM_h[M_\odot h^{-1}]$')
    plt.ylabel(r'$lgM_\ast[M_\odot h^{-2}]$')
    #plt.savefig('con_dist',dpi=600)
    plt.show()
    return
#fred_dist(L=True)
def run_control(T):
    #show_data(vv=True)
    #dis_o_data(kk=True)
    #static_data(uu=True)
    ###第三个函数时间比较久，做图显示边界分布
    #devi_data(xx=True)
    dist_data(tt=True)###对模拟数据分析拟合并和观测数据比较
    #hist_proba(rr=True)
    #fred_dist(L=True)
    return
run_control(T=True)