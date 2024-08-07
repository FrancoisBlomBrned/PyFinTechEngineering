"""
This file includes the functions needed to draw plots
"""
# 导入所需的包并修改字体
import matplotlib.pyplot as plt
from pylab import mpl

from Chapter12.options.options_greeks import delta_EurOpt
from Chapter12.options.options_models import option_BSM, SOME_GLOBAL_VAR_UTILIZED_EVERYWHERE_UNRECOMMENDED

mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 1.使用欧式看涨期权的Delta计算得到的价格与BSM计算得到的价格相对比
def plot_EurOpt_price_DeltaVersusBSM(S_0, S_list, K_1, sigma_1, r_1, T_1):
    """
    此函数用于欧式看涨期权：
    假设当天的基础资产价格变化，其他条件不变，将利用Delta计算得到的近似期权理论价格和用Black-Scholes模型计算得到的期权理论价格用图像相对比
    S_0：代表当天的基础资产价格
    S_list：代表预设的变动后的当天的基础资产价格列表
    K：代表欧式期权的行权价格
    sigma：代表欧式期权的基础资产收益率的波动率
    r：代表无风险收益率
    T：代表期权的期限（年）
    """
    # 第1步：运用自定义函数option_BSM计算布莱克-斯科尔斯-默顿模型的期权价格
    value_list = [option_BSM(S = S, K = K_1, sigma = sigma_1, r = r_1, T = T_1,
                             opt = "call") for S in S_list]  # 不同基础资产价格对应的期权价格（运用BSM模型）

    value_one = option_BSM(S = S_0, K = K_1, sigma = sigma_1, r = r_1, T = T_1,
                           opt="call")  # 基础资产价格等于S_0元/股对应的期权价格

    value_approx1 = value_one + 5 * (S_list - S_0)  # 用Delta计算不同基础资产价格对应的近似期权价格
    # delta_EurOpt1
    # 第2步：将运用BSM模型计算得到的期权价格与运用Delta计算得到的近似期权价格进行可视化
    plt.figure(figsize = (9, 6))
    plt.plot(S_list, value_list, "b-", label = u"运用BSM模型计算得到的看涨期权价格", lw = 2.5)
    plt.plot(S_list, value_approx1, "r-", label = u"运用Delta计算得到的看涨期权近似价格", lw = 2.5)
    plt.xlabel(u"股票价格", fontsize = 13)
    plt.ylabel(u"期权价格", fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title(u"运用BSM模型计算得到的期权价格与运用Delta计算得到的近似期权价格的关系图", fontsize = 13)
    plt.legend(fontsize = 13)
    plt.grid()
    plt.show()
    pass



# 2.运用Python将基础资产价格（股票价格）与欧式期权多头Delta之间的对应关系可视化
def plot_EurOptCallDelta_changing_S(S_list, K_1, sigma_1, r_1, T_1):
    """
    此函数用于欧式看涨期权：
    假设当天的基础资产价格变化，其他条件不变，观察基础资产价格与欧式期权多头Delta之间的对应关系
    S_list：代表预设的变动后的当天的基础资产价格列表
    K_1：代表欧式期权的行权价格
    sigma_1：代表欧式期权的基础资产收益率的波动率
    r_1：代表无风险收益率
    T_1：代表期权的期限（年）
    """
    Delta_EurCall = [delta_EurOpt(S = S, K = K_1, sigma = sigma_1, r = r_1, T = T_1, optype = "call",
                                  positype = "long") for S in S_list]  # 计算欧式看涨期权的Delta
    Delta_EurPut = [delta_EurOpt(S = S, K = K_1, sigma = sigma_1, r = r_1, T = T_1, optype = "put",
                                 positype = "long") for S in S_list]  # 计算欧式看跌期权的Delta

    plt.figure(figsize = (9, 6))
    plt.plot(S_list, Delta_EurCall, "b-", label = u"欧式看涨期权多头", lw = 2.5)
    plt.plot(S_list, Delta_EurPut, "r-", label = u"欧式看跌期权多头", lw = 2.5)
    plt.xlabel(u"股票价格", fontsize = 13)
    plt.ylabel(u"Delta", fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title(u"股票价格与欧式期权多头Delta", fontsize = 13)
    plt.legend(fontsize = 13)
    plt.grid()
    plt.show()
    pass

# 3.运用Python将欧式看涨期权的期权期限（年）与欧式看涨期权多头Delta之间的对应关系可视化
def plot_EurOptCallDelta_changing_T(S_InTheMoney, S_AtTheMoney, S_OutOfTheMoney, K_1, sigma_1, r_1, T_list):
    """
    此函数用于欧式看涨期权：
    假设当天的基础资产价格变化，其他条件不变，观察实值、平值和虚值欧式看涨期权的期限（年）与该欧式看涨期权多头Delta之间的对应关系
    S_InTheMoney：代表当天的实值期权对应的基础资产价格
    S_AtTheMoney：代表当天的平值期权对应的基础资产价格
    S_OutOfTheMoney：代表当天的虚值期权对应的基础资产价格
    K_1：代表欧式期权的行权价格
    sigma_1：代表欧式期权的基础资产收益率的波动率
    r_1：代表无风险收益率
    T_list：代表预设的变动后的当天的期权的期限（年）列表
    """
    Delta_list1 = delta_EurOpt(S = S_InTheMoney, K = K_1, sigma = sigma_1, r = r_1, T = T_list, optype = "call",
                               positype = "long")  # 实值看涨期权的Delta
    Delta_list2 = delta_EurOpt(S = S_AtTheMoney, K = K_1, sigma = sigma_1, r = r_1, T = T_list, optype = "call",
                               positype = "long")  # 平价看涨期权的Delta
    Delta_list3 = delta_EurOpt(S = S_OutOfTheMoney, K = K_1, sigma = sigma_1, r = r_1, T = T_list, optype = "call",
                               positype = "long")  # 虚值看涨期权的Delta

    plt.figure(figsize=(9, 6))
    plt.plot(T_list, Delta_list1, "b-", label = u"实值看涨期权多头", lw = 2.5)
    plt.plot(T_list, Delta_list2, "r-", label = u"平价看涨期权多头", lw = 2.5)
    plt.plot(T_list, Delta_list3, "g-", label = u"虚值看涨期权多头", lw = 2.5)
    plt.xlabel(u"期权期限（年）", fontsize = 13)
    plt.ylabel(u"Delta", fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title(u"期权期限与欧式看涨期权多头Delta的关系图", fontsize = 13)
    plt.legend(fontsize = 13)
    plt.grid()
    plt.show()
    pass

# 以看涨期权作为分析对象，针对不同的股价，通过公式并运用Python计算期权的近似价格并用可视化对比以下三种方法得到的结果
# 1.通过BSM模型得到的期权价格；2.仅运用Delta计算得到的期权近似价格；3.运用Delta和Gamma计算得到的期权近似价格
def plot_3ways_BSM_Delta_DeltaGamma(S_list, gamma_list):
    value_approx2 = (value_one + delta_EurOpt1 * (S_list1 - S_ABC) +
                     0.5 * gamma_Eur * pow(S_list1 - S_ABC, 2))  # 用Delta和Gamma计算近似的期权价格

    plt.figure(figsize = (9, 6))
    plt.plot(S_list, value_list, "b-", label = u"运用BSM模型计算的看涨期权价格", lw = 2.5)
    plt.plot(S_list, value_approx1, "r-", label = u"仅用Delta计算的看涨期权近似价格", lw = 2.5)
    plt.plot(S_list, value_approx2, "m-", label = u"用Delta和Gamma计算的看涨期权近似价格", lw = 2.5)
    plt.plot(S_0, value_one, "o", label = u"股价等于3.27元/股对应的期权价格", lw = 2.5)
    plt.xlabel(u"股票价格", fontsize = 13)
    plt.ylabel(u"期权价格", fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.title(u"运用BSM模型、仅用Delta以及用Delta和Gamma计算的期权价格", fontsize = 13)
    plt.legend(fontsize = 13)
    plt.grid()
    plt.show()
    pass

# def plot_theta(S_list, theta_list_call, theta_list_put):
    # ... 现有代码 ...
    pass

# def plot_vega(S_list, vega_list):
    # ... 现有代码 ...
    pass

# def plot_rho(S_list, rho_list_call, rho_list_put):
    # ... 现有代码 ...
    pass

if __name__ == '__main__':
    import numpy as np
    plot_EurOpt_price_DeltaVersusBSM(S_0 = 20, S_list = np.linspace(10, 40, 200), K_1 = 9, sigma_1 = 0.7,
                                     r_1 = 0.02, T_1 = 0.6)