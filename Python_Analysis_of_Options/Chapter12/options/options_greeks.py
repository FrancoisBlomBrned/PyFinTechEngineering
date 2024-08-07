"""
This file includes the functions needed to calculate the options' greeks
"""
# 导入需要的包和函数
import numpy as np

from Chapter12.options.options_models import American_call, American_put

"""
1.Delta
"""
# 欧式期权的Delta
def delta_EurOpt(S, K, sigma, r, T, optype, positype):
    """
    定义一个欧式期权Delta的函数
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    optype：代表期权的类型，输入optype="call"表示看涨期权，输入其他则表示看跌期权
    positype：代表期权头寸的方向，输入positype="long"表示期权多头，输入其他则表示期权空头
    """
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数
    from numpy import log, sqrt  # 从NumPy模块中导入log、sqrt函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # d1的表达式
    if optype == "call":  # 当期权是看涨期权
        if positype == "long":  # 当期权头寸是多头
            delta = norm.cdf(d1)  # 计算期权的delta
        else:  # 当期权头寸是空头
            delta = -norm.cdf(d1)
    else:  # 当期权是看跌期权
        if positype == "long":
            delta = norm.cdf(d1) - 1
        else:
            delta = 1 - norm.cdf(d1)
    return delta
    pass

# 运用Python自定义一个计算美式期权Delta的函数，并且按照看涨期权、看跌期权分别进行定义
def delta_AmerCall(S, K, sigma, r, T, N, positype):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Delta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    positype：代表期权头寸方向，输入positype="long"表示期权多头，输入其他则表示期权空头
    """
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    call_matrix = np.zeros((N + 1, N + 1))  # 创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N - N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上到下排序）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上到下排序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素从大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i - j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上到下排序）
        call_strike = np.maximum(Si - K, 0)  # 计算提前行权时的期权收益
        call_nostrike = np.exp(-r * t) * (p * call_matrix[: i + 1, i + 1] +
                                          (1 - p) * call_matrix[1: i + 2, i + 1])  # 计算不提前行权时的期权价值
        call_matrix[: i + 1, i] = np.maximum(call_strike,
                                             call_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta = (call_matrix[0, 1] - call_matrix[1, 1]) / (S * u - S * d)  # 计算期权Delta
    if positype == "long":  # 当期权头寸是多头时
        result = Delta
    else:  # 当期权头寸是空头时
        result = -Delta
    return result
    pass

def delta_AmerPut(S, K, sigma, r, T, N, positype):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Delta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    positype：代表期权头寸方向，输入positype="long"表示期权多头，输入其他则表示期权空头
    """
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    put_matrix = np.zeros((N + 1, N + 1))  # 创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N - N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    put_matrix[:, -1] = np.maximum(K - S_end, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上往下顺序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i - j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        put_strike = np.maximum(K - Si, 0)  # 计算提前行权时的期权收益
        put_nostrike = np.exp(-r * t) * (p * put_matrix[: i + 1, i + 1] + (1 - p) *
                                         put_matrix[1: i + 2, i + 1])  # 计算不提前行权时的期权价值
        put_matrix[: i + 1, i] = np.maximum(put_strike, put_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta = (put_matrix[0, 1] - put_matrix[1, 1]) / (S * u - S * d)  # 计算期权Delta=(Π1,1-Π1,0)/(S0u-S0d)
    if positype == "long":  # 当期权头寸是多头时
        result = Delta
    else:  # 当期权头寸是空头时
        result = -Delta
    return result
    pass

"""
2.Gamma
"""
# 通过Python自定义一个计算欧式期权的Gamma的函数
def gamma_EurOpt(S, K, sigma, r, T):
    """
    定义一个计算欧式期权Gamma的函数
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import exp, log, pi, sqrt  # 从NumPy模块导入exp、log、pi和sqrt函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算d1
    gamma = exp(-pow(d1, 2) / 2) / (S * sigma * sqrt(2 * pi * T))  # 计算Gamma
    return gamma
    pass

# 运用Python自定义一个计算美式期权Gamma的函数，并且按照看涨期权、看跌期权分别进行定义
def gamma_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Gamma的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    t = T / N  # 计算每一步步长的期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    call_matrix = np.zeros((N + 1, N + 1))  # 创建N+1行，N+1列的矩阵并且元素均为0，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N - N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上往下排序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i - j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike = np.maximum(Si - K, 0)  # 计算提前行权时的期权收益
        call_nostrike = np.exp(-r * t) * (p * call_matrix[: i + 1, i + 1] + (1 - p) *
                                          call_matrix[1: i + 2, i + 1])  # 计算不提前行权时的期权价值
        call_matrix[: i + 1, i] = np.maximum(call_strike, call_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta1 = (call_matrix[0, 2] - call_matrix[1, 2]) / (S * pow(u, 2) - S)  # 计算一个Delta
    Delta2 = (call_matrix[1, 2] - call_matrix[2, 2]) / (S - S * pow(d, 2))  # 计算另一个Delta
    Gamma = 2 * (Delta1 - Delta2) / (S * pow(u, 2) - S * pow(d, 2))  # 计算美式看涨期权Gamma
    return Gamma
    pass

def gamma_AmerPut(S, K, sigma, r, T, N):
    """定义一个运用N步二叉树模型计算美式看跌期权Gamma的函数
    S：代表基础资产当前的价格。
    K：代表期权的行权价格。
    sigma：代表基础资产收益率的波动率（年化）。
    r：代表连续复利的无风险收益率。
    T：代表期权的期限（年）。
    N：代表二叉树模型的步数"""
    t = T / N
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(r * t) - d) / (u - d)
    put_matrix = np.zeros((N + 1, N + 1))
    N_list = np.arange(0, N + 1)
    S_end = S * pow(u, N - N_list) * pow(d, N_list)
    put_matrix[:, -1] = np.maximum(K - S_end, 0)
    i_list = list(range(0, N))
    i_list.reverse()
    for i in i_list:
        j_list = np.arange(i + 1)
        Si = S * pow(u, i - j_list) * pow(d, j_list)
        put_strike = np.maximum(K - Si, 0)
        put_nostrike = np.exp(-r * t) * (p * put_matrix[: i + 1, i + 1] + (1 - p) * put_matrix[1: i + 2, i + 1])
        put_matrix[: i + 1, i] = np.maximum(put_strike, put_nostrike)
    Delta1 = (put_matrix[0, 2] - put_matrix[1, 2]) / (S * pow(u, 2) - S)
    Delta2 = (put_matrix[1, 2] - put_matrix[2, 2]) / (S - S * pow(d, 2))
    Gamma = 2 * (Delta1 - Delta2) / (S * pow(u, 2) - S * pow(d, 2))
    return Gamma
    pass

"""
3.Theta
"""
# 欧式期权的Theta
def theta_EurOpt(S, K, sigma, r, T, optype):
    """
    定义一个计算欧式期权Theta的函数
    S：代表基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    optype：代表期权的类型，输入optype="call"表示看涨期权，输入其他则表示看跌期权
    """
    from numpy import exp, log, pi, sqrt  # 从NumPy模块导入exp、log、pi和sqrt函数
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d1
    d2 = d1 - sigma * sqrt(T)  # 计算参数d2
    theta_call = (-(S * sigma * exp(-pow(d1, 2) / 2)) / (2 * sqrt(2 * pi * T)) - r * K * exp(-r * T) *
                  norm.cdf(d2))  # 计算看涨期权的Theta
    theta_put = theta_call + r * K * np.exp(-r * T)  # 计算看跌期权的Theta
    if optype == "call":  # 当期权是看涨期权时
        theta = theta_call
    else:  # 当期权是看跌期权时
        theta = theta_put
    return theta
    pass

# 运用Python自定义一个计算美式期权Theta的函数，并且按照看涨期权、看跌期权分别进行定义
def theta_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算没事看涨期权Theta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨时的比例
    call_matrix = np.zeros((N+1, N+1))  # 创建N+1行，N+1列的零矩阵，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N+1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N-N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上到下排序）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上到下排序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i+1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i-j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上到下排序）
        call_strike = np.maximum(Si - K, 0)  # 计算提前行权时的期权收益
        call_nostrike = np.exp(-r * t) * (p * call_matrix[: i+1, i+1] +
                                          (1 - p) * call_matrix[1: i+2, i+1])  # 计算不提前行权时的期权价值
        call_matrix[: i+1, i] = np.maximum(call_strike, call_nostrike)
    Theta = (call_matrix[1, 2] - call_matrix[0, 0]) / (2 * t)
    return Theta
    pass

def theta_AmerPut(S,K,sigma,r,T,N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Theta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    t = T / N
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(r * t) - d) / (u - d)
    put_matrix = np.zeros((N+1, N+1))
    N_list = np.arange(0, N+1)
    S_end = S * pow(u, N-N_list) * pow(d, N_list)
    put_matrix[:, -1] = np.maximum(K - S_end, 0)
    i_list = list(range(0, N))
    i_list.reverse()
    for i in i_list:
        j_list = np.arange(i+1)
        Si = S * pow(u, i-j_list) * pow(d, j_list)
        put_strike = np.maximum(K - Si, 0)
        put_nostrike = np.exp(-r * t) * (p * put_matrix[: i+1, i+1] + (1 - p) * put_matrix[1: i+2, i+1])
        put_matrix[: i+1, i] = np.maximum(put_strike, put_nostrike)
    Theta = (put_matrix[1, 2] - put_matrix[0, 0]) / (2 * t)
    return Theta
    pass

"""
4.Vega
"""
# 欧式期权Vega
def vega_EurOpt(S, K, sigma, r, T):
    """
    定义一个计算欧式期权Vega的函数
    S：代表基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import exp, log, pi, sqrt  # 从NumPy模块导入exp、log、pi以及sqrt函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d1
    vega = S * sqrt(T) * exp(-pow(d1, 2) / 2) / sqrt(2 * pi)  # 计算期权的Vega
    return vega
    pass

# 运用Python自定义一个计算美式期权Vega的函数，并且按照看涨期权、看跌期权分别设定
def vega_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Vega的函数，并且假定基础资产收益率的波动率是增加0.0001
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_call(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_call(S, K, sigma+0.0001, r, T, N)  # 新二叉树模型计算的期权价值
    vega = (Value2 - Value1) / 0.0001  # 计算美式看涨期权的Vega
    return vega
    pass

def vega_AmerPut(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Vega的函数，依然假定基础资产收益率的波动率是增加0.0001
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_put(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_put(S, K, sigma + 0.0001, r, T, N)  # 新二叉树模型计算的期权价值
    vega = (Value2 - Value1) / 0.0001  # 计算美式看跌期权的Vega
    return vega
    pass

# 欧式期权的Rho
def rho_EurOpt(S, K, sigma, r, T, optype):
    """
    定义一个计算欧式期权Rho的函数
    S：代表基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    optype：代表期权的类型，输入optype="call"表示看涨期权，输入其他则表示看跌期权
    """
    from numpy import exp, log, sqrt  # 从NumPy模块导入exp、log和sqrt函数
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数
    d2 = (log(S / K) + (r - pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d2
    if optype == "call":  # 当期权是看涨期权时
        rho = K * T * exp(-r * T) * norm.cdf(d2)  # 计算期权的Rho
    else:  # 当期权是看跌期权时
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)
    return rho
    pass

# 运用Python自定义一个计算美式期权Rho的函数，并且按照看涨期权、看跌期权分别设定
def rho_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Rho的函数，并且假定无风险收益率增加0.0001（1个基点）
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_call(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_call(S, K, sigma, r+0.0001, T, N)  # 新二叉树模型计算的期权价值
    rho = (Value2 - Value1) / 0.0001  # 计算美式看涨期权的Rho
    return rho
    pass

def rho_AmerPut(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Rho的函数，依然假定无风险收益率增加0.0001（1个基点）
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_put(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_put(S, K, sigma, r+0.0001, T, N)  # 新二叉树模型计算的期权价值
    rho = (Value2 - Value1) / 0.0001  # 计算美式看跌期权的Rho
    return rho
    pass

# 跨文件导入变量值实例
def pass_value_by_func(x: int, a: int = 0, b: int = 0, c: int = 1) -> bool:  # 函数声明
    """
    我们将根据二次函数是否大于100来返回真假值
    :param a: 代表可以取10~30来试验
    :param b: 代表可以取1~100来试验
    :param c: 代表可以取-100~100来试验
    :return: a*x^2+bx+c>100
    """
    value = a * x ** 2 + b * x + c
    if value > 100:
        return True
    else:
        return False

if __name__ == '__main__':
    print(pass_value_by_func(1, 2, 3, 4))
    print(pass_value_by_func(2, 3, 4, 5))
    print(pass_value_by_func(3, 4, 5, 6))
    print(pass_value_by_func(4, 5, 6, 7))
    print(pass_value_by_func(5, 6, 7, 8))
    print(pass_value_by_func(6, 7, 8, 9))
    print(pass_value_by_func(7, 8, 9, 10))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
    print(pass_value_by_func(0))
