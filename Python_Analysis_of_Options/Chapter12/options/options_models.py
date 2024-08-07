"""
This file includes the functions needed to calculate the theoretical price of options
by using Black-Scholes Model(for European options) or Binomial Tree Model(for American options)
"""
SOME_GLOBAL_VAR_UTILIZED_EVERYWHERE_UNRECOMMENDED = None
# 导入所需的包
import numpy as np

# 通过Python自定义一个运用布莱克-斯科尔斯-默顿模型计算欧式看涨、看跌期权价格的函数
def option_BSM(S, K, sigma, r, T, opt):
    """
    定义一个运用布莱克-斯科尔斯-默顿模型计算欧式期权价格的函数
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    opt：代表期权类型，输入opt="call"表示看涨期权，输入其他则表示看跌期权
    """
    from numpy import log, exp, sqrt  # 从NumPy模块导入log、exp、sqrt这3个函数
    from scipy.stats import norm  # 从SciPy的子模块stats导入norm函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d1
    d2 = d1 - sigma * sqrt(T)  # 计算参数d2
    if opt == "call":  # 针对欧式看涨期权
        value = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)  # 计算期权价格
    else:  # 针对欧式看跌期权
        value = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)  # 计算期权价格
    return value

    pass

# 针对计算美式看涨期权价值的Python自定义函数
def American_call(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权价值的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    # 为了便于理解代码编写逻辑，分为以下3个步骤
    # 第1步时计算相关系数
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    call_matrix = np.zeros((N+1, N+1))  # 创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值

    # 第2步是计算到期日节点的基础资产价格与期权价值
    N_list = np.arange(0, N+1)  # 创建从0到N的自然数数列（数组形式）
    S_end = S * pow(u, N-N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格。按照节点从上往下排序，参见式（11-38）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上往下排序）

    # 第3步是计算期权非到期日节点的基础资产价格与期权价值
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排列（从N-1到0）
    for i in i_list:
        j_list = np.arange(i+1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i-j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike = np.maximum(Si - K, 0) # 计算提前行权时的期权收益
        call_nostrike = (p * call_matrix[: i+1, i+1] + (1 - p) * call_matrix[1: i+2, i+1]) * np.exp(-r * t)  # 计算不提前行权时的期权价值
        call_matrix[:i+1, i] = np.maximum(call_strike, call_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
        call_begin = call_matrix[0, 0]  # 期权初始价值
    return call_begin

    pass

# 针对计算美式看跌期权价值的Python自定义函数
def American_put(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权价值的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数
    """
    # 第1步计算相关参数
    t = T / N
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(r * t) - d) / (u - d)
    put_matrix = np.zeros((N + 1, N + 1))
    # 第2步计算期权到期日节点的基础资产价格与期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * u ** (N - N_list) * d ** (N_list)  # 计算期权到期时节点的基础资产价格。按照节点从上往下排序
    put_matrix[:, -1] = np.maximum(K - S_end, 0)  # 计算期权到期时节点的看跌期权价值。按照节点从上往下排序
    # 第3步计算期权非到期日节点的基础资产价格与期权价值
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * u ** (i - j_list) * d ** (j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        put_strike = np.maximum(K - Si, 0)  # 计算提前行权时的期权收益
        put_nostrike = (p * put_matrix[:i + 1, i + 1] + (1 - p) * put_matrix[1:i + 2, i + 1]) * np.exp(
            -r * t)  # 计算不提前行权时的期权收益
        put_matrix[:i + 1, i] = np.maximum(put_strike, put_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权收益中的最大值
    put_begin = put_matrix[0, 0]  # 期权初始价值
    return put_begin

    pass
