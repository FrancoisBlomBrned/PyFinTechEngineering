# TODO:基于Delta的对冲
def Delta_hedge(N_put, N_Stock, ):
    N_put = 1e6  # 持有看跌期权多头头寸

    N_ABC = np.abs(delta_EurOpt3 * N_put)  # 用于对冲的农业银行A股股票数量（变量Delta_EurOpt3在前面已设定）
    N_ABC = int(N_ABC)  # 转换为整型
    print("2020年7月16日买入基于期权Delta对冲的农业银行A股数量", N_ABC)

    import datetime as dt  # 导入datetime模块

    T0 = dt.datetime(2020, 7, 16)  # 设置期权初始日（也就是对冲初始日）
    T1 = dt.datetime(2020, 8, 31)  # 设置交易日2020年8月31日
    T2 = dt.datetime(2021, 1, 16)  # 设置期权到期日
    T_new = (T2 - T1).days / 365  # 2020年8月31日至期权到期日的剩余期限（年）

    S_Aug31 = 3.21  # 2020年8月31日农业银行A股股价
    shibor_Aug31 = 0.02636  # 2020年8月31日6个月期Shibor

    put_Jul16 = option_BSM(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC,
                           opt="put")  # 期权初始日看跌期权价格
    put_Aug31 = option_BSM(S=S_Aug31, K=K_ABC, sigma=sigma_ABC, r=shibor_Aug31, T=T_new,
                           opt="put")  # 2020年8月31日看跌期权价格
    print("2020年7月16日农业银行A股欧式看跌期权价格", round(put_Jul16, 4))
    print("2020年8月31日农业银行A股欧式看跌期权价格", round(put_Aug31, 4))

    port_chagvalue = N_ABC * (S_Aug31 - S_ABC) + N_put * (put_Aug31 - put_Jul16)  # 静态对冲策略下2020年8月31日投资组合的累积盈亏
    print("静态对冲策略下2020年8月31日投资组合的累积盈亏", round(port_chagvalue, 2))

    # 第2步：计算在2020年8月31日看跌期权的Delta以及保持该交易日期权Delta中性而需要针对基础资产（农业银行A股）新增交易情况
    delta_Aug31 = delta_EurOpt(S=S_Aug31, K=K_ABC, sigma=sigma_ABC, r=shibor_Aug31, T=T_new,
                               optype="put", positype="long")  # 计算2020年8月31日的期权Delta
    print("2020年8月31日农业银行A股欧式看跌期权Delta", round(delta_Aug31, 4))

    N_ABC_new = np.abs(delta_Aug31 * N_put)  # 2020年8月31日保持Delta中性而用于对冲的农业银行A股股票数量
    N_ABC_new = int(N_ABC_new)  # 转换为整型
    print("2020年8月31日保持Delta中性而用于对冲的农业银行A股股票数量", N_ABC_new)

    N_ABC_change = N_ABC_new - N_ABC  # 保持Delta中性而发生的股票数量变化
    print("2020年8月31日保持Delta中性而发生的股票数量变化", N_ABC_change)
    pass