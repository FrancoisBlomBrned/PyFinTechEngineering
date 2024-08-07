#! "D:/PythonProjects/PyFinTechnology/PyFinTechEngineering/venv/scripts/python.exe"
#_*_coding: UTF-8_*_

def plotting(figsize, x_value, y_value, plotting_legend, line_label, width, x_label, y_label, fontsize, title):
    # 导入所需的包以及修改字体
    import matplotlib.pyplot as plt
    from pylab import mpl
    mpl.rcParams["font.sans-serif"] = ["FangSong"]
    mpl.rcParams["axes.unicode_minus"] = False
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    # 绘图
    # plt.figure(figsize)
    plt.plot(x_value, y_value, plotting_legend, label = line_label, lw = width)
    # TODO: plt.plot(S_list, value_approx1, "r-", label = u"运用Delta计算得到的看涨期权近似价格", lw = 2.5)
    plt.xlabel(x_label, fontsize)
    plt.ylabel(y_label, fontsize)
    plt.xticks(fontsize)
    plt.yticks(fontsize)
    plt.title(title, fontsize)
    plt.legend(fontsize)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plotting((9, 6), [1, 2, 3], [3, 4, 5], "r-", u"这是一个linelabel",
             2.5, u"这是一个xlabel", u"这是一个ylabel", 13, u"这是一个title")
