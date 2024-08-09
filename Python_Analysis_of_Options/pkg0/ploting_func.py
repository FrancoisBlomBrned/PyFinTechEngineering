#! "D:/PythonProjects/PyFinTechnology/PyFinTechEngineering/venv/scripts/python.exe"
#_*_coding: UTF-8_*_

import traceback
from typing import Tuple, Dict

def plotting_correct(figsize, x_value, y_value, plotting_legend, line_label, width, x_label, y_label, fontsize, title)\
        -> Tuple[Dict, Dict]:

    try:
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
        #########plt.xlabel(x_label, fontsize)
        #########plt.ylabel(y_label, fontsize)

        plt.xlabel(x_label, fontsize = fontsize)
        plt.ylabel(y_label, fontsize = fontsize)

        ########plt.xticks(fontsize)
        ########plt.yticks(fontsize)
        ########plt.title(title, fontsize)
        ########plt.legend(fontsize)

        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.title(title, fontsize = fontsize)
        plt.legend(fontsize = fontsize)

    except ImportError as implE:
        print(implE)
    except BaseException as bsE:
        print(bsE)
    finally:
        plt.grid()
        plt.show()

def plotting_error(figsize, x_value, y_value, plotting_legend, line_label, width, x_label, y_label, fontsize, title)\
        -> None:

    try:
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
        #########plt.xlabel(x_label, fontsize)
        #########plt.ylabel(y_label, fontsize)
        plt.xlabel(x_label, fontsize)
        plt.ylabel(y_label, fontsize)
        ############  65 行報錯
####################  66 亦報錯

        plt.xticks(fontsize)
        plt.yticks(fontsize)
        plt.title(title, fontsize)
        plt.legend(fontsize)

    except ImportError as implE:
        print(implE)
    except BaseException as bsE:
        print(bsE)
    finally:
        plt.grid()
        plt.show()



if __name__ == "__main__":
    plotting_error((9, 6), [1, 2, 3], [3, 4, 5], "r-", u"这是一个linelabel",
             2.5, u"这是一个xlabel", u"这是一个ylabel", 13, u"这是一个title")

    plotting_correct((9, 6), [1, 20, 35, 42.5, 45.5, 47.25, 49, 50.05, 50.53], [3, 4, 5, 6, 7, 8, 9, 10, 11], "r-", r"demo label",
             2.5, r"look-xlabel", r"look-ylabel", 13, r"some title")
