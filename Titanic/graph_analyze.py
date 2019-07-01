import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_train = pd.read_csv("Data/train.csv")


def show1():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
    plt.title(u"获救情况 (1为获救)")  # 标题
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    ax = plt.subplot2grid((2, 3), (0, 2))
    age_df = data_train.loc[(data_train.Survived == 1), 'Age'].value_counts().sort_index()
    age_df_0 = data_train.loc[(data_train.Survived == 0), 'Age'].value_counts().sort_index()
    df = pd.DataFrame({'ss':age_df, "00":age_df_0})
    df.plot(kind="line", ax = ax)
    plt.ylabel('num')  # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")  # plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.

    plt.subplot2grid((2, 3), (1, 2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.show()


def show2():
    fig = plt.figure()
    fig.set(alpha=0.2)

    ax = plt.subplot2grid((1, 2), (0, 0))
    survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    df = pd.DataFrame({"Survived_1": survived_1, "Survived_0": survived_0})
    df.plot(kind="bar", stacked=True, ax=ax)
    plt.xlabel("阶层")
    plt.ylabel("人数")

    ax = plt.subplot2grid((1, 2), (0, 1))
    survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
    survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({"Survived_1": survived_1, "Survived_0": survived_0})
    df.plot(kind="bar", stacked=True, ax=ax)
    plt.xlabel("性别")
    plt.ylabel("人数")

    plt.show()

show1()
