from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
import pandas as pd
import numpy as np


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    x = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def get_dummies(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def scaling(df):
    scaler = preprocessing.StandardScaler()
    age_transpose = np.array(df['Age']).reshape(-1, 1)
    age_scale_param = scaler.fit(age_transpose)
    df['Age_scaled'] = scaler.fit_transform(age_transpose, age_scale_param)

    fare_transpose = np.array(df['Fare']).reshape(-1, 1)
    fare_scale_param = scaler.fit(fare_transpose)
    df['Fare_scaled'] = scaler.fit_transform(fare_transpose, fare_scale_param)
    return df


def preprocess(data_train):
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    data_train = get_dummies(data_train)
    data_train = scaling(data_train)
    return data_train, rfr


def training(data_train):
    # 用正则取出我们要的属性值
    train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    return clf


# ===============================

def output_test_result(rfr, clf, data_test):
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

    age_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # 用得到的预测结果填补原缺失数据
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

    data_test = set_Cabin_type(data_test)
    data_test = get_dummies(data_test)
    data_test = scaling(data_test)

    test = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("Data/logistic_regression_predictions.csv", index=False)


if __name__ == '__main__':
    data_train = pd.read_csv("Data/train.csv")
    data_test = pd.read_csv("Data/test.csv")

    data_train, rfr = preprocess(data_train)
    if 0:
        clf = training(data_train)
        output_test_result(rfr, clf, data_test)
    else:

        # 简单看看打分情况
        clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        all_data = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        X = all_data.as_matrix()[:, 1:]
        y = all_data.as_matrix()[:, 0]
        print(model_selection.cross_val_score(clf, X, y, cv=5))

