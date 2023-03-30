import pickle
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore', category=FutureWarning)

df = pd.read_csv('/Users/abdullahhanani/Desktop/a2/bike_buyers_clean.csv', sep=',')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

df = df.replace({'No': 0, 'Yes': 1})
df = df.replace({'Male': 0, 'Female': 1})
df = df.replace({'Married': 0, 'Single': 1})
df = df.replace({'Bachelors': 0, 'Partial College': 1, 'High School': 2,
                 'Partial High School': 3, 'Graduate Degree': 4})
df = df.replace({'Skilled Manual': 0, 'Clerical': 1, 'Professional': 2, 'Manual': 3, 'Management': 4})
df = df.replace({'0-1 Miles': 0, '1-2 Miles': 1, '2-5 Miles': 2, '5-10 Miles': 3, '10+ Miles': 4})
df = df.replace({'Europe': 0, 'Pacific': 1, 'North America': 2})

def viewAndGetOutliersByPercentile(dataframe, colName, lowerP, upperP):
    up = dataframe[colName].quantile(upperP)
    lp = dataframe[colName].quantile(lowerP)
    return lp, up

LOWER_PERCENTILE = 0.025
UPPER_PERCENTILE = 0.925
lp, up = viewAndGetOutliersByPercentile(df, 'Income', LOWER_PERCENTILE, UPPER_PERCENTILE)

df = df[(df["Income"] >lp) & (df["Income"] < up)]

X = df.drop(['Purchased Bike', 'ID'], axis=1)

y = df['Purchased Bike']

model = RandomForestClassifier()

# rfe = RFE(model, n_features_to_select=8)
# ffs = f_regression(X, y)

# rfe = rfe.fit(X, y)
# print('Backward Feature Elimination')
# for i in range(0, len(X.keys())):
#     if(rfe.support_[i]):
#         print(X.keys()[i])

# print('\n\nForward Feature Selection')
# featuresDf = pd.DataFrame()
# for i in range(0, len(X.columns)):
#     featuresDf = featuresDf.append({"feature":X.columns[i],
#                                     "ffs":ffs[0][i]}, ignore_index=True)
# featuresDf = featuresDf.sort_values(by=['ffs'])
# print(featuresDf)

featuresDf = df[['Gender', 'Income', 'Education', 'Occupation', 'Commute Distance', 'Age']]

X_train, X_test, y_train, y_test = train_test_split(featuresDf, y, test_size=0.3, random_state=999)

model.fit(X_train, y_train)

# with open('model_pkl', 'wb') as files:
#     pickle.dump(model, files)

# with open('model_pkl', 'rb') as f:
#     loadedModel = pickle.load(f)

result = model.score(X_test, y_test)

print('Model accuracy score: {0:0.4f}'.format(result))
