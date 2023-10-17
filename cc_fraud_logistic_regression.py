import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Initial import of the CC transaction fraud dataset.
fraud_data = pd.read_csv("/Users/JP/Desktop/new_fraudTrain.csv")
fraud_data

# info function shows the type of data being worked with and gives insight into how many data points there are as well as any null data points. It can be seen that there is an even combination of categorical and numerical data.
fraud_data.info()

# While the info function gave us an idea of the presence or lack of null values, using the isnull() function paired with the sum() function confirms that there are no missing or incomplete variables.
fraud_data.isnull().sum()

# Fraudulent credit card transactions naturally have a tendency to be imbalanced. Using the value_counts() function of a specific feature shows how balanced or imbalanced a feature is. Luckily the dependent variable (is_fraud) has an even split of fraudulent and legitimate transactions.
fraud_data['is_fraud'].value_counts()

# drop() function executes the removal of features from the dataset. In this case, features were removed from the dataset that did not seem to be necessary for the overall training of the model.
fraud_data2 = fraud_data.drop(['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat','merch_long'], axis=1)
fraud_data2.head(10)

fraud_data2.describe()

# This scatterplot shows the relationship between transaction amount and gender as they relate to fraudulent and legitimate transactions. A quick look at this visual shows the fraudulent transactions linger between $500 and $1500.
sns.scatterplot(data=fraud_data2, x="amt", y="gender", hue="is_fraud")
plt.show()

# This scatterplot shows the relationship between city population and gender as they relate to fraudulent and legitimate transactions. Fraudulent transactions here seem to be spread out, potentially implying that city population is not a major indicator of fraud.
sns.scatterplot(data=fraud_data2, x="city_pop", y="gender", hue="is_fraud")
plt.show()

# This scatterplot shows the relationship between transaction amount and transaction category as they relate to fraudulent and legitimate transactions. A quick glance shows a good chunk of fraudulent transactions occurring over online transactions.
sns.scatterplot(data=fraud_data2, x="amt", y="category", hue="is_fraud")
plt.show()

# This scatterplot shows the relationship between city population and transaction category as they relate to fraudulent and legitimate transactions. just like the prior scatterplot that incorporated city population, this graph also shows a spread out view of fraudulent transactions amongst the transaction categories. While this confirms what was suspected with city population, it also helps to display a closer view of the many fraudulent transactions in online shopping.
sns.scatterplot(data=fraud_data2, x="city_pop", y="category", hue="is_fraud")
plt.show()

# This scatterplot incorporates the credit card number feature. There is not much to pull from this visual which could indicate the inclusion of the credit card numbers will neither help or harm the overall model.
sns.scatterplot(data=fraud_data2, x="cc_num", y="category", hue="is_fraud")
plt.show()

# This scatterplot again include CC numbers as well as transaction amount. Once more, confirming the transaction amounts that fraudulent transactions seem to be clustered in.
sns.scatterplot(data=fraud_data2, x="cc_num", y="amt", hue="is_fraud")
plt.show()

# This step and the next are splitting the transaction fraud data into X and Y datasets. The X dataset includes all of the independent variables such as CC number, merchant, category, amount, gender, state, and city population.
X = fraud_data2.drop(['is_fraud'], axis=1)

# The Y dataset consists of the dependent variable which is the focal point of the logistic regression, the feature that indicates fraudulent and legitimate transactions.
Y = fraud_data2[['is_fraud']]

# OrdinalEncoder() function will encode all the categorical variables included in the model. This transforms words into assigned numbers that can be trained on the model.
categorical_variables = ['merchant', 'category', 'gender', 'state']
encoder = OrdinalEncoder()
encoder.fit(X[categorical_variables])

X_enc = encoder.transform(X[categorical_variables])
X[categorical_variables] = X_enc
X.head(10)

# The MinMaxScaler() function organizes the variables in a way that levels out the numerical values. For example, the credit card numbers, though they are CC numbers, without scaling those numbers, they throw off the model training because of how large the credit card numbers are.
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# train_test_split() splits the X & Y datasets into training and testing datasets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# This step creates and trains the logistic regression model.
fraud_model = LogisticRegression()
fraud_model.fit(X_train, Y_train)

fraud_predict = fraud_model.predict(X_test)

# Accuracy, precision, recall, and F1 scores are all calculations that will help determine model accuracy and performance.
accuracy = accuracy_score(Y_test, fraud_predict)
precision = precision_score(Y_test, fraud_predict)
recall = recall_score(Y_test, fraud_predict)
f1 = f1_score(Y_test, fraud_predict)

print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)
print('F1 Score', f1)

# The confusion matrix shows the ratio of positive and false positive predictions. This is another indicator of how well the model is performing.
cm = confusion_matrix(Y_test, fraud_predict)
labels = ['Negative', 'Positive']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
print(df_cm)

# The ROC curve is a visual representation of how the model is performing. The closer to 1 the peak of the curve is, the better the model is performing.
fpr, tpr, _ = roc_curve(Y_test, fraud_predict)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()