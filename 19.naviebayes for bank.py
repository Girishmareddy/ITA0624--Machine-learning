import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
dataset = pd.read_csv("C:\\Users\\giris_pu2cvr5\\Downloads\\machine learning\\bank_loan_data.csv")
X = dataset[['Age', 'Income', 'Loan_Amount']]
y = dataset['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
new_customer_features = pd.DataFrame({
    'Age': [float(input("Enter customer's age: "))],
    'Income': [float(input("Enter customer's income: "))],
    'Loan_Amount': [float(input("Enter loan amount: "))]
})
loan_default_prediction = clf.predict(new_customer_features)
if loan_default_prediction[0] == 'Approved':
    print('The new customer is likely to be approved for a loan.')
else:
    print('The new customer is unlikely to be approved for a loan.')
