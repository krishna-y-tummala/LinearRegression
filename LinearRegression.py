#BEGIN
#IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

#READ DATA
customers = pd.read_csv('C:\\Users\\User\\Documents\\Ecommerce Customers.csv')

#DATA DESCRIPTIONS
print(customers.head())
print('\n',customers.describe())
print('\n',customers.info())

#TIME ON WEBSITE VS YEARLY AMOUNT SPENT
i = sns.jointplot(data=customers,x='Time on Website',y='Yearly Amount Spent')
i.savefig('TimeonWebsite vs YearlyAmt.jpg')
plt.show()

#TIME ON APP VS YEARLY AMOUNT SPENT
i2 = sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent')
i2.savefig('TimeonApp vs YearlyAmt.jpg')
plt.show()

#TIME ON APP VS LENGTH OF MEMBERSHIP
i3 = sns.jointplot(data=customers,x='Time on App',y='Length of Membership',kind='hex')
i3.savefig('TimeonApp vs LengthofMembership Hex Plot.jpg')
plt.show()

#PAIRPLOT
i4 = sns.pairplot(data=customers,palette='grey')
i4.savefig('Pairplot.jpg')
plt.show()

#CORRELATION TABLE
print('\nCorrelation Table:','\n',customers.corr(),'\n')

#LINEAR REGRESSION PLOT
i5 = sns.lmplot(data=customers,x='Length of Membership',y='Yearly Amount Spent',height=7)
i5.savefig('LinearRegPlot.jpg')
plt.show()

#TRAIN AND TEST DATA SPLIT
from sklearn.model_selection import train_test_split

X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

#METRICS
#Coefficients
print('Coefficients:\n',lm.coef_)

#Predictions
predictions = lm.predict(X_test)

#Actual Y vs Predicted Y 
i6 = plt.scatter(x=y_test,y=predictions,s=10)
plt.xlabel('Actual Y values')
plt.ylabel('Predicted Y values')
plt.savefig('Actual Y vs Precicted Y.jpg')
plt.show()

from sklearn import metrics

#Errors
print('\nMAE:',metrics.mean_absolute_error(y_test,predictions),'\nMSE:',metrics.mean_squared_error(y_test,predictions),'\nRMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

#Error Distribution
i7 = sns.displot(x=(y_test-predictions),kde=True,bins=50)
i7.savefig('Predictions Distribution Plot.jpg')
plt.show()

#Coefficients
coeffs = pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coefficient'])
print('\n',coeffs)

#Insights
print("\nFor one unit increase in Avg. Session Length, the Yearly Amount Spent increases by",coeffs['Coefficient']['Avg. Session Length'], "units",
"\nFor one unit increase in Time on App, the Yearly Amount Spent increases by",coeffs['Coefficient']['Time on App'], "units",
"\nFor one unit increase in Time on Website, the Yearly Amount Spent increases by",coeffs['Coefficient']['Time on Website'], "units",
"\nFor one unit increase in Length of Membership, the Yearly Amount Spent increases by",coeffs['Coefficient']['Length of Membership'], "units")

print('\nThe company can look at the results in two way. The first would be to focus on developing their Website to catch up with the impact their mobile application has.\n'
     'The second is to develop their Mobile App to cotinue driving revenue. The second approach is advisable because Time on App is positively correlated to Length of Membership.\n'
     'The time on Website has negative correlation with the Yearly Amount Spent, the more time customers spend on the Website, the lesses they spend.\n'
     'Considering all this, it is advised to focus on developing the Mobile Application.\n')

#END