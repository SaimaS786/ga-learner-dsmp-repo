# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path
data = pd.read_csv(path)

loan_status = data['Loan_Status'].value_counts()

# bar chart

loan_status.plot(kind="bar")

# display plot
plt.show()

#Code starts here


# --------------
#Code starts here




property_and_loan = data.groupby(['Property_Area', 'Loan_Status']).size().unstack()

property_and_loan.plot(kind = 'bar', stacked=False)

plt.xlabel('Property Area')
plt.ylabel('Loan Status')

plt.xticks(rotation=45)

plt.show()


# --------------
#Code starts here




education_and_loan = data.groupby(['Education', 'Loan_Status']).size().unstack()

education_and_loan.plot(kind='bar')

plt.xlabel('Education Status')

plt.ylabel('Loan Status')

plt.xticks(rotation=45)

plt.show()


# --------------
#Code starts here
#dragon = df[df['Type 1'] == 'Dragon']
graduate = data[data['Education'] == 'Graduate']
#graduate = data['Education'] == 'Graduate'

not_graduate = data[data['Education'] == 'Not Graduate']
#not_graduate = data['Education'] == 'Not Graduate'


LoanAmount = pd.Series(np.random.randn(1000))

LoanAmount.plot(kind='density', label='Graduate')

LoanAmount.plot(kind='density', label='Not Graduate')






#Code ends here

#For automatic legend display
plt.legend()

plt.show()


# --------------

#data = pd.DataFrame(columns = ['TotalIncome'])
#
#data = data(columns= 'TotalIncome')
#TotalIncome = ApplicantIncome.value_counts() + CoapplicantIncome.value_counts()
#data['TotalIncome'] = (data.ApplicantIncome, data.CoapplicantIncome).sum()
#sum = data['ApplicantIncome'] + data['CoapplicantIncome']
#data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

#data = pd.DataFrame(columns = ['TotalIncome'])
#sum = data['ApplicantIncome'] + data['CoapplicantIncome']
#data['TotalIncome'] = sum
#data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
#print(data)













#############################################################

#Code starts here
# Initialize figure and axes
fig, (ax_1,ax_2,ax_3) = plt.subplots(nrows = 3 , ncols = 1)

ax_1.scatter(data.ApplicantIncome, data.LoanAmount)
ax_1.set_title('Applicant Income')

ax_2.scatter(data.CoapplicantIncome, data.LoanAmount)
ax_2.set_title('Coapplicant Income')

#data = pd.DataFrame(columns = ['TotalIncome'])

data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

ax_3.scatter(data.TotalIncome, data.LoanAmount)
ax_3.set_title('Total Income')

plt.show()















