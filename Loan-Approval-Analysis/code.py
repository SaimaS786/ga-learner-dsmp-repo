# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 


filepath = path
df=pd.read_csv(filepath, sep=',', delimiter=None)
#print(df)


# code starts here
bank = pd.DataFrame(columns=['Path', 'File_Name']).set_index('Path')

categorical_var = df.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = df.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
#banks = DataFrame.drop(self, labels=None, axis=0, index=None, columns='Loan ID', level=None, inplace=False, errors='raise')
#bank = pd.DataFrame(columns=['Path', 'File_Name']).set_index('Path')
#print(bank)

banks = df.drop(labels=None, axis=0, index=None, columns='Loan_ID', level=None, inplace=False, errors='raise')
print(banks)
#banks = df.drop('Loan_ID')

print(banks.isnull().sum())
#print(isnull().sum())


#bank_mode = mode(banks)
#DataFrame.mode(self, axis=0, numeric_only=False, dropna=True)
bank_mode = df.mode(axis=0, numeric_only=False)

#banks.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)

# replacing na values in college with No college 
#nba["College"].fillna("No College", inplace = True) 
banks.fillna('bank_mode', inplace = True)

print(banks)


#code ends here



# --------------

avg_loan_amount = pd.pivot_table(banks, index=['Gender','Married','Self_Employed'], values='LoanAmount',aggfunc=np.mean)
print(avg_loan_amount)

# code ends here



# --------------

#loan_approved_se = bank[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')].count()
#print(loan_approved_se)
#loan_approved_nse = bank[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')].count()
#print(loan_approved_nse)
#Loan_Status = 614
#percentage_se = (loan_approved_se * 100 / 614)
#print(percentage_se)
#percentage_nse = (loan_approved_nse * 100 / 614)
#print(percentage_nse)

# code ends here

loan_approved_se = len(banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')])
print(loan_approved_se)
loan_approved_nse = len(banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')])
print(loan_approved_nse)
total = banks.shape[0]
percentage_se = (loan_approved_se/total*100)
print(percentage_se)
percentage_nse = (loan_approved_nse/total*100)
print(percentage_nse)



# --------------
# code starts here



loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12) 
print(loan_term)

##big_loan_term = len(banks['Loan_Amount_Term'] >= 25).count()
#big_loan_term = len(banks['loan_term'] >= 25)
#big_loan_term = len(banks[loan_term] >= 25)
big_loan_term = len(banks[loan_term >= 25])
print(big_loan_term)

# code ends here














#df[col].apply()
#df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
#df['Loan_Amount_Term'] = df['Loan_Amount_Term'].apply(lambda x: calendar.month_abbr[x])
#loan_term = df['Loan_Amount_Term'].apply(lambda x: calendar.month_abbr[x])








# --------------
# code starts here


loan_groupby = banks.groupby('Loan_Status')
print(loan_groupby)

loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]
print(loan_groupby)

mean_values = loan_groupby.mean()
print(mean_values)

# code ends here


