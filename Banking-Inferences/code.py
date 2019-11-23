# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  

# path        [File location variable]
data = pd.read_csv(path)

#Code starts here
data_sample = data.sample(n=sample_size, random_state=0)
print(data_sample)

# finding the mean of the sample
sample_mean = data_sample['installment'].mean()
print(sample_mean)

sample_std = data_sample['installment'].std()
print(round(sample_std, 2))

# finding the margin of error
margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))
print(round(margin_of_error, 2))

#confidence_interval
c1 = round(sample_mean - margin_of_error, 2) 
c2 = round(sample_mean + margin_of_error, 2)
confidence_interval = [c1, c2]
print(confidence_interval)

#true_mean = data['installment'].mean()
#if true_mean >= c1 and true_mean <= c2:
#    print(round(true_mean, 2))
true_mean = round(data.installment.mean(), 2)
if(true_mean >= c1 and true_mean <= c2):
    print('Population mean falls between the confidence interval')
else:
    print('Population mean does not fall between the confidence interval')








# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
#fig, (ax_1,ax_2,ax_3) = plt.subplots(nrows = 3 , ncols = 1)
fig, axes = plt.subplots(nrows = 3 , ncols = 1)

for i in range (len(sample_size)):
    m = []
    for j in range (1000):
        data_sample = data['installment'].sample(sample_size[i]).mean() #, random_state=0)
        print(data_sample)
        m.append(sample_mean)
        print(m)
    mean_series = pd.Series(m)
    print(mean_series)

mean_series.hist()
plt.show() 


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
#data['int.rate'].map(lambda x: str(x)[:-1])

#data['int.rate'] = data['int.rate'].div(100).round(2)
#data['int.rate'] = data['int.rate'].astype(float)
#data['int.rate'] = data['int.rate']/100

#data['int.rate'].astype(float)/100
#x1 = data[data['purpose'] == 'small_business']['int.rate']
#value = data['int.rate'].mean()


#z_statistic, p_value = stests.ztest(data['int.rate'], x1=data[data['purpose']=='small_business']['int.rate'], value=data['int.rate'].mean(), alternative='larger')
#print(round(z_statistic, 2))
#print(round(p_value, 35))

#z_statistic, p_value = ztest(data[data['purpose'] == 'small_business']['int.rate'] , value = data['int.rate'].mean(), alternative='larger')
#print("Z-statistics = ",z_statistic)
#print("p-value = ",p_value)

#if p_value < 0.05:
#    inference = "Reject"
#else:
#    inference = "Accept"





data['int.rate'] = data['int.rate'].apply(lambda x : str(x).replace('%', ''))
data['int.rate'] = data['int.rate'].astype(float)
data['int.rate'] = data['int.rate']/100
x1 = data[data['purpose'] == 'small_business']['int.rate']
value = data['int.rate'].mean()
z_statistic, p_value = ztest(data[data['purpose'] == 'small_business']['int.rate'] , value = data['int.rate'].mean(), alternative='larger')
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)
if(p_value < 0.005):
    inference = "Reject"
else:
    inference = "Can't Reject"
print('Inference: '+inference+' Null Hypothesis')


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
x1 = data[data['paid.back.loan']=='No']['installment']
x2 = data[data['paid.back.loan']=='Yes']['installment']
#value = data['int.rate'].mean()
z_statistic, p_value = ztest(data[data['paid.back.loan']=='No']['installment'], data[data['paid.back.loan']=='Yes']['installment'])
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)
if(p_value < 0.005):
    inference = "Reject"
else:
    inference = "Accept"
print('Inference: '+inference+' Null Hypothesis')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
#Critical value
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1
yes = data[data['paid.back.loan'] == 'Yes']['purpose'].value_counts()
no = data[data['paid.back.loan'] == 'No']['purpose'].value_counts()
observed = pd.concat([yes.transpose(), no.transpose()], keys= ['Yes','No'], axis=1)
print(observed)
#observed = yes.transpose(axis=1, keys=['Yes', 'No'])+no.transpose(axis=1, keys=['Yes', 'No'])

chi2, p, dof, ex = chi2_contingency(observed) #, correction = False)
print("chi2 = ",z_statistic)
print("p-value = ",p_value)
print("dof = ",dof)
print("ex = ",ex)
print("critical_value = ",critical_value)


if(chi2 > critical_value):
    inference = "Reject"
else:
    inference = "Cannot be rejected"
print('Inference: '+inference+' Null Hypothesis')



