# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
path 
df = pd.read_csv(path)
total = len(df)
print(total)

temp = df[df['fico']>700]
sum_A = len(temp.index)
p_a = sum_A/total
print(p_a)

tempdf = df[df['purpose']=='debt_consolidation']
sum_B = len(tempdf.index)
p_b = sum_B/total
print(p_b)

df1 = 'purpose' == 'debt_consolidation'
print(df1)

probability_both = p_a * p_b

p_a_b = probability_both/p_a
print(p_a_b)

result = p_a_b == p_a
print(result)

# code ends here


# --------------
# code starts here
prob_lp = df[df['paid.back.loan'] == 'Yes'].shape[0]/df.shape[0]
print(prob_lp)


#tempdf1 = df[df['paid.back.loan'] == 'Yes']
#sum_AA = len(tempdf1.index)
#prob_lp = sum_AA/total
#print(prob_lp)
prob_cs = df[df['credit.policy'] == 'Yes'].shape[0]/df.shape[0]
print(prob_cs)

#tempdf2 = df[df['credit.policy'] == 'Yes']
#sum_AB = len(tempdf2.index)
#prob_cs = sum_AB/total
#print(prob_cs)

new_df = df[df['paid.back.loan'] == 'Yes']
#print(new_df)

#probability_both = prob_lp * prob_cs

#prob_pd_cs = probability_both/prob_lp
#print(prob_pd_cs)
#p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0]/new_df.shape[0]
print(prob_pd_cs)

bayes = (((prob_pd_cs)*(prob_lp))/prob_cs)
print(bayes)

# code ends here


# --------------
# code starts here
#plt.bar('purpose')
#purpose.plot(kind = 'bar')
#df.plot.bar('purpose')
#plt.show()

df1 = df[df['paid.back.loan'] == 'No']
df1.plot(kind='bar')
plt.show()

# code ends here


# --------------
# code starts here

inst_median = df['installment'].median()
inst_mean = df['installment'].mean()

plt.hist('installment', bins=8)

plt.hist('log.annual.inc', bins=8)


# code ends here


