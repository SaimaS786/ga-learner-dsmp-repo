# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)

#Code starts here 

data['Gender'].replace('_', 'Agender', inplace=True)

gender_count = data['Gender'].value_counts()

gender_count.plot(kind='bar')
#plt.bar(gender_count)
plt.show()


# --------------
#Code starts here

alignment = data['Alignment'].value_counts()
print(alignment)

#alignment.plot(kind="pie")

plt.pie(alignment) #, labels=Character Alignment)
plt.title('Character Alignment')
plt.show()



# --------------
#Code starts here
sc_df = data[['Strength', 'Combat']]
#print(sc_df)
sc_covariance = sc_df.Strength.cov(sc_df.Combat)
print(sc_covariance)
sc_strength = sc_df['Strength'].std()
#print(sc_strength)
sc_combat = sc_df['Combat'].std()
#print(sc_combat)
sc_pearson = sc_covariance/((sc_strength)*(sc_combat))
print(sc_pearson)

ic_df = data[['Intelligence','Combat']]
print(ic_df)
ic_covariance = ic_df.Intelligence.cov(ic_df.Combat)
print(ic_covariance)
ic_intelligence= ic_df['Intelligence'].std()
#print(ic_intelligence)
ic_combat = ic_df['Combat'].std()
#print(ic_combat)
ic_pearson = ic_covariance/((ic_intelligence)*(ic_combat))
print(ic_pearson)



# --------------
#Code starts here

total_high = data.Total.quantile(q=0.99)

super_best = data[data['Total'] > total_high]
print(super_best)

super_best_names = [super_best]
print(super_best_names)




# --------------
#Code starts here

fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows = 3 , ncols = 1)

ax_1.boxplot(data.Intelligence) #, axis = ax_1)
ax_1.set_title('Intelligence')

ax_2.boxplot(data.Speed) #, axis = ax_2)
ax_2.set_title('Speed')

ax_3.boxplot(data.Power) #, axis = ax_3)
ax_3.set_title('Power')

plt.show()


