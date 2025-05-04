#A/B Test
#The dataset is not shared because it's exclusive to Miuul Data Science Bootcamp

# Variables:
# total_bill: total price of the meal
# tip: tips
# sex: gender of the person paying the fee (0=male, 1=female)
# smoker: Is there anyone in the group who smokes? (0=No, 1=Yes)
# day: day (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: time? (0=Day, 1=Night)
# size: How many people are in the group?

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Confidence Intervals
############################

df = sns.load_dataset("tips")

df.describe().T

df.head()

sms.DescrStatsW(df["total_bill"]).tconfint_mean()

sms.DescrStatsW(df["tip"]).tconfint_mean()

######################################################
# Correlation
######################################################
df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")

plt.show()

df["tip"].corr(df["total_bill"])

######################################################
# AB Testing
######################################################

############################
#Is there any difference between the account values of smokers and non-smokers?
############################
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})

############################
# 1. Hypothesis
############################

# H0: M1 = M2 #no difference
# H1: M1 != M2

############################
# 2. Normality and Variance check
############################

############################
# Normality
############################
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

############################
# Variance
############################
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

############################
# 3 and 4. Hypothesis
############################

############################
#T test 
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

############################
#Mannwhitneyu test 
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
