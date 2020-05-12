# imports
import numpy as np 
import pandas as pd 
import collections

import matplotlib.pyplot as plt 
import seaborn as sns

# load datasets

##TODO parent directory
q1_data = pd.read_csv('/Users/Tavo/code/moloco/molocopower/data/q1_data.csv')
q2_data = pd.read_csv('/Users/Tavo/code/moloco/molocopower/data/q2_data.csv')

# exploratory q1
q1_data.shape
q1_data.isna().sum()
q1_data.describe()

# to diminish suspicions of faulty data, let's at least
# check that all user_id's look similar (simple check: same number of characters)
# this check can bemade more complex if there are worries of data integrity being compromised
np.mean([len(u) for u in q1_data.user_id])

#################
################
###############
#############
# question 1
'''
Consider only the rows with country_id = "BDV" 
(there are 844 such rows). For each site_id, we can 
compute the number of unique user_id's found in these
 844 rows. Which site_id has the largest number of 
 unique users? And what's the number?
 '''

q1_bdv = q1_data[q1_data.country_id == 'BDV']
sites_bdv = q1_bdv.site_id.unique()

# as an overkill but to demonstrate coding skills, 
# I will prepare a couple of functions to answer this question

def get_number_of_unique_users_in_country(country):
    '''
    This function gets the code of a country and 
    outputs a list of sites and a corresponding list of nuber fo unique users.
    
    Parameters:
    country (str): the code of the desired coutnry to analyse

    Returns:
    sites_country (str): list of the site codes found in the country
    n_unique_users (int): list of the number of unique users fo each site
    '''

    # get country data
    q1_country = q1_data[q1_data.country_id == country]
    
    # get sites in country
    sites_country = list(q1_country.site_id.unique())
    
    # get number of unique user per site
    n_unique_users = []
    for s in sites_country:
        print(s)
        n = q1_country[q1_country['site_id'] == s].user_id.nunique()
        n_unique_users.append(n)
    # let's keep it as lists as they preserve order
    return sites_country, n_unique_users

def get_site_with_max_unique_users(list_str, list_int):
    '''
    This function gets two lists, one of which is numerical,
    and outputs the corresponding value in the non-numerical list.
    It assumes there is order preserved and matching in both lists.

    Parameters:
    list_str (str): a list of strings (data type not enforced)
    list_int (int): a list of int or float

    Returns:
    site_with_largest_unique_users (str): the value in list_str corresponding to the max value in list_int
    '''

    # index of the max number of unique users
    idx_max_n_users = np.argmax(list_int)
    n_unique_users_for_site = np.max(list_int)
    # extract the site code
    site_with_largest_unique_users = list_str[idx_max_n_users]

    return site_with_largest_unique_users, n_unique_users_for_site

# apply
country = 'BDV'
list_sites, list_unique_users = get_number_of_unique_users_in_country('BDV')
site_with_largest_n_unique_users, max_unique_users = get_site_with_max_unique_users(list_sites, list_unique_users)

print("The site in the country {0} with the largest number of unique users ({1} unique users) is: {2}".format(country, max_unique_users, site_with_largest_n_unique_users))

# answer
# 5NPAU with 544 unique users

#################
################
###############
#############
# question 2
 
'''Between 2019-02-03 00:00:00 and 2019-02-04 23:59:59, 
there are four users who visited a certain site 
more than 10 times. Find these four users & which sites 
they (each) visited more than 10 times. 
(Simply provides four triples in the form 
(user_id, site_id, number of visits) in the box below.)'''

# check for datetime format
q1_data.dtypes

# apply datetime format, it seems it is a standard format
# YYYY-MM-DD HH:MM:SS, so we do not need too much manipulation
q1_data.ts = pd.to_datetime(q1_data.ts)

# let's make ts the index -- that way we can use some pandas functionalities
# very handy!
q1_data = q1_data.set_index('ts')
from_ts = pd.to_datetime('2019-02-03 00:00:00')
until_ts = pd.to_datetime('2019-02-04 23:59:59')

# for transparency, let's create a count column
q1_data.loc[:, 'count_col'] = 1

results_visits = q1_data.loc[from_ts:until_ts].groupby(['user_id', 'site_id']).sum().sort_values('count_col').tail(4)

# let's product the tuples of (user_id, site_id, number_of_visits)
list_of_tuples = []
for i, r in zip(list(results_visits.index), list(results_visits.count_col.values)):
    j = list(i)
    j.append(r)
    j = tuple(j)
    list_of_tuples.append(j)
    print(j)

list_of_tuples

# answer
#[('LC3C7E', '3POLC', 15), 
# ('LC3C9D', 'N0OTG', 17), 
# ('LC06C3', 'N0OTG', 25), 
# ('LC3A59', 'N0OTG', 26)]

#################
################
###############
#############
# question 3

'''For each site, compute the unique number of users 
whose last visit (found in the original data set) 
was to that site. For instance, user "LC3561"'s 
last visit is to "N0OTG" based on timestamp data. 
Based on this measure, what are top three sites? 
(hint: site "3POLC" is ranked at 5th with 28 users 
whose last visit in the data set was to 3POLC; 
simply provide three pairs in the form 
(site_id, number of users).)'''

# first, calculate the last visit of each user
# ...for each unique user
def calculate_first_last_visits(df):
    '''this function extracts the first 
    and last visit site for each user.

    Parameters:
    df (DataFrame): dataframe with timestamp-user_id-site_id fore ach visit

    Returns:
    users: list of unique users
    first_visits: list of first visit for each unique user
    last_visits: list of last visits for each unique user
    '''

    users = []
    first_visits = []
    last_visits = []
    for u in df.user_id.unique():
        users.append(u)
        # ...compute site first visited
        first_site = df[df.user_id == u].sort_index(ascending=False).iloc[-1].site_id
        first_visits.append(first_site)
        # ...compute site last visited
        last_site = df[df.user_id == u].sort_index(ascending=False).iloc[0].site_id
        last_visits.append(last_site)
    # note, this can be also made with a groupby method using a max(timestamp) as aggregating function
    return users, first_visits, last_visits

users, first_visits, last_visits = calculate_first_last_visits(q1_data)
# we can make it pretty and put it in a table (dataframe)
user_last_visits = pd.DataFrame()
user_last_visits.loc[:, 'user_id'] = users
user_last_visits.loc[:, 'last_site'] = last_visits

# and we can group by site and get the frequency
top_three = user_last_visits.groupby('last_site').count().sort_values('user_id').tail(3)
# and we can finally extract the results as tuples, as requested in teh exercise
print(list(zip(list(top_three.index), list(top_three.user_id.values))))

# answer
#[('QGO3G', 289), ('N0OTG', 561), ('5NPAU', 992)]

#################
################
###############
#############
# question 3

'''For each user, determine the first site he/she visited
 and the last site he/she visited based on the timestamp 
 data. Compute the number of users whose first/last 
 visits are to the same website. What is the number?'''

# we already have the function that will give us first and last site visited per user
# for clarity's sake, let's repeat that step here...
users, first_visits, last_visits = calculate_first_last_visits(q1_data)

user_visits = pd.DataFrame()
user_visits.loc[:, 'user_id'] = users
user_visits.loc[:, 'first_site'] = first_visits
user_visits.loc[:, 'last_site'] = last_visits

# ===> interpretation 1 (discarded): 
# to calculate all pairs of first/last site pairs are the same
# for this I will exclude the pairs that do not repeat
 
# to calculate groups of users with common first and last sites, 
# we can group by both first and last sites
interpretation_1 = user_visits.groupby(['first_site', 'last_site']).count().reset_index()
# from teh above, we can count the number of counts >1 
# and the remainig number of rows is the answer of this interpretation.
# However, I discarded this option after carefully reading the question again.

# ===> interpretation 2 (most likely interpretation):
# calculate all users that start AND end in the SAME site
first_last_same = [1 if f == l else 0 for f,l in zip(user_visits.first_site, user_visits.last_site)]

# we can add it to the dataframe for QA
user_visits.loc[:, 'first_last_same'] = first_last_same

print("The total number of users with first and last site being the same is: ", sum(first_last_same))

# answer: 1670 users




