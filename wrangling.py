import numpy as np
import pandas as pd

tests_df = pd.read_csv('Covid-19_Tests.csv')

# Exploration
# How many rows and columns are there?
print('Shape:\n', tests_df.shape)

# Data type per column, memory usage
print(tests_df.info())

# Inspect the start and end of the data
print('\nFirst five rows:\n', tests_df.head())
print('\nLast five rows:\n', tests_df.tail())

# Basic statistics
print('\nStats:\n', tests_df.describe())


# Select, Filter, Sort
# Find rows with above average positive percentage
above_avg = tests_df[tests_df['pct'] > tests_df['pct'].mean()]
print('\nAbove avg positive percentage days:\n', above_avg.iloc[:, 0:3])

# Find rows with over 500 positive tests
above_500 = tests_df.loc[tests_df['pos'] > 500, ['specimen_collection_date', 'tests', 'pos', 'pct']]
print('\nOver 500 positive tests:\n', above_500.iloc[:, 0:3])

# Select 30 rows with the highest amount of indeterminate tests
top_indeterminate = tests_df.sort_values(by='indeterminate').tail(30)
print('\nRows with highest number of indeterminate tests:\n', top_indeterminate.iloc[:, [0, 1, 5]])

# Find rows with over 10000 tests taken
above_10000 = tests_df[tests_df['tests'] > 10000]
print('\nOver 10000 tests taken:\n', above_10000.iloc[:, [0, 1, 3]])

# Sort days with most tests taken by percentage of positive tests - highest at top
print('\n10000 tests taken, sorted:\n', above_10000.sort_values(by='pct', ascending=False).iloc[:, 0:4])

# Sort rows by percentage of positive tests - highest at top
tests_df.sort_values(by='pct', inplace=True, ascending=False)
print('\nSorting all by positive tests:\n', tests_df.iloc[:, 0:4])


# Clean data
# Check if any columns have null values
print('\nColumns with null values:\n', tests_df.isnull().any())

# Alternatively, check if any rows have null values
print('\nRows with null values:\n', tests_df.isnull().any(axis=1))
print('\nRows with null values:\n', tests_df.isnull().any(axis=1).value_counts())

# Check for duplicates
print('\nRows that are duplicates:\n', tests_df.duplicated().value_counts())

# Determine interquartile range
percentages = tests_df['pct'].sort_index()
q1 = percentages.quantile(0.25)
q3 = percentages.quantile(0.75)
iqr = q3 - q1

# Calculate 1.5iqr +/- quartiles for pct
high_bound = q3 + (1.5 * iqr)
low_bound = q1 - (1.5 * iqr)

# Determine outliers with respect to pct and replace with null values
nwp = percentages.where(percentages.between(low_bound, high_bound))
compare = pd.DataFrame({'before': percentages, 'after': nwp})
print('\nComparing percentages with and without outliers included:\n')
print(compare.describe())

# Fix index of DataFrame
tests_df = tests_df.reset_index(drop=True)
print('\nDataFrame with index reset:\n', tests_df[['specimen_collection_date', 'tests', 'pos', 'pct']].head(10))


# Transformations
# Group rows by number of positive tests
positive_groups = tests_df.groupby('pos')
print('\nGroups based on number of positive tests:\n', positive_groups.describe())

# Add column for negative pct
tests_df['neg_pct'] = tests_df['neg']/tests_df['tests']
print('\nNegative percentage column:\n', tests_df[['specimen_collection_date', 'tests', 'neg', 'neg_pct']].head(10))
