"""This file will help you get started with Assignment 2.

You don't need any other dependencies, and you also don't need to read any other
tables. Just remember to go to your own branch, copy this file, rename it
`sol-<student ID>.py`, modify it accordingly and then push it (again, to YOUR
branch). Your results will be displayed in the repo's "Actions" tab.

Also, remember to lint your code like I taught you.
"""

# Imports
import os
import pandas as pd
import statsmodels.api as sm

# Read tables (don't change anything here)
table_items = pd.read_csv(os.path.join('..', 'data', 'items.csv'))
table_orders = pd.read_csv(os.path.join('..', 'data', 'orders.csv'))
table_products = pd.read_csv(os.path.join('..', 'data', 'products.csv'))
table_reviews = pd.read_csv(os.path.join('..', 'data', 'reviews.csv'))

# Q1. Create binary target
table_reviews['happy'] = table_reviews['review_score'].gt(3).astype(int)

# Q2. Keep only the last review of each order
table_reviews = table_reviews.sort_values('review_date').drop_duplicates('order_id', keep='last')

# Q3 Declare agg_items
agg_items = (
    table_items
    .groupby('order_id')
    .agg(
        n_items=('item_id', 'count'),
        total_value=('price', 'sum'),
    )
    .reset_index()
)
# Q4. Join table_reviews and agg_items to create df
df = pd.merge(
    table_reviews[['order_id', 'target']],
    agg_items,
    on='order_id',
    how='left',
)

# Q5. Calculate days_delay in table_orders
table_orders['order_date'] = pd.to_datetime(table_orders['order_date'])
table_orders['delivery_date'] = pd.to_datetime(table_orders['delivery_date'])
table_orders['days_delay'] = (table_orders['delivery_date'] - table_orders['order_date']).dt.days       

# Q6. Join df and table_orders to add days_delay
df = pd.merge(
    df,
    table_orders[['order_id', 'days_delay']],
    on='order_id',
    how='left',
)

# Q7. Join table_items and table_products to calculate avg_pics
items_products = pd.merge(
    table_items,
    table_products[['product_id', 'n_pictures']],
    on='product_id',
    how='left',
)   
avg_pics = (
    items_products
    .groupby('order_id')
    .agg(avg_pics=('n_pictures', 'mean'))
    .reset_index()
)
df = pd.merge(
    df,
    avg_pics,
    on='order_id',
    how='left',
)       
# Q8. Add avg_pics to df
df = pd.merge(
    df,
    avg_pics,
    on='order_id',
    how='left',
)   

# Q9. Add 'const' to df
df['const'] = 1

# Q10. Fit model 1
model1 = sm.Logit(
    df['target'],
    df[['const', 'n_items', 'total_value', 'days_delay']],
).fit(disp=0)

# Q11. Fit model 2
model2 = sm.Logit(
    df['target'],
    df[['const', 'n_items', 'total_value', 'days_delay', 'avg_pics']],
).fit(disp=0)   

# Q12. Compare models
lr_stat = 2 * (model2.llf - model1.llf)
p_value = 1 - sm.stats.chi2.cdf(lr_stat, df=1)  # df=1 because only one additional parameter in model2     
print(f'Likelihood Ratio Statistic: {lr_stat}')
print(f'P-value: {p_value}')
