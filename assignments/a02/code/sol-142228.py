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
table_reviews['happy'] = table_reviews ['review_score'].gt(3).astype(int)
# Q2. Keep only the last review of each order
table_reviews['review_creation_date'] = pd.to_datetime(table_reviews['review_creation_date'])
table_reviews = table_reviews.sort_values('review_creation_date').drop_duplicates(subset=['order_id'], keep='last')
# Q3 Declare agg_items
agg_items = table_items.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'})
# Q4. Join table_reviews and agg_items to create df
df = table_reviews.merge(agg_items, on='order_id')
# Q5. Calculate days_delay in table_orders
table_orders['order_delivered_customer_date'] = pd.to_datetime(table_orders['order_delivered_customer_date'])
table_orders['order_estimated_delivery_date'] = pd.to_datetime(table_orders['order_estimated_delivery_date'])
# Q6. Join df and table_orders to add days_delay
df = df.merge(table_orders[['order_id', 'days_delay']], on='order_id', how='inner')

# Q7. Join table_items and table_products to calculate avg_pics
items_products = table_items.merge(table_products[['product_id', 'product_photos_qty']], on='product_id')

# Q8. Add avg_pics to df
df = df.merge(avg_pics.reset_index(name='avg_pics'), on='order_id')

# Q9. Add 'const' to df
df['const'] = 1
# Q10. Fit model 1
model_1 = sm.Logit(df_clean['happy'], df_clean[['const', 'days_delay']]).fit()
print("--- Resultado Modelo 1 ---")
print(model_1.summary())
# Q11. Fit model 2
model_2 = sm.Logit(df_clean['happy'], df_clean[['const', 'days_delay', 'avg_pics']]).fit()
print("\n--- Resultado Modelo 2 ---")
print(model_2.summary())