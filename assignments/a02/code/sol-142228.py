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
# Las instrucciones piden la columna 'happy', no 'target'
table_reviews['happy'] = table_reviews['review_score'].ge(4).astype(int)

# Q2. Keep only the last review of each order
table_reviews['review_answer_timestamp'] = pd.to_datetime(table_reviews['review_answer_timestamp'], format='%Y-%m-%d %H:%M:%S')
table_reviews = table_reviews.sort_values('review_answer_timestamp')
table_reviews = table_reviews.drop_duplicates(subset=['order_id'], keep='last')

# Q3 Declare agg_items
agg_items = (table_items.groupby('order_id').agg(n_items=('order_item_id', 'count'), avg_price=('price', 'mean'),
        avg_shipping=('freight_value', 'mean'),).reset_index())

# Q4. Join table_reviews and agg_items to create df
df = pd.merge(table_reviews, agg_items, on='order_id', how='inner')

# Q5. Calculate days_delay in table_orders
table_orders['order_estimated_delivery_date'] = pd.to_datetime(table_orders['order_estimated_delivery_date'], format='%Y-%m-%d %H:%M:%S')
table_orders['order_delivered_customer_date'] = pd.to_datetime(table_orders['order_delivered_customer_date'], format='%Y-%m-%d %H:%M:%S')
table_orders['days_delay'] = (table_orders['order_delivered_customer_date'] - table_orders['order_estimated_delivery_date']).dt.days

# Q6. Join df and table_orders to add days_delay
df = pd.merge(df, table_orders[['order_id', 'order_status', 'days_delay']],
    on='order_id', how='inner',)

# Q7. Join table_items and table_products to calculate avg_pics
median_photos = table_products['product_photos_qty'].median()
table_products['product_photos_qty'] = table_products['product_photos_qty'].fillna(
    median_photos)

items_products = pd.merge(table_items, table_products, on='product_id', how='inner')
agg_pics = (items_products.groupby('order_id')['product_photos_qty'].mean().reset_index())
agg_pics.columns = ['order_id', 'avg_pics']

# Q8. Add avg_pics to df
df = pd.merge(df, agg_pics, on='order_id', how='inner')

# Q9. Add 'const' to df
df['const'] = 1

# Q10. Fit model 1
mask_delivered = df['order_status'] == 'delivered'
cols_m1 = ['const', 'n_items', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics']

m1 = sm.Logit(df.loc[mask_delivered, 'happy'],df.loc[mask_delivered, cols_m1], missing='drop',)
m1_res = m1.fit()

# Q11. Fit model 2
mask_m2 = (df['order_status'] == 'delivered') & (df['n_items'] == 1)
cols_m2 = ['const', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics']
m2 = sm.Logit(df.loc[mask_m2, 'happy'],df.loc[mask_m2, cols_m2],missing='drop')
m2_res = m2.fit()