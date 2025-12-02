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

# --- Data Cleaning and Preparation ---

# Q1. Create binary target
# We say a customer is "happy" with their order if they leave a review score of
# 4 or 5.
table_reviews['happy'] = (table_reviews['review_score'] >= 4).astype(int)

# Q2. Keep only the last review of each order
# Convert timestamp to datetime
table_reviews['review_answer_timestamp'] = pd.to_datetime(
    table_reviews['review_answer_timestamp'], format='%Y-%m-%d %H:%M:%S'
)
# Sort by timestamp (latest first) and keep the first unique order_id
table_reviews = table_reviews.sort_values(
    'review_answer_timestamp', ascending=False
).drop_duplicates(subset=['order_id'], keep='first')

# Q3 Declare agg_items
# Create a new dataframe that has the number of items, the average price,
# and the average freight value of the items in each order.
agg_items = table_items.groupby('order_id').agg(
    n_items=('order_item_id', 'count'),
    avg_price=('price', 'mean'),
    avg_shipping=('freight_value', 'mean')
).reset_index()

# Rename columns to match requirements
agg_items.columns = ['order_id', 'n_items', 'avg_price', 'avg_shipping']

# Q4. Join table_reviews and agg_items to create df
# Inner join between table_reviews and agg_items using order_id as join key.
df = table_reviews.merge(agg_items, on='order_id', how='inner')

# Q5. Calculate days_delay in table_orders
# Convert date columns to datetime
table_orders['order_estimated_delivery_date'] = pd.to_datetime(
    table_orders['order_estimated_delivery_date'], format='%Y-%m-%d %H:%M:%S'
)
table_orders['order_delivered_customer_date'] = pd.to_datetime(
    table_orders['order_delivered_customer_date'], format='%Y-%m-%d %H:%M:%S'
)
# Calculate the difference and extract the day part
table_orders['days_delay'] = (
    table_orders['order_delivered_customer_date']
    - table_orders['order_estimated_delivery_date']
).dt.days

# Q6. Join df and table_orders to add days_delay
# Inner join df with table_orders, only adding 'order_status' and 'days_delay'.
df = df.merge(
    table_orders[['order_id', 'order_status', 'days_delay']],
    on='order_id',
    how='inner'
)

# Q7. Join table_items and table_products to calculate avg_pics
# Fill null values of product_photos_qty with its median
median_photos = table_products['product_photos_qty'].median()
table_products['product_photos_qty'] = table_products[
    'product_photos_qty'
].fillna(median_photos)

# Add product_photos_qty to table_items
temp_items = table_items.merge(
    table_products[['product_id', 'product_photos_qty']],
    on='product_id',
    how='left'
)

# Calculate avg_pics per order
agg_pics = temp_items.groupby('order_id').agg(
    avg_pics=('product_photos_qty', 'mean')
).reset_index()

# Q8. Add avg_pics to df
# Inner join df and agg_pics on order_id
df = df.merge(agg_pics, on='order_id', how='inner')

# Q9. Add 'const' to df
df['const'] = 1

# --- Modeling ---

# Filter for delivered orders
df_delivered = df[df['order_status'] == 'delivered'].copy()

# Q10. Fit model 1
# Endogenous: happy
# Exogenous: const, n_items, avg_price, avg_shipping, days_delay, avg_pics

Y1 = df_delivered['happy']
X1 = df_delivered[[
    'const', 'n_items', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics'
]]

m1 = sm.Logit(Y1, X1, missing='drop')
m1_res = m1.fit(disp=0)  # disp=0 suppresses fitting output

# Q11. Fit model 2
# Use delivered orders and exclude orders with multiple items (n_items == 1).
# Endogenous: happy
# Exogenous: const, avg_price, avg_shipping, days_delay and avg_pics

df_single_item = df_delivered[df_delivered['n_items'] == 1].copy()

Y2 = df_single_item['happy']
X2 = df_single_item[[
    'const', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics'
]]

m2 = sm.Logit(Y2, X2, missing='drop')
m2_res = m2.fit(disp=0)  # disp=0 suppresses fitting output

# You can optionally print the results to verify they ran:
# print("--- Model 1 Results ---")
# print(m1_res.summary())
# print("\n--- Model 2 Results ---")
# print(m2_res.summary())
