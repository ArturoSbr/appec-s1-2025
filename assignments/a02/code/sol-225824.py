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
table_reviews['happy'] = (table_reviews['review_score'] >= 4).astype(int)

# Q2. Keep only the last review of each order
table_reviews['review_answer_timestamp'] = pd.to_datetime(
    table_reviews['review_answer_timestamp'],
    format='%Y-%m-%d %H:%M:%S'
)
table_reviews.info()

# dejar solante la reseña más reciente por cada orden
table_reviews = (
    table_reviews
    .sort_values(['order_id', 'review_answer_timestamp'])
    .drop_duplicates(subset='order_id', keep='last')
)
table_reviews.info()

# Q3 Declare agg_items

agg_items = (
    table_items
    .groupby('order_id', as_index=False)
    .agg(
        n_items=('order_item_id', 'count'),
        avg_price=('price', 'mean'),
        avg_shipping=('freight_value', 'mean')
    )
)

agg_items.describe()

# Q4. Join table_reviews and agg_items to create df

df = pd.merge(
    left=table_reviews,
    right=agg_items,
    how='inner',
    on=['order_id']
)
df.info()

# Q5. Calculate days_delay in table_orders

# convertir a formato de fecha
table_orders['order_estimated_delivery_date'] = pd.to_datetime(
    table_orders['order_estimated_delivery_date'],
    format='%Y-%m-%d %H:%M:%S'
)

table_orders['order_delivered_customer_date'] = pd.to_datetime(
    table_orders['order_delivered_customer_date'],
    format='%Y-%m-%d %H:%M:%S'
)
table_orders.info()

# crear var days_delay

table_orders['days_delay'] = (
    table_orders['order_delivered_customer_date']
    - table_orders['order_estimated_delivery_date']
).dt.days

table_orders.describe()

# Q6. Join df and table_orders to add days_delay

df = pd.merge(
    left=df,
    right=table_orders[['order_id', 'order_status', 'days_delay']],
    how='inner',
    on=['order_id']
)
df['days_delay'].isna().sum()

# Q7. Join table_items and table_products to calculate avg_pics

median_photos = table_products['product_photos_qty'].median()
        table_products['product_photos_qty'] = \
        table_products['product_photos_qty'].fillna(median_photos)

table_products['product_photos_qty'].describe()

# Unir product_id' y 'product_photos_qty' a table_items

tmp_items = table_items.merge(
    table_products[['product_id', 'product_photos_qty']],
    on='product_id',
    how='inner'
)

# Añadir avg_pics por cada orden

agg_pics = (
    tmp_items
    .groupby('order_id', as_index=False)
    .agg(avg_pics=('product_photos_qty', 'mean'))
)

agg_pics.describe()

# Q8. Add avg_pics to df
df = pd.merge(
    left=df,
    right=agg_pics[['order_id', 'avg_pics']],
    how='inner',
    on=['order_id']
)

df.describe()

# Q9. Add 'const' to df
df['const'] = 1
df.describe()

# Q10. Fit model 1
df_model = df[df['order_status'] == 'delivered'].copy()
df_model.describe()

# seleccion de variables
cols = ['happy', 'const', 'n_items', 'avg_price', 'avg_shipping',
        'days_delay', 'avg_pics']
X = df_model[['const', 'n_items', 'avg_price', 'avg_shipping',
              'days_delay', 'avg_pics']]
y = df_model['happy']

# modelo
# Convert days_delay

m1 = sm.Logit(y, X, missing='drop')

m1_res = m1.fit()
m1_res.summary()

# Q11. Fit model 2

df_m2 = df[(df['order_status'] == 'delivered') & (df['n_items'] == 1)].copy()

X2 = df_m2[['const', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics']]
y2 = df_m2['happy']

m2 = sm.Logit(y2, X2, missing='drop')
m2_res = m2.fit()
m2_res.summary()
