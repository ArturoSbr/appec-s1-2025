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
# A customer is "happy" if review_score is 4 or 5.
table_reviews["happy"] = (table_reviews["review_score"] >= 4).astype(int)

# Q2. Keep only the last review of each order
# Convert timestamp and keep the latest review per order_id.
table_reviews["review_answer_timestamp"] = pd.to_datetime(
    table_reviews["review_answer_timestamp"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)

idx_latest = table_reviews.groupby("order_id")["review_answer_timestamp"].idxmax()
table_reviews = table_reviews.loc[idx_latest].copy()

# Q3 Declare agg_items
# Aggregate items per order: number of items, average price, average freight.
agg_items = (
    table_items.groupby("order_id")
    .agg(
        n_items=("order_item_id", "count"),
        avg_price=("price", "mean"),
        avg_shipping=("freight_value", "mean"),
    )
    .reset_index()
)

# Q4. Join table_reviews and agg_items to create df
# Inner join on order_id to keep only orders present in both tables.
df = table_reviews.merge(agg_items, on="order_id", how="inner")
# Q5. Calculate days_delay in table_orders
# Convert to datetime and compute delivered - estimated in days.
table_orders["order_estimated_delivery_date"] = pd.to_datetime(
    table_orders["order_estimated_delivery_date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)

table_orders["order_delivered_customer_date"] = pd.to_datetime(
    table_orders["order_delivered_customer_date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)

table_orders["days_delay"] = (
    table_orders["order_delivered_customer_date"]
    - table_orders["order_estimated_delivery_date"]
).dt.days

# Q6. Join df and table_orders to add days_delay
# Only add order_status and days_delay.
orders_subset = table_orders[["order_id", "order_status", "days_delay"]]

df = df.merge(orders_subset, on="order_id", how="inner")

# Q7. Join table_items and table_products to calculate avg_pics
# Fill missing product_photos_qty with its median and compute avg pictures per order.
median_photos = table_products["product_photos_qty"].median()
table_products["product_photos_qty"] = table_products["product_photos_qty"].fillna(
    median_photos,
)

items_with_pics = table_items.merge(
    table_products[["product_id", "product_photos_qty"]],
    on="product_id",
    how="left",
)

agg_pics = (
    items_with_pics.groupby("order_id")["product_photos_qty"]
    .mean()
    .reset_index()
    .rename(columns={"product_photos_qty": "avg_pics"})
)

# Q8. Add avg_pics to df
# Inner join to keep only orders present in both df and agg_pics.
df = df.merge(agg_pics, on="order_id", how="inner")

# Q9. Add 'const' to df
# Constant term for the intercept in the logistic models.
df["const"] = 1.0

# Q10. Fit model 1
# Use only delivered orders. Endog: happy. Exog:
# const, n_items, avg_price, avg_shipping, days_delay, avg_pics.
df_m1 = df.loc[df["order_status"] == "delivered"].copy()

y_m1 = df_m1["happy"]
X_m1 = df_m1[
    ["const", "n_items", "avg_price", "avg_shipping", "days_delay", "avg_pics"]
]

m1 = sm.Logit(y_m1, X_m1, missing="drop")
m1_res = m1.fit(disp=0)

# Q11. Fit model 2
# Only delivered orders with a single item.
# Endog: happy. Exog: const, avg_price, avg_shipping, days_delay, avg_pics.
df_m2 = df_m1.loc[df_m1["n_items"] == 1].copy()

y_m2 = df_m2["happy"]
X_m2 = df_m2[["const", "avg_price", "avg_shipping", "days_delay", "avg_pics"]]

m2 = sm.Logit(y_m2, X_m2, missing="drop")
m2_res = m2.fit(disp=0)