"""Assignment 2 – Olist happiness and logistic regression."""

import os
import pandas as pd
import statsmodels.api as sm

# Read tables (don't change anything here)
table_items = pd.read_csv(os.path.join("..", "data", "items.csv"))
table_orders = pd.read_csv(os.path.join("..", "data", "orders.csv"))
table_products = pd.read_csv(os.path.join("..", "data", "products.csv"))
table_reviews = pd.read_csv(os.path.join("..", "data", "reviews.csv"))

# Q1. Create binary target
table_reviews["happy"] = table_reviews["review_score"].ge(4).astype(int)

# Q2. Keep only the last review of each order
table_reviews["review_answer_timestamp"] = pd.to_datetime(
    table_reviews["review_answer_timestamp"],
    format="%Y-%m-%d %H:%M:%S",
)

table_reviews = (
    table_reviews.sort_values("review_answer_timestamp")
    .groupby("order_id", as_index=False)
    .last()
)

# Q3. Declare agg_items
agg_items = (
    table_items.groupby("order_id", as_index=False)
    .agg(
        n_items=("order_item_id", "count"),
        avg_price=("price", "mean"),
        avg_shipping=("freight_value", "mean"),
    )
)

# Q4. Join table_reviews and agg_items to create df
df = table_reviews.merge(agg_items, on="order_id", how="inner")

# Q5. Calculate days_delay in table_orders
for col in [
    "order_estimated_delivery_date",
    "order_delivered_customer_date",
]:
    table_orders[col] = pd.to_datetime(
        table_orders[col],
        format="%Y-%m-%d %H:%M:%S",
    )

table_orders["days_delay"] = (
    table_orders["order_delivered_customer_date"]
    - table_orders["order_estimated_delivery_date"]
).dt.days

# Q6. Join df and table_orders to add days_delay and order_status
df = df.merge(
    table_orders[["order_id", "order_status", "days_delay"]],
    on="order_id",
    how="inner",
)

# Q7. Join table_items and table_products to calculate avg_pics
table_products["product_photos_qty"] = table_products["product_photos_qty"].fillna(
    table_products["product_photos_qty"].median(),
)

table_items = table_items.merge(
    table_products[["product_id", "product_photos_qty"]],
    on="product_id",
    how="left",
)

agg_pics = (
    table_items.groupby("order_id", as_index=False)
    .agg(avg_pics=("product_photos_qty", "mean"))
)

# Q8. Add avg_pics to df
df = df.merge(agg_pics, on="order_id", how="inner")

# Q9. Add 'const' to df
df["const"] = 1.0

# Q10. Fit model 1
df_delivered = df[df["order_status"] == "delivered"].copy()

X1 = df_delivered[
    ["const", "n_items", "avg_price", "avg_shipping", "days_delay", "avg_pics"]
]
y1 = df_delivered["happy"]

m1 = sm.Logit(y1, X1, missing="drop")
m1_res = m1.fit(disp=False)

# Q11. Fit model 2
df_single = df_delivered[df_delivered["n_items"] == 1].copy()

X2 = df_single[["const", "avg_price", "avg_shipping", "days_delay", "avg_pics"]]
y2 = df_single["happy"]

m2 = sm.Logit(y2, X2, missing="drop")
m2_res = m2.fit(disp=False)
