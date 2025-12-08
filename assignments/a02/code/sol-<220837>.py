# Imports
import os
import pandas as pd
import statsmodels.api as sm

# Read tables (updated paths)
DATA_DIR = os.path.join("assignments", "a02", "data")

table_items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
table_orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
table_products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
table_reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

# Q1. Create binary target
table_reviews["happy"] = (table_reviews["review_score"] >= 4).astype(int)

# Q2. Keep only the last review of each order
table_reviews["review_answer_timestamp"] = pd.to_datetime(
    table_reviews["review_answer_timestamp"],
    format="%Y-%m-%d %H:%M:%S",
)

idx_last = table_reviews.groupby("order_id")["review_answer_timestamp"].idxmax()
table_reviews = table_reviews.loc[idx_last].reset_index(drop=True)

# Q3 Declare agg_items
agg_items = (
    table_items
    .groupby("order_id", as_index=False)
    .agg(
        n_items=("order_item_id", "count"),
        avg_price=("price", "mean"),
        avg_shipping=("freight_value", "mean"),
    )
)

# Q4. Join table_reviews and agg_items to create df
df = table_reviews.merge(agg_items, on="order_id", how="inner")

# Q5. Calculate days_delay in table_orders
table_orders["order_estimated_delivery_date"] = pd.to_datetime(
    table_orders["order_estimated_delivery_date"],
    format="%Y-%m-%d %H:%M:%S",
)

table_orders["order_delivered_customer_date"] = pd.to_datetime(
    table_orders["order_delivered_customer_date"],
    format="%Y-%m-%d %H:%M:%S",
)

table_orders["days_delay"] = (
    table_orders["order_delivered_customer_date"]
    - table_orders["order_estimated_delivery_date"]
).dt.days

# Q6. Join df and table_orders to add days_delay
df = df.merge(
    table_orders[["order_id", "order_status", "days_delay"]],
    on="order_id",
    how="inner",
)

# Q7. Join table_items and table_products to calculate avg_pics
median_photos = table_products["product_photos_qty"].median()
table_products["product_photos_qty"] = table_products["product_photos_qty"].fillna(
    median_photos
)

table_items = table_items.merge(
    table_products[["product_id", "product_photos_qty"]],
    on="product_id",
    how="left",
)

agg_pics = (
    table_items
    .groupby("order_id", as_index=False)
    .agg(avg_pics=("product_photos_qty", "mean"))
)

# Q8. Add avg_pics to df
df = df.merge(agg_pics, on="order_id", how="inner")

# Q9. Add 'const' to df
df["const"] = 1.0

# Q10. Fit model 1
df_delivered = df[df["order_status"] == "delivered"].copy()

y1 = df_delivered["happy"]
X1 = df_delivered[
    ["const", "n_items", "avg_price", "avg_shipping", "days_delay", "avg_pics"]
]

m1 = sm.Logit(y1, X1, missing="drop")
m1_res = m1.fit()

# Q11. Fit model 2
df_single = df_delivered[df_delivered["n_items"] == 1].copy()

y2 = df_single["happy"]
X2 = df_single[["const", "avg_price", "avg_shipping", "days_delay", "avg_pics"]]

m2 = sm.Logit(y2, X2, missing="drop")
m2_res = m2.fit()
