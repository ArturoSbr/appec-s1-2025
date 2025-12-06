# Imports
import os
import pandas as pd
import statsmodels.api as sm

# Read tables (don't change anything here)
table_items = pd.read_csv(os.path.join("..", "data", "items.csv"))
table_orders = pd.read_csv(os.path.join("..", "data", "orders.csv"))
table_products = pd.read_csv(os.path.join("..", "data", "products.csv"))
table_reviews = pd.read_csv(os.path.join("..", "data", "reviews.csv"))

# Q1. Create binary target
# A customer is "happy" if review_score is 4 or 5.
table_reviews["happy"] = table_reviews["review_score"].isin([4, 5]).astype(int)

# Q2. Keep only the last review of each order
table_reviews["review_answer_timestamp"] = pd.to_datetime(
    table_reviews["review_answer_timestamp"],
    format="%Y-%m-%d %H:%M:%S",
)

idx_last_review = (
    table_reviews.groupby("order_id")["review_answer_timestamp"].idxmax()
)
table_reviews = table_reviews.loc[idx_last_review].reset_index(drop=True)

# Q3 Declare agg_items
# Aggregate items per order: number of items, average price and average shipping.
agg_items = (
    table_items.groupby("order_id")
    .agg(
        n_items=("order_item_id", "size"),
        avg_price=("price", "mean"),
        avg_shipping=("freight_value", "mean"),
    )
    .reset_index()
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

# Q6. Join df and table_orders to add days_delay (and order_status only)
df = df.merge(
    table_orders[["order_id", "order_status", "days_delay"]],
    on="order_id",
    how="inner",
)

# Q7. Join table_items and table_products to calculate avg_pics
# First, fill missing product_photos_qty with its median.
median_photos = table_products["product_photos_qty"].median()
table_products["product_photos_qty"] = table_products["product_photos_qty"].fillna(
    median_photos
)

# Add product_photos_qty to items and compute average photos per order.
items_with_photos = table_items.merge(
    table_products[["product_id", "product_photos_qty"]],
    on="product_id",
    how="left",
)

agg_pics = (
    items_with_photos.groupby("order_id")
    .agg(avg_pics=("product_photos_qty", "mean"))
    .reset_index()
)

# Q8. Add avg_pics to df
df = df.merge(agg_pics, on="order_id", how="inner")

# Q9. Add 'const' to df
df["const"] = 1.0

# Q10. Fit model 1
# Use only delivered orders.
delivered = df[df["order_status"] == "delivered"].copy()

exog_cols_m1 = ["const", "n_items", "avg_price", "avg_shipping", "days_delay", "avg_pics"]
m1 = sm.Logit(
    delivered["happy"],
    delivered[exog_cols_m1],
    missing="drop",
)
m1_res = m1.fit(disp=False)

# Q11. Fit model 2
# Delivered orders with a single item only (robustness check).
single_item = delivered[delivered["n_items"] == 1].copy()

exog_cols_m2 = ["const", "avg_price", "avg_shipping", "days_delay", "avg_pics"]
m2 = sm.Logit(
    single_item["happy"],
    single_item[exog_cols_m2],
    missing="drop",
)
m2_res = m2.fit(disp=False)
