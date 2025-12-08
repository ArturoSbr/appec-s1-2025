# Import the package
import os
import pandas as pd
import statsmodels.api as sm

# Loading .csv file
table_items = pd.read_csv(os.path.join('..', 'data', 'items.csv'))
table_orders = pd.read_csv(os.path.join('..', 'data', 'orders.csv'))
table_products = pd.read_csv(os.path.join('..', 'data', 'products.csv'))
table_reviews = pd.read_csv(os.path.join('..', 'data', 'reviews.csv'))

#1. We say a customer is "happy" with their order if they leave a review score
#of 4 or 5. Assign a new column to `table_reviews` named `'happy'` that encodes
#this logic (1 = happy, 0 otherwise).

table_reviews['happy'] = (table_reviews['review_score'] >= 4).astype(int)
table_reviews.head()

#2. Note that there's more reviews than there are order IDs. This happens
#when a user changes their thoughts and updates their review. You need to update
#`table_reviews` so that only the latest review of each order is preserved.
#Convert column `review_answer_timestamp` to datetime using `pd.to_datetime` and
#then use it to get the latest review for each order (hint: use
#`format='%Y-%m-%d %H:%M:%S'` when casting the column to datetime format).

#convertir a formato de fecha
table_reviews['review_answer_timestamp'] = pd.to_datetime(
    table_reviews['review_answer_timestamp'],
    format='%Y-%m-%d %H:%M:%S'
)
table_reviews.info()

#dejar solante la reseña más reciente por cada orden
table_reviews = (
    table_reviews
    .sort_values(['order_id', 'review_answer_timestamp'])
    .drop_duplicates(subset='order_id', keep='last')
)
table_reviews.info()

#3. As you know, a single order may include multiple items. A peculiar thing
#about Olist is that their user satisfaction survey does not allow customers to
#review items individually. Instead, it only allows them to review the order as
#a whole. This means we need to aggregate orders with multiple items to get a
#1:1 relationship of rows between tables. For example, if an order has three
#products, it will show up three times in the items table, but the order only
#has one row in the reviews table! There's many different ways to go about this,
#but we will "compress" this information into a single row using aggregation
#functions.
#Use `table_items` to create a new dataframe that has the number of items, the
#average price, and the average freight value of the items in each order.
#The new dataframe must be named `agg_items`, and its columns should be named
#`'order_id'`, `'n_items'`, `'avg_price'` and `'avg_shipping'`.

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

#4. Create a new dataframe named `df`. This will be our primary input for all
#the models we will run later. Define this new table as the inner join between
#`table_reviews` and `agg_items` using `order_id` as join key.
#You may have noticed that `df` has less rows than `table_reviews`. This means
#there are orders that do not appear in the items table but do appear in the
#reviews table! This happens because `table_items` only shows items that were
#successfully delivered. Hence, this mismatch in rows might happen when a
#customer cancels the order before receiving it but still leaves a review.
#97,917

df = pd.merge(
    left= table_reviews,
    right= agg_items,
    how= 'inner',
    on= ['order_id']
)
df.info()

#5. It seems reasonable to think that an early or late delivery will impact the
#customer's review. To test this, we will calculate the number of days that
#an order was delayed. First, take `table_orders` and transform columns
#`'order_estimated_delivery_date'` and `'order_delivered_customer_date'` to
#DateTime format using `pd.to_datetime` (hint: use `format='%Y-%m-%d %H:%M:%S'`
#to parse the strings). Then create a new column called `days_delay` by
#subtracting the estimated delivery date from the customer delivery date
#(if it's negative, then it arrived early, and if it's positive,
#it arrived late) and only keep its `day` part
#(hint: access the new column's `.dt.days` attribute)

#convertir a formato de fecha
table_orders['order_estimated_delivery_date'] = pd.to_datetime(
    table_orders['order_estimated_delivery_date'],
    format = '%Y-%m-%d %H:%M:%S'
)

table_orders['order_delivered_customer_date'] = pd.to_datetime(
    table_orders['order_delivered_customer_date'],
    format = '%Y-%m-%d %H:%M:%S'
)
table_orders.info()

#crear var days_delay
#96,476
#means days_delay -11.877

table_orders['days_delay'] = (
    table_orders['order_delivered_customer_date']
    - table_orders['order_estimated_delivery_date']
).dt.days

table_orders.describe()

#6. Inner join `df` with `table_orders` using `order_id` as join key and make
#sure you're only adding columns `order_status`, `days_delay` to `df` (don't add
#any other columns from `table_orders`).
#Nan = 2,087
df = pd.merge(
    left= df,
    right= table_orders[['order_id', 'order_status', 'days_delay']],
    how= 'inner',
    on= ['order_id']
)
df['days_delay'].isna().sum()

#7. It seems reasonable to think that if a listing has many photos, the
#customer will be able to make a more informed decision when buying that product
#and would therefore be less likely to leave a negative review (because the
#photos would arguably serve as a proxy for the product's quality and features).
#First fill the null values of column `table_products['product_photos_qty']`
#with its own median value. Then, add column `product_photos_qty` to
#`table_items` by joining it with `table_products` on `product_id`. Finally,
#calculate column `avg_pics` as the average number of photos per order and
#store this aggregated dataset in a dataframe called `agg_pics`.
#This frame's only columns must be `order_id` and `avg_pics`.

median_photos = table_products['product_photos_qty'].median()
table_products['product_photos_qty'] = \
table_products['product_photos_qty'].fillna(median_photos)

table_products['product_photos_qty'].describe()

# Unir product_id' y 'product_photos_qty' a table_items
#98,666
tmp_items = table_items.merge(
    table_products[['product_id', 'product_photos_qty']],
    on='product_id',
    how='inner'
)

# Añadir avg_pics por cada orden
#2.232
agg_pics = (
    tmp_items
    .groupby('order_id', as_index=False)
    .agg(avg_pics=('product_photos_qty', 'mean'))
)

agg_pics.describe()

#8. The whole point of calculating `'avg_pics` is to use it as a feature in
#our models, so inner join `df` and `agg_pics` on `order_id` (do not add any
#other columns).
df = pd.merge(
    left= df,
    right= agg_pics[['order_id', 'avg_pics']],
    how= 'inner',
    on= ['order_id']
)
df.describe()

#9. Add a column full of ones named `'const'` to `df`. This is my gift to you.
df['const'] = 1
df.describe()

#10. Fit a logistic regression model to the final form of `df`.
#Make sure to only use delivered orders (filter `'order_status'`).
#The endogenous variable must be *happy*, and the exogenous variables must
#be *const, n_items, avg_price, avg_shipping, days_delay and avg_pics*.
#Store the model in `m1` and the fitted model in `m1_res`.
#Set `missing='drop'` to drop rows that have at least one null
#value in any of their columns.
#95,824
#nos quedamos solo con las ordenes entregadas
df = df[df['order_status'] == 'delivered'].copy()
df.describe()

#selección de variables
cols = ['happy', 'const', 'n_items', 'avg_price', 'avg_shipping',
        'days_delay', 'avg_pics']
df_model = df[cols]

#modelo
# Convert days_delay from Timedelta to numeric days

m1 = sm.Logit(
    df_model['happy'],
    df_model[['const', 'n_items', 'avg_price', 'avg_shipping',
              'days_delay', 'avg_pics']],
    missing='drop'
)

m1_res = m1.fit()
m1_res.summary()

#m1_res.nobs == 95824
#params['const']      ≈ 1.314973
#params['n_items']    ≈ -0.498888
#params['days_delay'] ≈ -0.063472

#11. Now fit a logistic regression model to the final form of `df`. Like before,
#use delivered orders, and this time, exclude orders with multiple items. This
#way, aggregated columns such as `'avg_price'` now represent the actual price of
#the only product in the order. This is just a robustness test to check
#if orders with more than one order are skewing our results.
#The endogenous variable must be *happy*, and the exogenous variables must be
#*const, avg_price, avg_shipping, days_delay and avg_pics*.
#We're dropping *n_items* because it is now full of ones.
#Store the model in `m2` and the fitted values in `m2_res`.
#Use `missing='drop'` again to exclude observations with null values.

df_m2 = df[(df['order_status'] == 'delivered') & (df['n_items'] == 1)].copy()


#elección de variables
m2 = sm.Logit(
    df_m2['happy'],
    df_m2[['const', 'avg_price', 'avg_shipping',
              'days_delay', 'avg_pics']],
    missing='drop'
)

m2_res = m2.fit()
m2_res.summary()
#m2_res.nobs == 86296
#coef const      ≈ 0.769101
#coef avg_price  ≈ 0.000106
#coef days_delay ≈ -0.071962
