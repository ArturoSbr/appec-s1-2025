# Assignment 2

# Introduction

We will use the Olist e-commerce database for this assignment. Our goal is to
train a series of models that help us understand which factors play a role in
making a customer "happy".

The fun thing about this database is that the data is divided into multiple
tables. Additionally, each order can contain multiple items (ie, the customer
ordered three items in a single session), and it can also be reviewed multiple
times (ie, when a customer changes their mind and updates their review).

We will clean and join the data to create a final dataset that will allow us to
use logistic regression models to understand which factors impact the
probability of a customer leaving a positive review.

## Questions

1. We say a customer is "happy" with their order if they leave a review score of
4 or 5. Assign a new column to `table_reviews` named `'happy'` that encodes
this logic (1 = happy, 0 otherwise).

2. Note that there's more reviews than there are order IDs. This happens
when a user changes their thoughts and updates their review. You need to update
`table_reviews` so that only the latest review of each order is preserved. Use
column `'review_answer_timestamp'` to get the latest review for each order.

3. As you know, a single order may include multiple items. A peculiar thing
about Olist is that their user satisfaction survey does not allow customers to
review items individually. Instead, it only allows them to review the order as a
whole. This means we need to aggregate orders with multiple items to get a 1:1
relationship of rows between tables. For example, if an order has three
products, it will show up three times in the items table, but the order only has
one row in the reviews table! There's many different ways to go about this, but
we will "compress" this information into a single row using aggregation
functions.
Use `table_items` to create a new dataframe that has the number of items, the
average price, and the average freight value of the items in each order. The new
dataframe must be named `agg_items`, and its columns should be named
`'order_id'`, `'n_items'`, `'avg_price'` and `'avg_shipping'`.

4. Create a new dataframe named `df`. This will be our primary input for all the
models we will run later. Define this new table as the inner join between
`table_reviews` and `agg_items` using `order_id` as join key.
You may have noticed that `df` has less rows than `table_reviews`. This means
there are orders that do not appear in the items table but do appear in the
reviews table! This happens because `table_items` only shows items that were
successfully delivered. Hence, this mismatch in rows might happen when a
customer cancels the order before receiving it but still leaves a review.

5. It seems reasonable to think that an early or late delivery will impact the
customer's review. To test this, we will calculate the number of days that
an order was delayed. First, take `table_orders` and transform columns
`'order_estimated_delivery_date'` and `'order_delivered_customer_date'` to
DateTime format using `pd.to_datetime` (hint: use `format='%Y-%m-%d %H:%M:%S'`
to parse the strings). Then create a new column called `days_delay` by
subtracting the estimated delivery date from the customer delivery date (if it's
negative, then it arrived early, and if it's positive, it arrived late) and only
keep its `day` part (hint: access the new column's `.dt.days` attribute).

6. Inner join `df` with `table_orders` using `order_id` as join key and make
sure you're only adding columns `order_status`, `days_delay` to `df` (don't add
any other columns from `table_orders`).

7. It seems reasonable to think that if a listing has many photos, the
customer will be able to make a more informed decision when buying that product
and would therefore be less likely to leave a negative review (because the
photos would arguably serve as a proxy for the product's quality and features).
First fill the null values of column `'product_photos_qty'` with its own median
value. Then, add column `product_photos_qty` to `table_items` by joining it with
`table_products` on `product_id`. Finally, calculate the average number of
photos per order and store this aggregated dataset in a dataframe called
`agg_pics`.

8. The whole point of calculating `'avg_pics` is to use it as a feature in
our models, so inner join `df` and `agg_pics` on `order_id` (do not add any
other columns).

9. Add a column full of ones named `'const'` to `df`. This is my gift to you.
You're welcome.

10. Fit a logistic regression model to the final form of `df`. Make sure to only
use delivered orders (filter `'order_status'`). The endogenous variable must be
*happy*, and the exogenous variables must be *const, n_items, avg_price,
avg_shipping, days_delay and avg_pics*

11. Now fit a logistic regression model to the final form of `df`. Like before,
use delivered orders, and this time, exclude orders with multiple items. This
way, aggregated columns such as `'avg_price'` now represent the actual price of
the only product in the order. This is just a robustness test to check if orders
with more than one order are skewing our results. The endogenous variable must
be *happy*, and the exogenous variables must be *const, avg_price, avg_shipping,
days_delay and avg_pics*. We're dropping *n_items* because it is now full of
ones.

12. Remember to lint your code using
`flake8 <path to your file> --max_line_length=88`.
