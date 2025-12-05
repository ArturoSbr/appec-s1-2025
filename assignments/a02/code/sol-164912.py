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
table_reviews['happy'] = (table_reviews['review_score'] >= 4).astype(int)

# Q2. Keep only the last review of each order

# Q3 Declare agg_items

# Q4. Join table_reviews and agg_items to create df

# Q5. Calculate days_delay in table_orders

# Q6. Join df and table_orders to add days_delay

# Q7. Join table_items and table_products to calculate avg_pics

# Q8. Add avg_pics to df

# Q9. Add 'const' to df

# Q10. Fit model 1

# Q11. Fit model 2
