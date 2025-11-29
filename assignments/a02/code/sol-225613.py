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

# Inspect tables
print("Tablas cargadas exitosamente. Filas en reviews:", len(table_reviews))

# Q1. Create binary target
# Algebra: H = 1 si Score >= 4, sino 0
table_reviews['happy'] = (table_reviews['review_score'] >= 4).astype(int)

# Q2. Keep only the last review of each order
# Algebra: Transformar a Tiempo -> Ordenar Descendente -> Proyección Única (Max)

# 1. Transformación de dominio (String -> Datetime)
table_reviews['review_answer_timestamp'] = pd.to_datetime(
    table_reviews['review_answer_timestamp'],
    format='%Y-%m-%d %H:%M:%S'
)

# 2. Ordenamiento (Más recientes primero)
table_reviews = table_reviews.sort_values(
    by='review_answer_timestamp',
    ascending=False
)

# 3. Reducción por Unicidad (Mantener el primero/reciente por order_id)
table_reviews = table_reviews.drop_duplicates(subset=['order_id'], keep='first')

# --- Check ---
print("Elementos después de filtrar Q2:", len(table_reviews))

# Q3 Declare agg_items
# Algebra: GroupBy(Order_ID) -> [Count, Mean, Mean]
agg_items = table_items.groupby('order_id').agg({
    'order_item_id': 'count',  # Contamos IDs de items (N)
    'price': 'mean',           # Promedio del precio (P_bar)
    'freight_value': 'mean'    # Promedio del envío (F_bar)
}).reset_index()

# Rename columns
agg_items = agg_items.rename(columns={
    'order_item_id': 'n_items',
    'price': 'avg_price',
    'freight_value': 'avg_shipping'
})

# --- Check ---
print("Dimensiones de agg_items:", agg_items.shape)
print(agg_items.head(2))

# Q4. Join table_reviews and agg_items to create df
# Algebra: df = table_reviews INTERSECT agg_items (on order_id)
df = pd.merge(
    table_reviews,
    agg_items,
    on='order_id',
    how='inner'
)

# --- Check ---
print("Dimensiones de df (Intersección):", df.shape)

# Q5. Calculate days_delay in table_orders
# Algebra: Delay = Date_Delivered - Date_Estimated

# 1. Transformación a Datetime (ambas columnas)
table_orders['order_delivered_customer_date'] = pd.to_datetime(
    table_orders['order_delivered_customer_date'],
    format='%Y-%m-%d %H:%M:%S'
)
table_orders['order_estimated_delivery_date'] = pd.to_datetime(
    table_orders['order_estimated_delivery_date'],
    format='%Y-%m-%d %H:%M:%S'
)

# 2. Cálculo de la diferencia y extracción de días
table_orders['days_delay'] = (
    table_orders['order_delivered_customer_date'] -
    table_orders['order_estimated_delivery_date']
).dt.days

# Q6. Join df and table_orders to add days_delay
# Algebra: df = df INTERSECT subset(table_orders)

# Definimos el subconjunto de columnas que nos interesa traer
cols_to_keep = ['order_id', 'order_status', 'days_delay']

df = pd.merge(
    df,
    table_orders[cols_to_keep],  # Solo unimos el subconjunto
    on='order_id',
    how='inner'
)

# --- Check ---
print("Columnas en df después de Q6:", df.columns.tolist())

# Q7. Join table_items and table_products to calculate avg_pics
# Algebra: Impute -> Join -> GroupBy -> Mean

# 1. Imputación (Llenar vacíos con la Mediana)
median_photos = table_products['product_photos_qty'].median()
# Usamos fillna para reemplazar los NaNs con la mediana calculada
table_products['product_photos_qty'] = table_products['product_photos_qty'].fillna(
    median_photos
)

# 2. Unión de Items con Productos (traer columna de fotos a tabla items)
items_with_pics = pd.merge(
    table_items,
    table_products[['product_id', 'product_photos_qty']],  # Solo ID y fotos
    on='product_id',
    how='inner'
)

# 3. Agregación (Promedio de fotos por orden)
agg_pics = (
    items_with_pics
    .groupby('order_id')['product_photos_qty']
    .mean()
    .reset_index()
)

# Renombramos la columna resultante a 'avg_pics'
agg_pics = agg_pics.rename(columns={'product_photos_qty': 'avg_pics'})

# Q8. Add avg_pics to df
# Algebra: df = df INTERSECT agg_pics
df = pd.merge(
    df,
    agg_pics,
    on='order_id',
    how='inner'
)

# --- Check ---
print("¡Listo! Columnas finales en df:", df.columns.tolist())

# Q9. Add 'const' to df
# Algebra: Vector Unitario para el Intercepto
df['const'] = 1

# Q10. Fit model 1
# Algebra: Logit(P(Happy)) ~ X1 (Subset: Delivered)

# Definimos el filtro de entregados
filter_delivered = df['order_status'] == 'delivered'

# Definimos las variables exógenas (X) y endógena (Y)
exog_vars_m1 = [
    'const', 'n_items', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics'
]
y_m1 = df.loc[filter_delivered, 'happy']
X_m1 = df.loc[filter_delivered, exog_vars_m1]

# Fit model
m1 = sm.Logit(y_m1, X_m1, missing='drop')
m1_res = m1.fit()

# Print model summary
print("\nResultados del Modelo 1:")
print(m1_res.summary())

# Q11. Fit model 2
# Algebra: Logit(P(Happy)) ~ X2 (Subset: Delivered AND n_items=1)

# Definimos el filtro compuesto (Entregado Y un solo item)
filter_robust = (df['order_status'] == 'delivered') & (df['n_items'] == 1)

# Definimos variables (quitamos n_items porque es constante)
exog_vars_m2 = [
    'const', 'avg_price', 'avg_shipping', 'days_delay', 'avg_pics'
]
y_m2 = df.loc[filter_robust, 'happy']
X_m2 = df.loc[filter_robust, exog_vars_m2]

# Fit model
m2 = sm.Logit(y_m2, X_m2, missing='drop')
m2_res = m2.fit()

print("\nResultados del Modelo 2 (Robustez):")
print(m2_res.summary())
