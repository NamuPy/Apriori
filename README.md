# Apriori
A fast and memory efficient implementation of the Apriori Algorithm from scratch

### Dataset used
- Currently the code is structed assuming input data is from the kaggle dataset of GroceryStoreData available at https://www.kaggle.com/shazadudwadia/supermarket/version/1
- However, this is a very small dataset containing only 20 baskets and 11 unique products
- This code is useful and fast compared to python library "apyori" when the algorithm has to be run for over 100k unique products
- This implementation however is limited to computing support, confidence and lift for each combination of 2 products, but using the same concepts can be expanded as per the requirement

### Advantages

The metric computations of support, confidence and lift for every combination of products is done using sparse matrices and using which makes the algorithm fast and memory efficient


### Structure of results
The results return three numpy matrcies:
- Order_of_products: This is a 1D array with the unique products presented in a specific order
- Top_N_for_each_product: This is a 2D array (number of products, Top N products), where each row is top N most frequently bought together products for each product from the above array "Order_of_products". The ordering of these top N most frequently bought together products is done based on highest value of the lift metric
- Lift_values_for_topN_products: This is a 2D array (number of products, Top N products), where each row is the lift metric of the top N most frequently bought together products shown in "Top_N_for_each_product"
