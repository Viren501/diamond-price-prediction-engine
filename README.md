# Diamond Valuation Prediction

A comprehensive machine learning project dedicated to predicting the market price of diamonds using multiple and polynomial regression techniques. This analysis highlights the end-to-end data science lifecycle, emphasizing feature scaling, categorical encoding, and model performance comparison.

## Dataset Description
The model is trained on the `DiamondsPrices.csv` dataset, which contains 53,940 records. 
* **Target Variable:** `price` (in USD).
* **Numerical Features:** `carat`, `depth`, `table`, and physical dimensions `x`, `y`, `z`.
* **Categorical Features:** `cut`, `color`, and `clarity`.

## Methodology
1. **Data Preprocessing:** * Split the dataset into an 80% training set and a 20% testing set.
2. **Feature Engineering:** * Applied `StandardScaler` to normalize all numerical features to ensure equal weighting. 
   * Utilized `OneHotEncoder` (dropping the first category) to convert categorical text data into a machine-readable format without introducing multicollinearity.
3. **Modeling:** * **Base Model:** Built and trained a standard `LinearRegression` model using the combined processed features.
   * **Advanced Model:** Generated degree-2 polynomial features (`PolynomialFeatures`) from the scaled numerical data to capture non-linear relationships, then trained a second `LinearRegression` model.

## Results & Evaluation
Both models performed exceptionally well, but capturing feature interactions yielded a measurable improvement:
* **Linear Regression:** Achieved an excellent R-squared ($R^2$) score of **~0.9189**, explaining nearly 92% of the variance in diamond prices.
* **Polynomial Regression:** Improved the R-squared ($R^2$) score to **~0.9306**, demonstrating that the relationships between a diamond's physical attributes and its price are not strictly linear. 

## How to Run Locally
1. Clone this repository to your local machine.
2. Ensure you have Python installed along with the necessary libraries: `pip install numpy pandas scikit-learn jupyter`.
3. Place `DiamondsPrices.csv` in the same directory as the notebook.
4. Open your terminal or command prompt, navigate to the folder, and run `jupyter notebook`.
5. Open the `.ipynb` file and run the cells sequentially to observe the data transformations and predictions.
