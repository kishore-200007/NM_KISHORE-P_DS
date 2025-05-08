# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gradio as gr
# 1. Data Preprocessing
df = pd.read_csv('/stock_data.csv')

# Check if 'Date' column exists in the DataFrame
if 'Date' in df.columns:
    # Parse dates if 'Date' column is present
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
else:
    # Print a warning if 'Date' column is not found
    print("Warning: 'Date' column not found in the DataFrame.")

# Check missing values
df = df.dropna()
# 2. Exploratory Data Analysis
print("Data Summary:\n", df.describe())
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Distribution of closing prices
sns.histplot(df['Close'], kde=True)
plt.title('Closing Price Distribution')
plt.show()
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ... (your existing code for data loading and preprocessing) ...

# 3. Feature Engineering
# ... (your existing feature engineering code) ...

# Check for infinite and large values in X
# Replace inf and -inf with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Find very large or small values that might be causing issues
# You can adjust the threshold based on your data
threshold = 1e6  # Example threshold
large_values = X[X.abs() > threshold]
print("Large values in X:\n", large_values)

# Now handle these large values
# Option 1: Remove rows with large values
X = X[X.abs() <= threshold].dropna()  # Removes rows with any large value or NaN
y = y[X.index]  # Update y to match X's index

# Option 2: Impute with mean or median
# for col in X.columns:
#     X[col] = X[col].fillna(X[col].mean())  # Replace NaN with mean
#     # Or use X[col].median() for median imputation

# Option 3: Winsorize - cap extreme values at a certain percentile
# from scipy.stats.mstats import winsorize
# for col in X.columns:
#     X[col] = winsorize(X[col], limits=[0.05, 0.05]) # Cap at 5th and 95th percentiles

# ... (rest of your code) ...
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib here
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ... (your existing code for data loading, preprocessing, and feature engineering) ...

# 4. Model Building & Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# 5. Visualization of Results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('True vs Predicted Closing Prices')
plt.show()
# 6. Deployment using Gradio
def predict_stock(Open, High, Low, Volume, MA7, MA21, Return, Lag1):
    input_data = pd.DataFrame({
        'Open': [Open],
        'High': [High],
        'Low': [Low],
        'Volume': [Volume],
        'MA7': [MA7],
        'MA21': [MA21],
        'Return': [Return],
        'Lag1': [Lag1]
    })
    pred = model.predict(input_data)
    return pred[0]

interface = gr.Interface(
    fn=predict_stock,
    inputs=[
        gr.Number(label="Open"),
        gr.Number(label="High"),
        gr.Number(label="Low"),
        gr.Number(label="Volume"),
        gr.Number(label="MA7"),
        gr.Number(label="MA21"),
        gr.Number(label="Return"),
        gr.Number(label="Lag1"),
    ],
    outputs=gr.Number(label="Predicted Close Price"),
    title="Stock Price Predictor"
)

interface.launch()