
---

# 📊 Unemployment Prediction Using Retail Data and Deep Learning

This project focuses on **predicting unemployment rates** using retail store data by leveraging **machine learning and deep learning techniques**. The dataset includes features from Walmart stores such as sales, promotions, weather, and economic indicators.

## 🧩 What We Did

1. **Data Loading & Merging**  
   - Combined `stores.csv` and `features.csv` on the `Store` column to create a unified dataset.

2. **Data Cleaning**  
   - Handled missing values (e.g., filled `MarkDown` fields with 0, dropped rows with missing `CPI` or `Unemployment`).
   - Converted `Date` to datetime and `IsHoliday` to binary (0/1).

3. **Exploratory Data Analysis (EDA)**  
   - Generated summary statistics.
   - Visualized correlations, unemployment distribution, store sizes by type, temperature trends, and relationships like Fuel Price vs CPI.

4. **Feature Engineering**  
   - Extracted `Year`, `Month`, and `Week` from the `Date` column to capture temporal patterns.

5. **Modeling with PyTorch**  
   - Built a **feedforward neural network** to predict unemployment.
   - Used `StandardScaler` for numeric features and `OneHotEncoder` for categorical features (`Type`).
   - Trained the model using **Adam optimizer** and **MSE loss** over 100 epochs.

6. **Evaluation**  
   - Evaluated performance using test MSE.
   - Plotted training loss and actual vs predicted values for analysis.

---

## 🛠️ Skills & Tools Used

| Skill | Description |
|------|-------------|
| **Python** | Core programming and scripting |
| **Pandas & NumPy** | Data manipulation, cleaning, and transformation |
| **Matplotlib & Seaborn** | Data visualization and EDA |
| **Scikit-learn** | Preprocessing (`StandardScaler`, `OneHotEncoder`), `train_test_split` |
| **PyTorch** | Building and training a neural network from scratch |
| **Machine Learning** | Regression, feature engineering, pipeline creation |
| **Deep Learning** | Feedforward neural networks, loss optimization, DataLoader |
| **Data Analysis** | Handling missing data, correlation analysis, distribution studies |

---

## 📈 Key Results

- Successfully trained a neural network to predict unemployment with reasonable convergence.
- Visualized model performance through loss curves and prediction scatter plots.
- Demonstrated strong integration of **data preprocessing**, **EDA**, and **deep learning** in a real-world context.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unemployment-prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn torch
   ```

3. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook unemployment_prediction.ipynb
   ```

> Make sure `stores.csv` and `features.csv` are in the working directory.

---

## 📁 Project Structure

```
unemployment-prediction/
│
├── stores.csv           # Store metadata (Store, Type, Size)
├── features.csv         # Weekly features (sales, temp, CPI, etc.)
├── unemployment_prediction.ipynb  # Main analysis and model
└── README.md            # This file
```

---

## ✅ Future Improvements

- Use more advanced models (e.g., LSTM for time-series trends).
- Predict sales instead of unemployment (original Kaggle task).
- Add cross-validation and hyperparameter tuning.

---

