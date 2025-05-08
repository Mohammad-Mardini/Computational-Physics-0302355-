## 📘 **Lecture Title: Introduction to Machine Learning for Physicists**

### 🎯 **Objective:**

By the end of the lecture, students will be able to:

* Define Machine Learning
* Distinguish between supervised and unsupervised learning
* Identify real-world and physics-related applications of both types

---

## 🧠 **1. What is Machine Learning?**

### 🔹 **Definition:**

> **Machine Learning (ML)** is a subset of artificial intelligence that enables systems to **learn from data**, identify patterns, and make **decisions or predictions** without being explicitly programmed.

### 🔹 **Why it Matters in Physics:**

* Analyze massive datasets from experiments (Gaia Dataset)
* Solve inverse problems (e.g., inferring parameters from observed data)
* Discover hidden patterns in simulations or observational data

### 📌 **Analogy:**

> Like teaching a student to solve problems by giving examples and letting them practice instead of memorizing formulas.

---

## 📊 **2. Types of Machine Learning**

### 🧩 **A. Supervised Learning**

#### ✔️ **Definition:**

> Learning from **labeled data** — each input comes with a known output.

#### 📘 **Examples:**

* Predicting house prices from features like size, location (regression)
* Classifying particles as electrons, muons, or pions from detector signals (classification)


```
# ----------------------------
# Step 0: import packages
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Step 1: Load Real Dataset
# ----------------------------
df = pd.read_csv("data.csv")

# ----------------------------
# Step 2: Define Features and Target
# ----------------------------
X = df[['Size', 'Location']]
y = df['Price']

# ----------------------------
# Step 3: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 4: Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Step 5: Predictions and Evaluation
# ----------------------------
y_pred = model.predict(X_test)
print("Model Evaluation:")
print("------------------")
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R² Score:", round(r2_score(y_test, y_pred), 2))
print()

# ----------------------------
# Step 6: New Prediction
# ----------------------------
new_house = np.array([[2600, 4]])
predicted_price = model.predict(new_house)
print("Prediction for New House:")
print("--------------------------")
print(f"Predicted Price for 2600 sqft, Location Score 4: ${predicted_price[0]:,.2f}")

# ----------------------------
# Step 7: Visualization
# ----------------------------
plt.figure(figsize=(8,6))
scatter = plt.scatter(df['Size'], df['Price'],
                      c=df['Location'], cmap='viridis', s=100, edgecolor='k')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($)')
plt.title('House Prices by Size and Location Score')
plt.colorbar(scatter, label='Location Score')
plt.grid(True)
plt.tight_layout()
plt.savefig("plot.png")
plt.show()

```




#### 📊 **In Physics:**

* Classifying phases of matter (e.g., ferromagnetic vs. paramagnetic)
* Identifying particle types in high-energy physics

#### 🧪 **Visual Example:**

* A dataset of galaxy images labeled as spiral or elliptical → train model to classify new images

---

### 🧩 **B. Unsupervised Learning**

#### ✔️ **Definition:**

> Learning from **unlabeled data** — the algorithm tries to find structure or patterns in the data.

#### 📘 **Examples:**

* Grouping customers by purchasing behavior (clustering)
* Reducing dimensionality of data for visualization (e.g., PCA)

#### 📊 **In Physics:**

* Discovering new particle types or states of matter from detector data
* Anomaly detection in sensor readings (e.g., gravitational wave noise analysis)

#### 🧪 **Visual Example:**

* Feeding raw detector signals without labels to cluster them into meaningful groups

---

## 📉 **3. Comparison Table:**

| Feature          | Supervised Learning        | Unsupervised Learning                |
| ---------------- | -------------------------- | ------------------------------------ |
| Data Type        | Labeled                    | Unlabeled                            |
| Goal             | Predict output             | Discover hidden structure            |
| Examples         | Classification, Regression | Clustering, Dimensionality Reduction |
| Physics Use Case | Event classification       | Pattern discovery in raw data        |

---

## 🎓 **4. Interactive Demonstration (Optional)**

* Use a Jupyter notebook to demonstrate:

  * Linear regression (supervised) with simple synthetic physics data
  * K-means clustering (unsupervised) on star brightness and temperature

---

## 🔚 **5. Summary**

* ML lets computers learn from data, and it's already transforming modern physics.
* Supervised learning uses labeled data to predict outcomes.
* Unsupervised learning explores data to uncover structure without labels.

---

## 🧩 **Further Reading & Tools**

* “The Elements of Statistical Learning” – Hastie, Tibshirani, Friedman
* Sci-kit Learn (Python)
* CERN Open Data portal
* [MIT Introduction to Machine Learning](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-036-introduction-to-machine-learning-fall-2020/)

---

Would you like slides or diagrams to accompany this lecture?
