# 🧠 Lecture: Numerical Computing with NumPy & SciPy + Hands-on Gravity Simulation

---

## 🗂️ Part 1: Introduction to Numerical Computing

### 📌 Why Numerical Computing?
Numerical computing involves solving mathematical problems approximately using algorithms and computers — especially useful when analytical methods are too complex or impossible.

---

## ⚙️ Part 2: NumPy — Numerical Python

### ✅ Key Features:
- Efficient array storage and manipulation
- Vectorized operations
- Built-in linear algebra and random number generation

### 🔢 NumPy Basics:

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3])
b = np.linspace(0, 10, 5)  # Evenly spaced numbers

# Array operations
sum_ab = a + a
dot_product = np.dot(a, a)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
inv_A = np.linalg.inv(A)
```

---

## 📚 Part 3: SciPy — Scientific Computing

### ✅ What SciPy adds:
- Numerical integration: `scipy.integrate`
- Optimization: `scipy.optimize`
- Interpolation, signal processing, special functions, etc.

### 🧮 Example: Calculate the following integration

$$
\int_0^1 x^2 \, dx 
$$

```python
from scipy.integrate import quad

def f(x):
    return x**2

result, error = quad(f, 0, 1)
print("∫₀¹ x² dx =", result)
```
---

## 🚀 Part 4: Hands-on Simulation – Motion Under Gravity

### 🎯 Objective:
Numerically simulate an object falling under gravity using **Euler's method**.

---

### 🧠 Physics Recap:

- Gravitational acceleration:  
  \[
  g = 9.81 \ \text{m/s}^2
  \]

- Velocity as a function of time:
  \[
  v(t) = v_0 + g \cdot t
  \]

- Position as a function of time:
  \[
  y(t) = y_0 + v_0 \cdot t + \frac{1}{2} g \cdot t^2
  \]

---

### 🧑‍💻 Python Simulation (Euler's Method):

```python
import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
y0 = 100.0   # Initial height (meters)
v0 = 0.0     # Initial velocity (m/s)
g = -9.81    # Gravity (m/s²)

dt = 0.01    # Time step (s)
t_max = 5    # Max simulation time (s)

# Time array
t = np.arange(0, t_max, dt)
n = len(t)

# Initialize arrays
y = np.zeros(n)
v = np.zeros(n)

y[0] = y0
v[0] = v0

# Euler integration
for i in range(1, n):
    v[i] = v[i-1] + g * dt
    y[i] = y[i-1] + v[i-1] * dt
    if y[i] < 0:
        y[i:] = 0
        break

# Plot results
plt.plot(t, y, label='Height (y)')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Motion Under Gravity')
plt.grid(True)
plt.legend()
plt.show()
```

---

## 🧪 Try it Yourself – Exercises

### 🛠️ Modify the simulation to include:

**1.** An initial upward velocity:

\[
v_0 = 10 \ \text{m/s}
\]

**2.** Air resistance proportional to velocity (drag force):

\[
F_{\text{drag}} = -k \cdot v
\]

Use \( k = 0.1 \) and update the velocity equation accordingly:

\[
v[i] = v[i-1] + (g - k \cdot v[i-1]) \cdot dt
\]

---

## 📍 Summary

- NumPy handles fast, efficient numerical array computations.
- SciPy extends NumPy with advanced mathematical capabilities.
- You can simulate motion under gravity using simple numerical methods like Euler's method.

---

### 📁 Bonus: Homework

**1.** Simulate projectile motion in 2D (include horizontal motion).  
