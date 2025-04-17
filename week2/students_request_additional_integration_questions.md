
## 📘 Step-by-Step Solutions

### 1️⃣ Integral of
$$
\int_{0}^{\pi} \frac{1}{\sqrt{x}}  dx
$$

$$
= \int_{0}^{\pi} x^{-1/2}  dx
= \left[ \frac{x^{1/2}}{1/2} \right]_0^{\pi}
= \left[ 2\sqrt{x} \right]_0^{\pi}
= 2\sqrt{\pi} - 0
= 2\sqrt{\pi}
$$

---

### 2️⃣ Integral of 

$$
\int_{-5}^{5} x^2 \sin(x) \, dx
$$

Note: x^2 \sin(x) is an **odd function**:

$$
f(-x) = (-x)^2 \cdot \sin(-x) = x^2 \cdot (-\sin(x)) = -f(x)
$$

So, the integral of an odd function over a symmetric interval is 0:

$$
\therefore \int_{-5}^{5} x^2 \sin(x) \, dx = 0
$$

---

### 3️⃣ Average value of `f(x) = x` over `[-1, 2]`

Average value of a function over \([a, b]\) is:

$$
\text{Average} = \frac{1}{b - a} \int_{a}^{b} f(x) \, dx
= \frac{1}{2 - (-1)} \int_{-1}^{2} x \, dx
= \frac{1}{3} \left[ \frac{x^2}{2} \right]_{-1}^{2}
= \frac{1}{3} \left( \frac{4}{2} - \frac{1}{2} \right)
= \frac{1}{3} \cdot \frac{3}{2}
= \frac{1}{2}
$$



## ✅ Python Code Using `scipy.integrate.quad`:

```python
import numpy as np
from scipy.integrate import quad

# 1. f(x) = 1/sqrt(x), interval: (0, pi)
f1 = lambda x: 1 / np.sqrt(x)
integral_1, _ = quad(f1, 0, np.pi)

# 2. f(x) = x^2 * sin(x), interval: (-5, 5)
f2 = lambda x: x**2 * np.sin(x)
integral_2, _ = quad(f2, -5, 5)

# 3. Average value of f(x) = x, interval: (-1, 2)
f3 = lambda x: x
a, b = -1, 2
integral_3, _ = quad(f3, a, b)
average_value = integral_3 / (b - a)

# Print results
print(f"1. Integral of 1/sqrt(x) from 0 to pi ≈ {integral_1}")
print(f"2. Integral of x^2 * sin(x) from -5 to 5 ≈ {integral_2}")
print(f"3. Average value of f(x) = x from -1 to 2 ≈ {average_value}")
```
