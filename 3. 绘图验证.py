import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('new_processed_option_data_with_abc.csv')

# Assuming '∆Vt' is the result of δMV - δBS
# Plotting ∆Vt against C_DELTA to check for an inverted parabolic shape
plt.figure(figsize=(10, 6))
plt.scatter(data['C_DELTA'], data['∆Vt'], alpha=0.5)
plt.title('Plot of ∆St vs Delta')
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('∆St (δMV - δBS)')
plt.grid(True)
plt.show()

from numpy.polynomial.polynomial import Polynomial

# Fit a second-degree polynomial (parabola) to the data
# This is equivalent to finding a parabolic trend in the scatter plot
coefficients = Polynomial.fit(data['C_DELTA'], data['∆Vt'], 2).convert().coef

# Generate a sequence of Delta values for plotting the parabola
delta_vals = np.linspace(data['C_DELTA'].min(), data['C_DELTA'].max(), num=500)
# Calculate the corresponding ∆Vt values using the fitted polynomial coefficients
fitted_vals = coefficients[0] + coefficients[1] * delta_vals + coefficients[2] * delta_vals**2

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(data['C_DELTA'], data['∆Vt'], alpha=0.5, label='Original Data')

# Plot the fitted parabolic curve
plt.plot(delta_vals, fitted_vals, color='red', linewidth=2, label='Fitted Parabolic Curve')

plt.title('Plot of ∆Vt vs Delta with Fitted Parabola')
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('∆Vt (δMV - δBS)')
plt.legend()
plt.grid(True)
plt.show()

# The inverted parabolic shape would be indicated by a negative coefficient for the squared term.
inverted_parabola = coefficients[2] < 0
inverted_parabola, coefficients
