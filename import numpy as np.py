import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import pandas as pd

dataset=pd.read_csv('construction_estimate.csv')

# Select two features for 3D visualization
X = dataset[['Builtup_Area', 'Floors']]
y = dataset['Cost']

# Train model
model = LinearRegression()
model.fit(X, y)

# Create a meshgrid for the two features
x_surf, y_surf = np.meshgrid(
    np.linspace(X['Builtup_Area'].min(), X['Builtup_Area'].max(), 50),
    np.linspace(X['Floors'].min(), X['Floors'].max(), 50)
)

# Predict cost over the grid
z_pred = model.predict(
    np.column_stack((x_surf.ravel(), y_surf.ravel()))
).reshape(x_surf.shape)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual data points
ax.scatter(X['Builtup_Area'], X['Floors'], y, color='blue', alpha=0.6, label='Actual Data')

# Regression plane
ax.plot_surface(x_surf, y_surf, z_pred, color='red', alpha=0.4, linewidth=0)

# Labels and title
ax.set_xlabel('Builtup Area')
ax.set_ylabel('Floors')
ax.set_zlabel('Cost')
ax.set_title('3D Regression Plane: Builtup Area & Floors vs Cost')
ax.legend()

plt.show()
