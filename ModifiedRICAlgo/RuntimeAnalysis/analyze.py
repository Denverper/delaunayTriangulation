import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("test.csv")
print(df)
points = df["Points"].values
build_time = df["Build"].values
query_times = df["Query"].values
space = df["Space"].values

# Model definitions
def linear(n, a, b): return a * n + b
def quadratic(n, a, b): return a * n**2 + b
def nlogn(n, a, b): return a * n * np.log(n) + b
def logn(n, a, b): return a * np.log(n) + b

x_fit = np.linspace(min(points), max(points), 200)

def best_fit_model(x, y, models):
    """Return best model based on R² score."""
    best_model = None
    best_r2 = -np.inf
    best_params = None
    predictions = {}
    
    for name, func in models.items():
        try:
            popt, _ = curve_fit(func, x, y)
            y_pred = func(x, *popt)
            r2 = r2_score(y, func(x, *popt))
            predictions[name] = (y_pred, popt, r2)
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
                best_params = popt
        except:
            continue
    return best_model, best_params, predictions

# --- BUILD TIME ---
build_models = {'Linear': linear, 'nlogn': nlogn, 'Quadratic': quadratic}
best_build, build_params, build_preds = best_fit_model(points, build_time, build_models)

plt.figure(figsize=(10, 5))
plt.scatter(points, build_time, label="Observed", marker='o')
for name, (y_pred, _, _) in build_preds.items():
    style = '-' if name == best_build else '--'
    lw = 3 if name == best_build else 1
    plt.plot(x_fit, build_models[name](x_fit, *curve_fit(build_models[name], points, build_time)[0]), 
             style, label=f"{name} Fit", linewidth=lw)
plt.title(f"Build Time vs. Points (Best: {best_build})")
plt.xlabel("Points")
plt.ylabel("Build Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- QUERY TIME ---
query_models = {'logn': logn, 'Linear': linear, 'Quadratic': quadratic}
best_query, query_params, query_preds = best_fit_model(points, query_times, query_models)

plt.figure(figsize=(10, 5))
plt.scatter(points, query_times, label="Observed", marker='o')
for name, (y_pred, _, _) in query_preds.items():
    style = '-' if name == best_query else '--'
    lw = 3 if name == best_query else 1
    plt.plot(x_fit, query_models[name](x_fit, *curve_fit(query_models[name], points, query_times)[0]),
             style, label=f"{name} Fit", linewidth=lw)
plt.title(f"Query Time vs. Points (Best: {best_query})")
plt.xlabel("Points")
plt.ylabel("Query Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- SPACE ---
space_models = {'Linear': linear, 'nlogn': nlogn, 'Quadratic': quadratic}
best_space, space_params, space_preds = best_fit_model(points, space, space_models)

plt.figure(figsize=(10, 5))
plt.scatter(points, space, label="Observed", marker='o')
for name, (y_pred, _, _) in space_preds.items():
    style = '-' if name == best_space else '--'
    lw = 3 if name == best_space else 1
    plt.plot(x_fit, space_models[name](x_fit, *curve_fit(space_models[name], points, space)[0]),
             style, label=f"{name} Fit", linewidth=lw)
plt.title(f"Space vs. Points (Best: {best_space})")
plt.xlabel("Points")
plt.ylabel("Space (bytes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
