from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def plot_var_points(points, build_time, query_times, space):
    # Model definitions
    def linear(n, a, b): return a * n + b
    def quadratic(n, a, b): return a * n**2 + b
    def nlogn(n, a, b): return a * n * np.log(n) + b
    def logn(n, a, b): return a * np.log(n) + b
    def constant(n, a): return a
    
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

    # Set global styling parameters
    rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.5,
        'axes.grid': False,
        'figure.dpi': 150
    })

    def plot_best_fit(x, y, model_dict, best_name, title, xlabel, ylabel):
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, label="Observed", marker='o', color='black', s=15)

        best_model_func = model_dict[best_name]
        best_params, _ = curve_fit(best_model_func, x, y)
        y_fit = best_model_func(x_fit, *best_params)
        plt.plot(x_fit, y_fit, label=f"{best_name} Fit", color='firebrick')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    # --- BUILD TIME ---
    build_models = {'Linear': linear, 'nlogn': nlogn, 'Quadratic': quadratic, 'Constant': constant}
    best_build, build_params, build_preds = best_fit_model(points, build_time, build_models)
    plot_best_fit(points, build_time, build_models, best_build,
                f"Build Time vs. Number of Points",
                "Number of Points (n)", "Build Time (s)")

    # --- QUERY TIME ---
    query_models = {'logn': logn, 'Linear': linear, 'Quadratic': quadratic, 'Constant': constant}
    best_query, query_params, query_preds = best_fit_model(points, query_times, query_models)
    plot_best_fit(points, query_times, query_models, best_query,
                f"Query Time vs. Number of Points",
                "Number of Points (n)", "Query Time (s)")

    # --- SPACE ---
    space_models = {'Linear': linear, 'nlogn': nlogn, 'Quadratic': quadratic, 'Constant': constant}
    best_space, space_params, space_preds = best_fit_model(points, space, space_models)
    plot_best_fit(points, space, space_models, best_space,
                f"Space vs. Number of Points",
                "Number of Points (n)", "Space (bytes)")
    
def plot_var_priorities(priorities, build_time, query_times, space):
    # Model definitions
    def linear(n, a, b): return a * n + b
    def quadratic(n, a, b): return a * n**2 + b
    def nlogn(n, a, b): return a * n * np.log(n) + b
    def logn(n, a, b): return a * np.log(n) + b
    def constant(n, a): return a
    
    x_fit = np.linspace(min(priorities), max(priorities), 200)

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

    # Set global styling parameters
    rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.5,
        'axes.grid': False,
        'figure.dpi': 150
    })

    def plot_best_fit(x, y, model_dict, best_name, title, xlabel, ylabel):
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, label="Observed", marker='o', color='black', s=15)

        best_model_func = model_dict[best_name]
        best_params, _ = curve_fit(best_model_func, x, y)
        y_fit = best_model_func(x_fit, *best_params)
        plt.plot(x_fit, y_fit, color='firebrick')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(0, 10)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    # --- BUILD TIME ---
    build_models = {'Linear': linear, 'nlogn': nlogn, 'Quadratic': quadratic, 'Constant': constant}
    best_build, build_params, build_preds = best_fit_model(priorities, build_time, build_models)
    plot_best_fit(priorities, build_time, build_models, best_build,
                f"Build Time vs. Number of Priorities on 10k Points",
                "Number of Priorities (p)", "Build Time (s)")

    # --- QUERY TIME ---
    query_models = {'logn': logn, 'Linear': linear, 'Quadratic': quadratic, 'Constant': constant}
    best_query, query_params, query_preds = best_fit_model(priorities, query_times, query_models)
    plot_best_fit(priorities, query_times, query_models, best_query,
                f"Query Time vs. Number of Priorities on 10k Points",
                "Number of Priorities (p)", "Query Time (s)")

    # --- SPACE ---
    space_models = {'Linear': linear, 'nlogn': nlogn, 'Quadratic': quadratic, 'Constant': constant}
    best_space, space_params, space_preds = best_fit_model(priorities, space, space_models)
    plot_best_fit(priorities, space, space_models, best_space,
                f"Space vs. Number of Priorities on 10k Points",
                "Number of Priorities (p)", "Space (bytes)")
    

def main():
    # Load data
    # df = pd.read_csv("data.csv")
    # print(df)
    # points = df["Points"].values
    # build_time = df["Build"].values
    # query_times = df["Query"].values
    # space = df["Space"].values
    # plot_var_points(points, build_time, query_times, space)
    
    df = pd.read_csv("p_runtime10k.csv")
    print(df)
    points = df["Priorities"].values
    build_time = df["Build"].values
    query_times = df["Query"].values
    space = df["Space"].values
    plot_var_priorities(points, build_time, query_times, space)
    
main()