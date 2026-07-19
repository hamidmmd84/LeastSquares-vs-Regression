import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def print_ascii_bar(label, value, max_val, char="█"):
    bar_length = 30
    if max_val == 0 or abs((value / max_val) * bar_length) < 1e-9:
        scaled_length = bar_length
    else:
        scaled_length = int((value / max_val) * bar_length)
    scaled_length = max(0, min(bar_length, scaled_length))
    bar = char * scaled_length + "░" * (bar_length - scaled_length)
    print(f"{label:<25} | {bar} {value:.6f}")

def run_creative_comparison(filepath):
    print("=" * 60)
    print("CONFLICT RESOLUTION: SYSTEM INCONSISTENCY ANALYZER")
    print("=" * 60)


    try:
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return


    A = df.iloc[:, :-1].values
    b = df.iloc[:, -1].values

    print(f"Dataset Loaded: {A.shape[0]} samples with {A.shape[1]} features.")

    print("\n Linear Algebra Diagnostics ")
    A_T_A = A.T @ A
    cond_num = np.linalg.cond(A_T_A)
    print(f"Matrix A^T * A Condition Number: {cond_num:.2f}")
    if cond_num > 1000:
        print("Warning: High multicollinearity detected. Matrix is close to singular.")
    else:
        print("Matrix is well-behaved and stable for inversion.")



    A_hat = np.hstack([np.ones((A.shape[0], 1)), A])
    w_ls = np.linalg.pinv(A_hat) @ b
    b_pred_ls = A_hat @ w_ls

    reg = LinearRegression(fit_intercept=True)
    reg.fit(A, b)
    b_pred_reg = reg.predict(A)

    mse_ls = mean_squared_error(b, b_pred_ls)
    r2_ls = r2_score(b, b_pred_ls)

    mse_reg = mean_squared_error(b, b_pred_reg)
    r2_reg = r2_score(b, b_pred_reg)

    print("\n Performance Comparison ")

    max_mse = max(mse_ls, mse_reg)
    print("\n[Mean Squared Error - MSE] (Lower is better): ")
    print_ascii_bar("Least Square (OLS)", mse_ls, max_mse)
    print_ascii_bar("Linear Regression (ML)", mse_reg, max_mse)

    print("\n[R^2 Score] (HighER is better, Max 1.0): ")
    print_ascii_bar("Least Square (OLS)", r2_ls, 1.0, char="▓")
    print_ascii_bar("Linear Regression (ML)", r2_reg, 1.0, char="▓")

    print("\n Verdict ")
    difference = abs(mse_ls - mse_reg)
    if difference < 1e-7:
        print("Both methods yielded IDENTICAL results.")
        print("Mathematical Proof: Standard Linear Regression is fundamentally OLS.")
    else:
        print(f"Minor difference detected ({difference:.2e}).")
        print("This id due to float precision or solver optimization differences.")
    print("=" * 60)


run_creative_comparison("dataset.csv")