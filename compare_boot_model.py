import csv
from statistics import mean
import random
from math import sqrt


def load_close_prices(path):
    closes = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            closes.append(float(row['Close']))
    return closes


def create_dataset(closes):
    X = closes[:-1]
    y = closes[1:]
    return X, y


def linear_regression(X, y):
    n = len(X)
    mean_x = mean(X)
    mean_y = mean(y)
    var_x = sum((x - mean_x) ** 2 for x in X)
    cov_xy = sum((x - mean_x) * (y_i - mean_y) for x, y_i in zip(X, y))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def bootstrap_models(X, y, n_boot=10):
    models = []
    n = len(X)
    for _ in range(n_boot):
        indices = [random.randrange(n) for _ in range(n)]
        Xb = [X[i] for i in indices]
        yb = [y[i] for i in indices]
        models.append(linear_regression(Xb, yb))
    return models


def predict_boot(models, x):
    total = 0.0
    for intercept, slope in models:
        total += intercept + slope * x
    return total / len(models)


def train_test_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def mse(preds, actual):
    return sum((p - a) ** 2 for p, a in zip(preds, actual)) / len(preds)


if __name__ == '__main__':
    path = 'MSFT_1986_2025-06-30.csv'
    closes = load_close_prices(path)
    X, y = create_dataset(closes)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    base_intercept, base_slope = linear_regression(X_train, y_train)
    base_preds = [base_intercept + base_slope * x for x in X_test]
    base_error = mse(base_preds, y_test)

    boot_models = bootstrap_models(X_train, y_train, n_boot=30)
    boot_preds = [predict_boot(boot_models, x) for x in X_test]
    boot_error = mse(boot_preds, y_test)

    print(f'Base Linear Regression MSE: {base_error:.6f}')
    print(f'Bootstrap Ensemble MSE:   {boot_error:.6f}')
    if boot_error < base_error:
        print('Bootstrap ensemble performed better.')
    else:
        print('Base linear regression performed better.')
