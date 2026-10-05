# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats


def to_float(s):
    return float(s.replace(",", "."))


def parse_input(filename):
    with open(filename, encoding="utf-8") as f:
        lines = [
            l.strip()
            for l in f
            if l.strip() and not l.strip().startswith("#")
        ]

    blocks = []
    block = {}
    data = {}
    current = None

    for line in lines:

        if line.startswith("type"):
            if block:
                block["data"] = data
                blocks.append(block)

            block = {"type": line.split("=")[1]}
            data = {}
            current = None
            continue

        if line.startswith("data"):
            current = line
            data[current] = []
            continue

        if current:
            data[current].append(to_float(line))
            continue

        if "=" in line:
            k, v = line.split("=")
            block[k] = to_float(v)

    if block:
        block["data"] = data
        blocks.append(block)

    return blocks


def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)

    def f(p):
        a, b = p
        return np.sum((y - (a + b * x)) ** 2)

    res = minimize(f, [0, 0])
    a, b = res.x

    y_pred = a + b * x
    return a, b, y_pred


def nonlinear_regression(x, y):
    x = np.array(x)
    y = np.array(y)

    def f(p):
        a, b, c = p
        return np.sum((y - (a + b * x + c * x**2)) ** 2)

    res = minimize(f, [0, 0, 0])
    a, b, c = res.x

    y_pred = a + b * x + c * x**2
    return a, b, c, y_pred


def multiple_regression(x1, x2, z):
    x1 = np.array(x1)
    x2 = np.array(x2)
    z = np.array(z)

    def f(p):
        a0, a1, a2, a11, a22, a12 = p
        pred = (a0 + a1 * x1 + a2 * x2 +
                a11 * x1**2 + a22 * x2**2 +
                a12 * x1 * x2)
        return np.sum((z - pred) ** 2)

    res = minimize(f, [0, 0, 0, 0, 0, 0])
    return res.x


def calc_stats(y, y_pred, p):
    y = np.array(y)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot
    r = np.sqrt(r2)

    n = len(y)
    err = np.sqrt(ss_res / (n - p - 1))

    return r, r2, err


def test_mode():
    blocks = parse_input("input.txt")

    for block in blocks:

        t = block["type"]

        if t == "linear":
            x = block["data"]["data_x"]
            y = block["data"]["data_y"]

            a, b, y_pred = linear_regression(x, y)

            print("\nЛинейная регрессия")
            print("a =", round(a, 4))
            print("b =", round(b, 4))

            r, r2, err = calc_stats(y, y_pred, 2)

            print("R =", round(r, 4))
            print("R^2 =", round(r2, 4))
            print("Ошибка =", round(err, 4))

            plt.figure()
            plt.scatter(x, y)
            plt.plot(x, y_pred)
            plt.grid()
            plt.show()

        elif t == "nonlinear":
            x = block["data"]["data_x"]
            y = block["data"]["data_y"]

            a, b, c, y_pred = nonlinear_regression(x, y)

            print("\nНелинейная регрессия")
            print("a =", round(a, 4))
            print("b =", round(b, 4))
            print("c =", round(c, 4))

            r, r2, err = calc_stats(y, y_pred, 3)

            print("R =", round(r, 4))
            print("R^2 =", round(r2, 4))
            print("Ошибка =", round(err, 4))

            plt.figure()
            plt.scatter(x, y)
            plt.plot(x, y_pred)
            plt.grid()
            plt.show()

        elif t == "multiple":
            x1 = block["data"]["data_x1"]
            x2 = block["data"]["data_x2"]
            z = block["data"]["data_z"]

            params = multiple_regression(x1, x2, z)

            print("\nМножественная регрессия")
            names = ["a0", "a1", "a2", "a11", "a22", "a12"]

            for n, v in zip(names, params):
                print(n, "=", round(v, 4))


def work_mode():
    k = int(input("Введите k: "))
    np.random.seed(2176)

    x = np.arange(k)
    x2 = np.arange(k)

    a = np.random.normal(3, 0.6)
    b = np.random.normal(0.3, 0.06)
    h = np.random.normal(12, 2.4)

    noise = np.random.normal(0, 1, size=k)

    y = a + b * x + h * noise

    print("\nЛинейная (генерация)")
    print("Истинные параметры:")
    print("a =", round(a, 4), "b =", round(b, 4), "h =", round(h, 4))

    a_est, b_est, y_pred = linear_regression(x, y)

    print("Оценки:")
    print("a =", round(a_est, 4), "b =", round(b_est, 4))

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.grid()
    plt.show()

    a = np.random.normal(3, 0.6)
    b = np.random.normal(0.3, 0.06)
    c = np.random.normal(0.03, 0.006)
    h = np.random.normal(70, 14)

    noise = np.random.normal(0, 1, size=k)

    y = a + b * x + c * x**2 + h * noise

    print("\nНелинейная (генерация)")
    print("Истинные параметры:")
    print("a =", round(a, 4), "b =", round(b, 4), "c =", round(c, 4), "h =", round(h, 4))

    a_est, b_est, c_est, y_pred = nonlinear_regression(x, y)

    print("Оценки:")
    print("a =", round(a_est, 4), "b =", round(b_est, 4), "c =", round(c_est, 4))

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.grid()
    plt.show()

    a0 = np.random.normal(3, 0.6)
    a1 = np.random.normal(0.3, 0.06)
    a2 = np.random.normal(0.3, 0.06)
    a11 = np.random.normal(0.3, 0.06)
    a22 = np.random.normal(0.3, 0.06)
    a12 = np.random.normal(0.3, 0.06)
    h = np.random.normal(25, 5)

    noise = np.random.normal(0, 1, size=k)

    z = (a0 + a1 * x + a2 * x2 +
         a11 * x**2 + a22 * x2**2 +
         a12 * x * x2 +
         h * noise)

    print("\nМножественная (генерация)")
    print("Истинные параметры:")
    print("a0 =", round(a0, 4))
    print("a1 =", round(a1, 4))
    print("a2 =", round(a2, 4))
    print("a11 =", round(a11, 4))
    print("a22 =", round(a22, 4))
    print("a12 =", round(a12, 4))
    print("h =", round(h, 4))

    params = multiple_regression(x, x2, z)

    print("Оценки:")
    names = ["a0", "a1", "a2", "a11", "a22", "a12"]
    for n, v in zip(names, params):
        print(n, "=", round(v, 4))


def main():
    print("1 - тестовый режим")
    print("2 - рабочий режим")

    choice = input("Выбор: ")

    if choice == "1":
        test_mode()
    else:
        work_mode()


if __name__ == "__main__":
    main()