import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, gamma, beta

SEED = 2176
BINS_COUNT = 15
np.random.seed(SEED)

def to_float(s):
    return float(s.replace(",", "."))


def calc_statistics(data):
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    skew = np.mean(((data - mean) / std) ** 3)
    kurt = np.mean(((data - mean) / std) ** 4) - 3
    return mean, var, std, skew, kurt

def analyze_continuous(name, data, dist_type, params):
    n = len(data)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, n + 1), data, marker='o')
    plt.title(f"Реализация СП ({name})")
    plt.xlabel("k")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    if dist_type == "uniform":
        a, b = params
        bins = np.linspace(0, a + b, BINS_COUNT + 1)
    else:
        bins = np.linspace(min(data), max(data), BINS_COUNT + 1)

    centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(data, bins=bins)
    rel_freq = counts / n
    F_emp = np.cumsum(rel_freq)

    theor_freq = []

    if dist_type == "uniform":
        theor_freq = [1 if c > 0 else 0 for c in counts]

    elif dist_type == "normal":
        mu, sigma = params
        for i in range(len(bins) - 1):
            theor_freq.append(
                norm.cdf(bins[i + 1], mu, sigma) -
                norm.cdf(bins[i], mu, sigma)
            )

    elif dist_type == "gamma":
        alpha, beta_p = params
        for i in range(len(bins) - 1):
            theor_freq.append(
                gamma.cdf(bins[i + 1], alpha, scale=beta_p) -
                gamma.cdf(bins[i], alpha, scale=beta_p)
            )

    elif dist_type == "beta":
        a, b = params
        for i in range(len(bins) - 1):
            theor_freq.append(
                beta.cdf(bins[i + 1], a, b) -
                beta.cdf(bins[i], a, b)
            )

    theor_freq = np.array(theor_freq)

    density = theor_freq / theor_freq.sum() if theor_freq.sum() != 0 else theor_freq

    print(f"{'Карман':>10} | {'Частота':>8} | {'Отн.част':>10} | {'F(Y)':>10} | {'Теор.част':>10} | {'Плотность':>10}")
    print("-" * 85)

    for c, cnt, rf, Fy, tf, d in zip(
            centers, counts, rel_freq, F_emp, theor_freq, density):
        print(f"{c:10.5f} | {cnt:8} | {rf:10.6f} | {Fy:10.6f} | {tf:10.6f} | {d:10.6f}")

    print("-" * 85)
    print(f"{'Сумма':>10} | {sum(counts):8} | {sum(rel_freq):10.6f} | {F_emp[-1]:10.6f} | {theor_freq.sum():10.6f} | {density.sum():10.6f}\n")

    mean, var, std, skew, kurt = calc_statistics(data)
    print("Статистические характеристики:")
    print(f"Мат. ожидание: {mean:.6f}")
    print(f"Дисперсия: {var:.6f}")
    print(f"СКО: {std:.6f}")
    print(f"Асимметрия: {skew:.6f}")
    print(f"Эксцесс: {kurt:.6f}\n")

    plt.figure(figsize=(7, 4))
    plt.bar(centers, rel_freq, width=centers[1] - centers[0], edgecolor='black')
    plt.title(f"Гистограмма относительных частот ({name})")
    plt.xlabel("Карман")
    plt.ylabel("Отн. частота")
    plt.grid(axis='y')
    plt.show()

    plt.figure(figsize=(7, 4))

    plt.bar(centers, rel_freq,
            width=centers[1] - centers[0],
            label="Относительная вероятность",
            edgecolor='black',
            alpha=0.7)


    plt.plot(centers, density,
             marker='o',
             linestyle='-',
             color='orange',
             label="Плотность")

    plt.title(f"Относительная вероятность и плотность ({name})")
    plt.xlabel("Карман")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(7, 4))
    plt.plot(centers, F_emp, marker='o')
    plt.title(f"Функция распределения F(Y) ({name})")
    plt.xlabel("Y")
    plt.ylabel("F(Y)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.show()

def parse_input(filename):
    with open(filename, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    blocks = []
    block = {}
    data_blocks = {}
    current_data = None

    for line in lines:
        if line.startswith("type"):
            if block:
                block["data_blocks"] = data_blocks
                blocks.append(block)
            block = {"type": line.split("=")[1]}
            data_blocks = {}
            current_data = None
            continue

        if line.startswith("data"):
            current_data = line
            data_blocks[current_data] = []
            continue

        if current_data:
            data_blocks[current_data].append(to_float(line))
            continue

        if "=" in line:
            k, v = line.split("=")
            block[k] = to_float(v)

    if block:
        block["data_blocks"] = data_blocks
        blocks.append(block)

    return blocks

def main():
    mode = input("Режим (1 — тестовый, 2 — рабочий): ").strip()

    if mode == "1":
        blocks = parse_input("input.txt")

        for b in blocks:
            t = b["type"]

            if t == "uniform":
                data = np.array(b["data_blocks"]["data"])
                analyze_continuous(
                    "Равномерное",
                    data,
                    "uniform",
                    (b["a"], b["b"])
                )

            elif t == "normal":
                data = np.array(b["data_blocks"]["data"])
                analyze_continuous(
                    "Нормальное",
                    data,
                    "normal",
                    (b["mu"], b["sigma"])
                )

            elif t == "gamma":
                data = np.array(b["data_blocks"]["data"])
                analyze_continuous(
                    "Гамма",
                    data,
                    "gamma",
                    (b["alpha"], b["beta"])
                )

            elif t == "beta":
                alpha = b["alpha"]
                db = b["data_blocks"]

                analyze_continuous("Beta α=β", np.array(db["data1"]), "beta", (alpha, alpha))
                analyze_continuous("Beta α=β (10n)", np.array(db["data2"]), "beta", (alpha, alpha))
                analyze_continuous("Beta α, β=4α", np.array(db["data3"]), "beta", (alpha, 4 * alpha))
                analyze_continuous("Beta α=1 β=1", np.array(db["data4"]), "beta", (1, 1))

    elif mode == "2":
        print("\nРАБОЧИЙ РЕЖИМ (генерация с seed = 2176)\n")

        a, b = 21, 42
        n = np.random.randint(100, 201)
        data = np.random.uniform(a, b, n)
        analyze_continuous(
            "Равномерное",
            data,
            "uniform",
            (a, b)
        )

        mu, sigma = 21, 0.21
        n = np.random.randint(100, 201)
        data = np.random.normal(mu, sigma, n)
        analyze_continuous(
            "Нормальное",
            data,
            "normal",
            (mu, sigma)
        )

        alpha = np.random.uniform(3, 10)
        beta_p = np.random.uniform(1, 7)
        n = np.random.randint(100, 201)
        data = np.random.gamma(alpha, beta_p, n)
        analyze_continuous(
            "Гамма",
            data,
            "gamma",
            (alpha, beta_p)
        )

        alpha = np.random.uniform(3, 7)
        n = np.random.randint(100, 201)
        data = np.random.beta(alpha, alpha, n)
        analyze_continuous(
            "Бета",
            data,
            "beta",
            (alpha, alpha)
        )

    else:
        print("Ошибка: введите 1 или 2")


main()
