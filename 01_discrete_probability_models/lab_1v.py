import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp


def parse_input(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    blocks = []
    block = {}
    reading_data = False
    reading_values = False
    data = []
    values = []

    for line in lines:
        if line.startswith("type"):
            if block:
                block["data"] = np.array(data, dtype=float)
                if values:
                    block["values"] = values
                blocks.append(block)
            block = {}
            data = []
            values = []
            reading_data = False
            reading_values = False
            block["type"] = line.split("=")[1].strip()
            continue
        if line == "values":
            reading_values = True
            reading_data = False
            continue
        if line == "data":
            reading_data = True
            reading_values = False
            continue
        if reading_values:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("Ошибка: строка в values должна быть вида: Xi Gen")
            xi, gen = parts
            values.append((float(xi), float(gen)))
            continue
        if reading_data:
            try:
                data.append(float(line))
            except ValueError:
                raise ValueError(f"Ошибка: строка '{line}' в data не число!")
            continue
        if "=" in line:
            k, v = line.split("=")
            try:
                block[k.strip()] = float(v.strip())
            except ValueError:
                block[k.strip()] = v.strip()
            continue
    if block:
        block["data"] = np.array(data, dtype=float)
        if values:
            block["values"] = values
        blocks.append(block)

    return blocks


def calc_theoretical_prob(block):
    """Возвращает словарь {значение: теоретическая вероятность}"""
    t = block["type"]
    if t == "bernoulli":
        p = block["p"]
        return {0: 1 - p, 1: p}

    elif t == "binomial":
        p = block["p"]
        m = int(block["m"])
        probs = {}
        for k in range(m + 1):
            probs[k] = factorial(m) / (factorial(k) * factorial(m - k)) * (p ** k) * ((1 - p) ** (m - k))
        return probs

    elif t == "poisson":
        lamb = block["lambda"]
        max_k = int(max(block["data"])) + 2
        probs = {}
        for k in range(max_k):
            probs[k] = (lamb ** k) * exp(-lamb) / factorial(k)
        return probs

    elif t == "discrete":
        probs = {}
        raws = block["values"]
        total = sum(gen for _, gen in raws)
        for xi, gen in raws:
            probs[xi] = gen / total
        return probs



def calc_empirical(block):
    """Считает частоты, относительные частоты и F(x)"""
    data = block["data"]
    values, counts = np.unique(data, return_counts=True)
    rel_freq = counts / len(data)
    F = np.cumsum(rel_freq)
    return values, counts, rel_freq, F


def calc_statistics(data):
    """Возвращает мат. ожидание, дисперсию, σ, асимметрию и эксцесс"""
    mean = np.mean(data)
    var = np.var(data, ddof=0)
    std = np.std(data, ddof=0)
    skew = np.mean(((data - mean) / std) ** 3)
    kurt = np.mean(((data - mean) / std) ** 4) - 3
    return mean, var, std, skew, kurt


def print_table(values, counts, rel_freq, theor_probs, F_label):
    """Печатает таблицу как в Excel"""
    print(f"{'Карман':>7} | {'Частота':>8} | {'Отн. част':>10} | {'Теор.':>10} | {F_label:>10}")
    print("-" * 60)
    F = 0
    for v, c, r in zip(values, counts, rel_freq):
        theor = theor_probs.get(v, 0)
        F += r
        print(f"{int(v):>7} | {int(c):>8} | {r:>10.6f} | {theor:>10.6f} | {F:>10.6f}")
    print("-" * 60)
    print(f"{'Сумма:':>7} | {sum(counts):>8} | {sum(rel_freq):>10.6f} | {'1':>10} | {'1':>10}\n")


def print_statistics(data):
    """Печатает основные статистические показатели"""
    mean, var, std, skew, kurt = calc_statistics(data)
    print("Статистические показатели:")
    print(f"Мат. ожидание:           {mean:.6f}")
    print(f"Среднеквадр. отклонение: {std:.6f}")
    print(f"Дисперсия:               {var:.6f}")
    print(f"Асимметрия:              {skew:.6f}")
    print(f"Эксцесс:                 {kurt:.6f}\n")

def generate_blocks():
    np.random.seed(476)
    blocks = []

    #Бернулли
    p = np.random.uniform(0.2, 0.8)
    k = np.random.randint(100, 201)
    
    data = np.random.binomial(1, p, size=k)

    blocks.append({
        "type": "bernoulli",
        "p": p,
        "data": data
    })

    #Биномиальное
    p = np.random.uniform(0.2, 0.8)
    m = np.random.randint(3, 7)
    k = np.random.randint(100, 201)

    data = np.random.binomial(m, p, size=k)

    blocks.append({
        "type": "binomial",
        "p": p,
        "m": m,
        "data": data
    })

    #Пуассон
    lamb = np.random.uniform(2.0, 10.0)
    k = np.random.randint(100, 201)
    
    data = np.random.poisson(lamb, size=k)

    blocks.append({
        "type": "poisson",
        "lambda": lamb,
        "data": data
    })

    #Дискретное
    k = np.random.randint(100, 201)

    Xi = np.array([5, 8, 9, 12, 14, 17, 19, 21, 25, 29])
    gen = np.random.uniform(0.0, 1.0, size=len(Xi))
    probs = gen / gen.sum()

    data = np.random.choice(Xi, size=k, p=probs)

    blocks.append({
        "type": "discrete",
        "values": list(zip(Xi, gen)),
        "data": data
    })

    return blocks



def analyze(block):
    name = block["type"].capitalize()
    F_label = "F(x)" if block["type"] != "binomial" and block["type"] != "poisson" else "F(Y)"
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    #График выборки
    x = np.arange(1, len(block["data"]) + 1)
    y = block["data"]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', linestyle='-', label=f"{name}")
    plt.title(f"Реализация СП ({name})")
    plt.xlabel("k")
    plt.ylabel("x")
    plt.grid(True)
    plt.legend()
    plt.show()

    values, counts, rel_freq, F = calc_empirical(block)
    theor_probs = calc_theoretical_prob(block)

    print_table(values, counts, rel_freq, theor_probs, F_label)

    print_statistics(block["data"])

    #Гистограмма относительных частот
    plt.figure(figsize=(6, 4))
    plt.bar(values, rel_freq, color='skyblue', edgecolor='black')
    plt.title(f"Гистограмма относительных частот ({name})")
    plt.xlabel("Карман (значение)")
    plt.ylabel("Относительная частота")
    plt.grid(axis='y')
    plt.show()

    #Сравнение относительных и теоретических частот
    theor_y = [theor_probs.get(v, 0) for v in values]
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(values - width / 2, rel_freq, width=width, label="Отн. част", color='skyblue')
    plt.bar(values + width / 2, theor_y, width=width, label="Теор.", color='orange')
    plt.title(f"Сравнение частот ({name})")
    plt.xlabel("Карман (значение)")
    plt.ylabel("Частота")
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    #График функции распределения
    plt.figure(figsize=(6, 4))
    plt.plot(values, F, color='purple', marker='o', linestyle='-', label=F_label)
    plt.title(f"Функция распределения {F_label} ({name})")
    plt.xlabel("Значение")
    plt.ylabel(F_label)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    mode = input("Выберите режим (1 — тестовый, 2 — рабочий): ").strip()

    if mode == "1":
        filename = "input.txt"
        blocks = parse_input(filename)

    elif mode == "2":
        blocks = generate_blocks()

    else:
        raise ValueError("Неверный режим")

    for block in blocks:
        analyze(block)
