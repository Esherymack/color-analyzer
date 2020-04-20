import numpy as np
import ast
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

families_dict = { "white":0, "grey":0, "black":0, "red":0, "warm red":0, "orange":0, "warm yellow":0, "yellow":0, "cool yellow":0, "yellow green":0, 
                 "warm green":0, "green":0, "cool green":0, "green cyan":0, "warm cyan":0, "cyan":0, "cool cyan":0, "blue cyan":0, "cool blue":0, 
                 "blue":0, "warm blue":0, "violet":0, "cool magenta":0, "magenta":0, "warm magenta":0, "red magenta":0, "cool red":0}


def process_dicts(data):
    for k, v in data.items():
        families_dict[k] += v


def main():

    f = open("./data_runs/first/data.txt", "r")

    data_1 = f.read()
    data_1 = ast.literal_eval(data_1)

    f.close()

    f = open("./data_runs/second/data.txt", "r")

    data_2 = f.read()
    data_2 = ast.literal_eval(data_2)

    f.close()

    f = open("./data_runs/third/data.txt", "r")

    data_3 = f.read()
    data_3 = ast.literal_eval(data_3)

    f.close()

    f = open("./data_runs/fourth/data.txt", "r")

    data_4 = f.read()
    data_4 = ast.literal_eval(data_4)

    f.close()

    f = open("./data_runs/fifth/data.txt", "r")

    data_5 = f.read()
    data_5 = ast.literal_eval(data_5)

    f.close()

    process_dicts(data_1)
    process_dicts(data_2)
    process_dicts(data_3)
    process_dicts(data_4)
    process_dicts(data_5)

    sums = list(families_dict.values())

    for i in range(len(sums)):
        sums[i] = sums[i] // 5 

    k = list(families_dict.keys())

    final = dict(zip(k, sums))

    names = list(final.keys())
    values = list(final.values())

    colors = [(1, 1, 1, 1),
              (0.5, 0.5, 0.5, 1),
              (0, 0, 0, 1),
              (0.941, 0, 0, 1),
              (1, 0.26, 0, 1),
              (0.969, 0.506, 0, 1),
              (0.988, 0.757, 0.02, 1),
              (0.996, 1, 0.043, 1),
              (0.757, 1, 0, 1),
              (0.475, 0.992, 0, 1),
              (0.231, 1, 0.02, 1),
              (0, 1, 0, 1),
              (0, 0.953, 0.239, 1),
              (0, 1, 0.506, 1),
              (0.024, 1, 0.78, 1),
              (0, 0.984, 1, 1),
              (0.047, 0.749, 1, 1),
              (0, 0.506, 0.961, 1),
              (0, 0.263, 0.98, 1),
              (0, 0, 1, 1),
              (0.255, 0, 0.976, 1),
              (0.518, 0, 0.976, 1),
              (0.757, 0, 0.957, 1),
              (1, 0, 1, 1),
              (0.988, 0, 0.733, 1),
              (0.988, 0, 0.482, 1),
              (0.957, 0, 0.243, 1)]

    plt.figure(figsize = (15, 8))

    for i in range(len(names)):
        plt.bar(i, values[i], tick_label = names[i], color = colors[i], edgecolor = "black")

    plt.title("Final average image color clustered into broad color family categories (n = 451, runs = 5)")
    plt.xticks(range(len(names)), names, rotation = 45, fontsize = 8)
    plt.yticks(fontsize = 8)
    ax = plt.gca()

    ax.xaxis.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer = True))

    plt.tight_layout()
    plt.savefig("./analyzed2/average_analysis.jpg")

main()