from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
import numpy as np
import sys

def format_table_line(input_line:list, n=10, delim="|", out_stream=sys.stdout):
    for i, item in enumerate(input_line):
        out_stream.write(f"{item:^{n}}")
        if i+1 != len(input_line):
            out_stream.write(f" {delim} ")
    out_stream.write("\n")

def print_header(queries_count:int=-1, query_file:str="", index_file:str="", out_file=sys.stdout):
    out_file.write("-------------------------\n")
    out_file.write("Date: " + str(datetime.now()) + "\n")
    if queries_count != -1:
        out_file.write("Query count: " + str(queries_count) + "\n")
    if query_file:
        out_file.write("Qrel: " + query_file + "\n")
    if index_file:
        out_file.write("Index: " + index_file + "\n")

def print_stats(stats:dict, metrics_at_k:list, queries_count:int, query_file:str, index_file:str, out_file):
    print_header(queries_count, query_file, index_file, out_file)
    for k1, b in stats.keys():
        out_file.write(f"Params [k1, b]: [{k1}, {b}]" + "\n")
        format_table_line(["@K"] + [str(val) for val in metrics_at_k], n=15, out_stream=out_file)
        out_file.write(7*15*"_" + "\n")
        
        data = stats[(k1, b)]

        for name, values in data.items():
            line_values = [name.upper()]
            for k in metrics_at_k:
                value = values[k]
                # Convert exec_time to milliseconds
                if name == "exec_time":            
                    value = value * 10**3

                line_values.append('{:.2f}'.format(round(value, 2)))
            format_table_line(line_values, n=15, out_stream=out_file)                        

def print_stats_raw(stats:dict, out_file):
    out_file.write(str(stats))

def plot_grid_search(stats:dict, output_path:str, k=10):
    
    k1_values = []
    b_values = []
    precision_values = []
    for k1, b in stats.keys():
        k1_values.append(k1)
        b_values.append(b)
        precision_values.append(stats[(k1, b)]["precision"][k])

    # 3d plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(k1_values, b_values, precision_values, cmap="viridis")
    #ax.scatter(k1_values, b_values, precision_values, c='b', marker='o')

    ax.set_xlabel('K1 values')
    ax.set_ylabel('B values')
    ax.set_zlabel(f"Precision")

    ax.set_title(f"Precision@{k} in relation to k1 and b")
    plt.savefig(output_path)    



def print_grid_table(stats:dict, metrics_at_k:list, queries_count:int, query_file:str, index_file:str, out_file):
    print_header(queries_count, query_file, index_file, out_file)
    data = []
    for k1, b in stats.keys():
        row = []
        row.append(float(b))
        row.append(float(k1))
        for k in metrics_at_k:
            row.append(round(stats[(k1, b)]["precision"][k],2))

        data.append(row)

    headers = ["b", "k1"] + [f"P@{k}" for k in metrics_at_k]

    table = tabulate(data, headers=headers, tablefmt="grid")
    out_file.write(table)    


