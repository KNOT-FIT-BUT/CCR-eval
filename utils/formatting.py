from datetime import datetime
import sys

def format_table_line(input_line:list, n=10, delim="|", out_stream=sys.stdout):
    for i, item in enumerate(input_line):
        out_stream.write(f"{item:^{n}}")
        if i+1 != len(input_line):
            out_stream.write(f" {delim} ")
    out_stream.write("\n")

def print_stats(stats:dict, metrics_at_k:list, queries_count:int, query_file:str, index_file:str, out_file):
    # Print metrics
    out_file.write("-------------------------\n")
    out_file.write("Date: " + str(datetime.now()) + "\n")
    out_file.write("Query count: " + str(queries_count) + "\n")
    out_file.write("Qrel: " + query_file + "\n")
    out_file.write("Index: " + index_file + "\n")

    format_table_line(["@K"] + [str(val) for val in metrics_at_k], n=15, out_stream=out_file)
    out_file.write(7*15*"_" + "\n")


    for name, statistic in stats.items():
        line_values = [name]
        for value in statistic.values():        
            
            # Convert exec_time to milliseconds
            if name == "EXEC TIME [ms]":            
                value = value * 10**3

            line_values.append('{:.2f}'.format(round(value, 2)))

        format_table_line(line_values, n=15, out_stream=out_file)
        out_file


