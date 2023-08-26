def format_table_line(input_line:list, n=10, delim="|", out_stream=sys.stdout):
    for i, item in enumerate(input_line):
        out_stream.write(f"{item:^{n}}")
        if i+1 != len(input_line):
            out_stream.write(f" {delim} ")
    out_stream.write("\n")
