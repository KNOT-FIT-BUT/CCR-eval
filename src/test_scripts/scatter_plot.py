import matplotlib.pyplot as plt
import numpy as np

exec_times = [1.26, 27.67,112.40,16.73,996.94,24618.09]
# exec_times = [3.89,35.0,15.0, 3.29, 6.17,6.14]
precision = [0.32, 0.39, 0.40, 0.41,  0.39, 0.36]

# Define x and y coordinates
x = np.array(precision)
y = np.array(exec_times)

# Define a list of different colors and corresponding labels
colors = ['red', 'green', 'blue', 'brown', 'purple', 'orange']

labels = [ "BM25 (cs)", "PLAIDX (cs)", "COLBERTv2 (en)", "COLBERTv2_noseg (en)", "SPLADEv2 (en)", "OpenAI ADAv2 (en)"]

# Create a scatter plot for each point with a different color and label
for i in range(len(x)):
    plt.scatter(x[i], y[i], c=colors[i], label=labels[i])

# Add a legend
plt.legend()
plt.grid()

# plt.yscale("log")

# Add labels to the axes
plt.xlabel('P@5')
plt.ylabel('Query latency [ms]')

# Add a title to the plot
plt.title('Query latency / p@5 comparison')

# Show the plot
plt.savefig("scatter_plot.png")
