import os
import json
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


stats_dir = "/home/xsteti05/project/dareczech/stats/top5"

samples = [os.path.join(stats_dir, file) for file in os.listdir(stats_dir) if file.endswith(".top5")]


overlap_matrix = np.ones((len(samples), len(samples)))

print("K-tau stats")
print("Reference,Compare,K-tau,overlap")
for sample_row, sample1_file in enumerate(samples):
    for sample_column, sample2_file in enumerate(samples):
        if sample_row == sample_column:
            continue
        
        with open(sample1_file) as s1:
            with open(sample2_file) as s2:
                sample1 = json.load(s1)
                sample2 = json.load(s2)

        k_tau = 0
        overlap = 0
        
        
        if len(sample1) == 5:
            sample1 = sample1["5"]
            
        if len(sample2) == 5:
            sample2 = sample2["5"]
            

        if len(sample1) != len(sample2):
            print("ERROR: length of samples does not match", len(sample1), len(sample2))
            print(sample1_file, len(sample1))
            print(sample2_file, len(sample2))
            exit(1)
                        
        num = 0
        for val1, val2 in zip(sample1.values(), sample2.values()):    
            cmp = []
            
            concordant = 0
            discordant = 0
            
            for url in val2:
                if url in val1:
                    cmp.append(val1.index(url))   
                    
            overlap += len(cmp)/5
            
            if len(cmp) == 0:
                continue  
            
            elif len(cmp) == 1:
                k_tau += 1.0
            
            else:
                for i, val in enumerate(cmp):
                    for x in range(i+1, len(cmp)):
                        if cmp[x] > cmp[i]:
                            concordant += 1
                        elif cmp[x] < cmp[i]:
                            discordant += 1
                
                k_tau += (concordant - discordant)/(concordant + discordant)
            

        k_tau /= len(sample1)
        overlap /= len(sample1)
        overlap_matrix[sample_row][sample_column] = round(overlap, 2)
        
        if sample_column > sample_row:
            continue
        
        print(f"{os.path.basename(sample1_file)},{os.path.basename(sample2_file)},{round(k_tau, 3)},{round(overlap, 3)}")
            
samples = [os.path.splitext(os.path.basename(sample))[0] for sample in samples]
data = overlap_matrix
mask = np.triu(np.ones_like(data), k=1)
masked_data = ma.masked_array(data, mask=mask)
fig, ax = plt.subplots()
cax = ax.matshow(masked_data, cmap='viridis')
ax.set_xticks(np.arange(len(samples)))
ax.set_yticks(np.arange(len(samples)))
ax.set_xticklabels(samples)
ax.set_yticklabels(samples)

ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')

plt.colorbar(cax)
plt.title("Model overlap_heatmap")

# Show the plot
plt.savefig("overlap_heatmap.png")
