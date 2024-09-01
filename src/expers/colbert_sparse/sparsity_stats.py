
# PLOTTED STATS: sparsity scores (quartiles), overall loss, colbert loss, sparsity loss

from matplotlib import pyplot as plt
import numpy as np
import torch
import os

BATCH_COUNT = 50
SPARSITY_DATA_DIR = "outputs/stats/sparsity_scores/lmbd-1.0"
PLOTS_OUT_DIR = "outputs/stats/sparsity_scores/sigmoid/plots/"

device = "cpu" if not torch.cuda.is_available() else "cuda:0"

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_stats(stats_dir:str, lmbd=""):
    lmbd_float = 0.0
    if lmbd:
        lmbd_float = float(lmbd.lstrip("lmbd-"))
    
    
    
    batch_data = []
    for batch_num in range(BATCH_COUNT):
        rank_data = []
        for rank in range(4):
            loaded_scores = torch.load(os.path.join(stats_dir, f"{lmbd_float}_rank{rank}_sparsity_scores_{batch_num*1000}.pt"), map_location=device)
            rank_data.append(loaded_scores)

        rank_data = torch.cat(rank_data, dim=0)
        batch_mean = torch.mean(rank_data, dim=0)       
        batch_data.append(batch_mean)
        
    batch_quartiles = [[] for _ in range(4)]    

    for batch_idx, doc in enumerate(batch_data):
        doc = doc.reshape(-1)
        doc = torch.sort(doc).values
        doc = doc.to(torch.float32)
        
        q1 = torch.quantile(doc, 0.25).item()
        q2 = torch.quantile(doc, 0.50).item()
        q3 = torch.quantile(doc, 0.75).item()
        q4 = torch.quantile(doc, 1.00).item()

        batch_quartiles[0].append(q1)
        batch_quartiles[1].append(q2)
        batch_quartiles[2].append(q3)
        batch_quartiles[3].append(q4)

    for i, quartiles in enumerate(batch_quartiles):
        qn_y = moving_average(np.array(quartiles))

        plt.plot(np.arange(len(qn_y))*1000, qn_y, label=f"Q{i+1}")

    plt.title("Sparsity value quartiles")
    plt.ylabel("Quartile value")
    plt.xlabel("Batch number")
    plt.legend()

    plt.savefig(os.path.join(PLOTS_OUT_DIR, f"sparsity_scores_quartiles_{lmbd}.png"))
    plt.clf()


    ## LOSS FUNCTIONS

    ### sparsity loss

    sparsity_loss_rank_data = [ [] for i in range(4)]
    for rank_idx in range(4):
        for batch_num in range(BATCH_COUNT):
            with open(os.path.join(stats_dir, f"{lmbd_float}_rank{rank_idx}_sparsity_loss_{batch_num*1000}.tsv")) as f:
                line = f.readline().strip()
                _, loss_value = line.split("\t")
                loss_value = float(loss_value)/lmbd_float
                sparsity_loss_rank_data[rank_idx].append(loss_value)
    
        sparsity_loss_rank_data[rank_idx] = np.array(sparsity_loss_rank_data[rank_idx])
    plt.title(f"Sparsity loss")
    plt.xlabel("Batch num")
    plt.ylabel("Sparsity loss value")
    for rank_idx in range(4):
        plt.plot(np.arange(len(sparsity_loss_rank_data[rank_idx]))*1000, sparsity_loss_rank_data[rank_idx], label=f"{rank_idx}")
    plt.savefig(os.path.join(PLOTS_OUT_DIR, f"sparsity_loss_{lmbd}.png"))
    plt.clf()


    ### no sparse loss

    no_sparse_loss_rank_data = [ [] for i in range(4)]
    for rank_idx in range(4):
        for batch_num in range(BATCH_COUNT):
            with open(os.path.join(stats_dir, f"{lmbd_float}_rank{rank_idx}_no_sparse_loss_{batch_num*1000}.tsv")) as f:
                line = f.readline().strip()
                _, loss_value = line.split("\t")
                loss_value = float(loss_value)
                no_sparse_loss_rank_data[rank_idx].append(loss_value)
    
        no_sparse_loss_rank_data[rank_idx] = np.array(no_sparse_loss_rank_data[rank_idx])
    plt.title(f"Loss (sparse exclude)")
    plt.xlabel("Batch num")
    plt.ylabel("Sparsity loss value")
    for rank_idx in range(4):
        plt.plot(np.arange(len(no_sparse_loss_rank_data[rank_idx]))*1000, no_sparse_loss_rank_data[rank_idx], label=f"{rank_idx}")
    plt.savefig(os.path.join(PLOTS_OUT_DIR, f"no_sparse_loss_{lmbd}.png"))
    plt.clf()

    ### overall loss (colbert loss + sparse loss)

    overall_loss_rank_data = [ [] for i in range(4)]
    for rank_idx in range(4):
        for batch_num in range(BATCH_COUNT):
            with open(os.path.join(stats_dir, f"{lmbd_float}_rank{rank_idx}_no_sparse_loss_{batch_num*1000}.tsv")) as f:
                line = f.readline().strip()
                _, loss_value = line.split("\t")
                loss_value = float(loss_value)
                overall_loss_rank_data[rank_idx].append(loss_value)
    
        overall_loss_rank_data[rank_idx] = np.array(overall_loss_rank_data[rank_idx])
    plt.title(f"Overall loss (colbert + sparse)")
    plt.xlabel("Batch num")
    plt.ylabel("Sparsity loss value")
    for rank_idx in range(4):
        plt.plot(np.arange(len(overall_loss_rank_data[rank_idx]))*1000, overall_loss_rank_data[rank_idx], label=f"{rank_idx}")
    plt.savefig(os.path.join(PLOTS_OUT_DIR, f"overall_{lmbd}.png"))
    plt.clf()
    
    return batch_quartiles, sparsity_loss_rank_data[0], no_sparse_loss_rank_data[0], overall_loss_rank_data[0]

sparsity_loss_arr = []
no_sparse_loss_arr = []
overall_loss_arr =[]

for i in [0.0001, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    i = float(i)
    print(f"Plotting stats i={i}")
    stats = plot_stats(f"outputs/stats/sparsity_scores/sigmoid/lmbd-{i}", str(i))
    quartiles, sparsity_loss, no_sparse_loss, overall_loss = stats
    
    sparsity_loss_arr.append(sparsity_loss)
    no_sparse_loss_arr.append(no_sparse_loss)
    overall_loss_arr.append(overall_loss)
    

# LOSS comparisons

def plot_loss_comparison(data_array:list, out_path:str, plot_name:str="Sample graph"):
    plt.clf()
    plt.title(plot_name)
    plt.xlabel("Batch num")
    plt.ylabel("Loss value")
    for i, data in enumerate(data_array):
        plt.plot(np.arange(len(data)) * 1000, data, label=str(i/10))
        
        if len(data_array) / 2 == i: 
            plt.legend()
            plt.savefig(out_path.rstrip(".png") + f"_{str(i)}_" + ".png")
            plt.clf()
            plt.xlabel("Batch num")
            plt.ylabel("Loss value")

comparisons = [sparsity_loss_arr, no_sparse_loss_arr, overall_loss_arr]
names = ["Sparsity loss comparison", "No sparse loss comparison", "Overall loss comparison"]

print("Plotting comparisons...")

for comp_data, plot_name in zip(comparisons, names):
    out_path = plot_name.replace(" ", "_").lower() + ".png"
    out_path = os.path.join(PLOTS_OUT_DIR, out_path)
    plot_loss_comparison(comp_data, out_path, plot_name)

    
print("Done.")
