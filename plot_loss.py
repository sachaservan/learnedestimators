import sys
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math 


###########################
# PLOT PARAMETERS
###########################
font = {'family': 'sans-serif',
        'sans-serif': ['Tahoma', 'DejaVu Sans', ' Lucida Grande', 'Verdana'],
        'size': 14}
matplotlib.rc('font', **font)
colors = ["#003f5c", "#ff6361", "#58508d", "#0091d4", "#D7005C"]

# more colors:
# https://www.color-hex.com/color-palettes/
edgecolor = '#34495e'
gridcolor = '#2f3640'
linestyle = 'dotted'
opacity = 1
###########################

###########################
# loss functions to compute 
###########################

def loss_weighted(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est) * y_true

def loss_l1(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est)

def loss_l2(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est) ** 2

# order of functions and labels must match
loss_functions = [loss_l1, loss_l2, loss_weighted]
loss_labels = ["Average Loss per item (L1)", "Average Loss (L2)", "Average Loss (Weighted)"]
###########################


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--results_learned", type=str, default='')
    argparser.add_argument("--results_cutoff", type=str, default='')
    argparser.add_argument("--x_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--y_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--title", type=str, default='')
    argparser.add_argument("--algo", type=str, default='Alg')
    argparser.add_argument("--aol", action='store_true', default=False)
    argparser.add_argument("--synth", action='store_true', default=False)
    argparser.add_argument("--max_space", type=float, default=4.0)
    args = argparser.parse_args()
    
    save_file_names = ["experiments/loss_l1_ip.pdf", "experiments/loss_l2_ip.pdf", "experiments/loss_weighted_ip.pdf", "experiments/selection_ip.pdf"]
    if args.aol:
        save_file_names = ["experiments/loss_l1_aol.pdf", "experiments/loss_l2_aol.pdf", "experiments/loss_weighted_aol.pdf", "experiments/selection_aol.pdf"]
    elif args.synth:
        save_file_names = ["experiments/loss_l1_synth.pdf", "experiments/loss_l2_synth.pdf", "experiments/loss_weighted_synth.pdf", "experiments/selection_synth.pdf"]
       
    results_regular = np.load(args.results_learned,  allow_pickle=True)
    space = np.array(results_regular['space_list'])
    true_counts = np.array(results_regular['true_values'])
    pred_counts = np.array(results_regular['oracle_predictions'])
    algo_predictions = np.array(results_regular['valid_algo_predictions'])
    count_sketch_predictions = np.array(results_regular['valid_count_sketch_predictions'])
    
    if args.results_cutoff:
        results_cutoff = np.load(args.results_cutoff,  allow_pickle=True)
        cutoff_algo_predictions = np.array(results_cutoff['valid_cutoff_algo_predictions'])
        cutoff_count_sketch_predictions = np.array(results_cutoff['valid_cutoff_count_sketch_predictions'])
        true_counts2 = np.array(results_cutoff['true_values'])
        print("num space " + str(len(cutoff_algo_predictions)))

    else:
        cutoff_algo_predictions = np.ones(len(algo_predictions))
        cutoff_count_sketch_predictions = np.ones(len(count_sketch_predictions))

    total = sum(true_counts)
    
    space_percent = [math.ceil(x) for x in (space*1e8) / ((len(true_counts)-1) * 4)]
         
    for i in range(len(loss_functions)):
        loss_sketch = []
        loss_learned =  []
        loss_just_cutoff =  []
        loss_learned_cutoff =  []
        #loss_learned_perfect_oracle_i = []

        for j in range(len(space)):
            loss_for_space_algo = 0
            loss_for_space_sketch = 0
            loss_for_space_cutoff_algo = 0
            loss_for_space_cutoff_count_sketch = 0

            for k in range(len(true_counts)):
                loss_for_space_algo = loss_functions[i](loss_for_space_algo, true_counts[k], algo_predictions[j][k])
                loss_for_space_sketch = loss_functions[i](loss_for_space_sketch, true_counts[k], count_sketch_predictions[j][k])
                loss_for_space_cutoff_algo = loss_functions[i](loss_for_space_cutoff_algo, true_counts[k], cutoff_algo_predictions[j][k])
                loss_for_space_cutoff_count_sketch = loss_functions[i](loss_for_space_cutoff_count_sketch, true_counts[k], cutoff_count_sketch_predictions[j][k])

            loss_learned.append(loss_for_space_algo)
            loss_sketch.append(loss_for_space_sketch)
            loss_learned_cutoff.append(loss_for_space_cutoff_algo)
            loss_just_cutoff.append(loss_for_space_cutoff_count_sketch)

        loss_sketch = np.array(loss_sketch) / total
        loss_learned = np.array(loss_learned) / total
        loss_just_cutoff = np.array(loss_just_cutoff) / total
        loss_learned_cutoff = np.array(loss_learned_cutoff) / total
        #loss_learned_perfect_oracle_i = np.array(loss_learned_perfect_oracle_i) / total


        ax = plt.figure().gca()

        ax.plot(space_percent, loss_sketch, label="Count Sketch", linewidth=3, color=colors[0], zorder=5)
        ax.plot(space_percent, loss_learned, label=args.algo, linewidth=3, color=colors[1], zorder=6)
        ax.plot(space_percent, loss_just_cutoff, label="Cutoff Count Sketch", linestyle='dashdot', color=colors[3])
        ax.plot(space_percent, loss_learned_cutoff, label="Cutoff " + args.algo , linestyle='dashdot', color=colors[2])
        # ax.plot(space, loss_learned_perfect_oracle_i, label=args.algo + " (perfect oracle)", linestyle='dotted', color=colors[4])


        ax.yaxis.grid(color=gridcolor, linestyle=linestyle)
        ax.xaxis.grid(color=gridcolor, linestyle=linestyle)
        ax.set_axisbelow(True)
        ax.set_yscale('log', basey=10)
        ax.set_ylabel(loss_labels[i])
        ax.set_xlabel('Percent space')

        if args.aol:
            ax.set_title('AOL Dataset')
        elif args.synth:
            ax.set_title('Synthetic Dataset')
        else:
            ax.set_title('IP Dataset')

        if args.y_lim:
            ax.set_ylim(args.y_lim)
        if args.x_lim:
            ax.set_xlim(args.x_lim)
            
        plt.legend(loc='best')
        plt.savefig(save_file_names[i])


    # fig, ax = plt.subplots(figsize=(12, 8))



    # true_counts = np.array(data['true_values'])
    # sort = np.argsort(true_counts)[::-1]

    # n_partitions = 2
    # partition_step = int(math.ceil(len(true_counts)/n_partitions))
    # min_index = 0
    # max_index = partition_step
    # j = 0
    # incorrect = np.zeros(n_partitions)
    # for i in range(len(sort)):
    #     if sort[i] > max_index or sort[i] < min_index:
    #         incorrect[j] += 1
    #     if i != 0 and i % partition_step == 0:
    #         min_index += partition_step
    #         max_index += partition_step
    #         j += 1
    
    # x = np.arange(n_partitions)
    # percent_incorrect = np.array(incorrect)/partition_step

    # # Define bar width. We'll use this to offset the second bar.
    # bar_width = 0.25

    # print(sort[:100])

    # # Same thing, but offset the x by the width of the bar.
    # b1 = ax.bar(x, percent_incorrect, width=bar_width, label='Oracle', color=colors[1])
                

    # ax.set_ylabel('Fraction of Predictions Selected')
    # ax.set_xlabel('Space (MB)')
    # plt.xticks(x + bar_width, space)
    # plt.legend(loc='best')
    # plt.savefig(save_file_names[3])
