import sys
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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


def loss_weighted(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est) * y_true

def loss_l1(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est)

def loss_l2(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est) ** 2

# order of functions and labels must match
loss_functions = [loss_l1, loss_l2, loss_weighted]
loss_labels = ["Average Loss per item (L1)", "Average Loss (L2)", "Average Loss (Weighted)"]

skip = 0 # number of values to skip

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--data", type=str, default='')
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
    if args.data:
        data = np.load(args.data,  allow_pickle=True)

        space = data['space_list']
        true_counts = np.array(data['true_values'])
        algo_predictions = np.array(data['test_algo_predictions'])
        count_sketch_predictions = np.array(data['test_count_sketch_predictions'])

        sum_counts = np.sum(true_counts)

        for i in range(len(loss_functions)):
            loss_sketch_i = []
            loss_learned_i =  []
            # loss_just_cutoff_i =  []
            # loss_learned_cutoff_i =  []
            # loss_learned_perfect_oracle_i = []

            for j in range(len(space)):
                loss_for_space_algo = 0
                loss_for_space_sketch = 0
                for k in range(skip, len(true_counts)):
                    loss_for_space_algo = loss_functions[i](loss_for_space_algo, true_counts[k], algo_predictions[j][k])
                    loss_for_space_sketch = loss_functions[i](loss_for_space_sketch, true_counts[k], count_sketch_predictions[j][k])

                loss_learned_i.append(loss_for_space_algo)
                loss_sketch_i.append(loss_for_space_sketch)
              

            loss_sketch_i = np.array(loss_sketch_i) / sum_counts
            loss_learned_i = np.array(loss_learned_i) / sum_counts
           # loss_just_cutoff_i = np.array(loss_just_cutoff_i) / args.sum
            #loss_learned_cutoff_i = np.array(loss_learned_cutoff_i) / args.sum
           # loss_learned_perfect_oracle_i = np.array(loss_learned_perfect_oracle_i) / args.sum


            ax = plt.figure().gca()

            ax.plot(space, loss_sketch_i, label="Count sketch", linewidth=3, color=colors[0], zorder=5)
            ax.plot(space, loss_learned_i, label=args.algo, linewidth=3, color=colors[1], zorder=6)
          #  ax.plot(space, loss_just_cutoff_i, label="Count sketch (with cutoff)", linestyle='dashdot', color=colors[3])
          #  ax.plot(space, loss_learned_cutoff_i, label=args.algo + " (with cutoff)", linestyle='dashdot', color=colors[2])
           # ax.plot(space, loss_learned_perfect_oracle_i, label=args.algo + " (perfect oracle)", linestyle='dotted', color=colors[4])

            ax.yaxis.grid(color=gridcolor, linestyle=linestyle)
            ax.xaxis.grid(color=gridcolor, linestyle=linestyle)
            ax.set_axisbelow(True)
            ax.set_yscale('log')
            ax.set_ylabel(loss_labels[i])
            ax.set_xlabel('Space (MB)')

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


    fig, ax = plt.subplots(figsize=(12, 8))

    # if args.data:
    #     data = np.load(args.data)
    #     space = data['space_list']
    #     selections = data['selections_oracle_no_cutoff']    
    #     false_selections = data['false_selections_oracle_no_cutoff']    

    #     for i in range(len(selections)):
    
    #         grouped_selections = np.array_split(np.array(selections[i]), 200)
    #         grouped_false_selections = np.array_split(np.array(false_selections[i]), 200)
    #         grouped_avg = []
    #         grouped_false_avg = []
    #         for j in range(len(grouped_selections)):
    #             grouped_avg.append(np.mean(grouped_selections[j]))
    #             grouped_false_avg.append(np.mean(grouped_false_selections[j]))

    #         x = range(len(grouped_avg))
    #         x_false = range(len(grouped_false_avg))
    #         coef = np.polyfit(x, grouped_false_avg, 1) 
    #         coef_false = np.polyfit(x, grouped_false_avg, 1) 
    #         poly1d_fn = np.poly1d(coef) 
    #         poly1d_false_fn = np.poly1d(coef_false) 
    #         ax.scatter(x, grouped_avg)
    #         ax.scatter(x, grouped_false_avg)
    #         # ax.plot(x, poly1d_fn(x), '--')   
    #         # ax.plot(x_false, poly1d_fn(x_false), '--')   

    #     ax.set_yscale('log')
    #     #ax.set_xscale('log')
    #     ax.set_ylabel('Fraction of Predictions Selected')
    #     ax.set_xlabel('Item Frequency')
    #     plt.legend(loc='best')
    #     #plt.show()
    #     plt.savefig('experiments/selection.pdf')

    exit(0)
    if args.data:
        data = np.load(args.data)
        space = data['space_list']
      
        selections_raw = data['selections_oracle_no_cutoff']    
        false_selections_raw = data['false_selections_oracle_no_cutoff'] 

        percent_selection_oracle = np.zeros(len(space), dtype=float)
        percent_selection_sketch = np.zeros(len(space), dtype=float)
        percent_wrong_selection = np.zeros(len(space), dtype=float)

        for i in range(len(space)):
            percent_selection_oracle[i] = np.sum(selections_raw[i]) / len(selections_raw[i])
            percent_selection_sketch[i] = 1.0 - percent_selection_oracle[i]
            percent_wrong_selection[i] = np.sum(false_selections_raw[i]) / len(false_selections_raw[i])


        x = np.arange(len(space))

        # Define bar width. We'll use this to offset the second bar.
        bar_width = 0.25

        # Same thing, but offset the x by the width of the bar.
        b1 = ax.bar(x, percent_selection_oracle,
                    width=bar_width, label='Oracle', color=colors[1])
                    
        # Note we add the `width` parameter now which sets the width of each bar.
        b2 = ax.bar(x + bar_width, percent_selection_sketch,
                    width=bar_width, label='Count Sketch', color=colors[0])
        
        # Note we add the `width` parameter now which sets the width of each bar.
        b3 = ax.bar(x + 2*bar_width, percent_wrong_selection,
                    width=bar_width, label='Wrong Selection', color='black')


        ax.set_ylabel('Fraction of Predictions Selected')
        ax.set_xlabel('Space (MB)')
        plt.xticks(x + bar_width, space)
        plt.legend(loc='best')
        plt.savefig(save_file_names[3])
