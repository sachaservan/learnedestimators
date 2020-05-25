import sys
import time
import argparse
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
###########################



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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--results_learned", type=str, default='')
    argparser.add_argument("--results_cutoff", type=str, default='')
    argparser.add_argument("--aol", action='store_true', default=False)
    argparser.add_argument("--synth", action='store_true', default=False)
    argparser.add_argument("--N", type=int, default=100)
    args = argparser.parse_args()

    file_name = "experiments/err_ip.pdf"
    if args.aol:
        file_name= "experiments/err_aol.pdf"
    if args.synth:
        file_name = "experiments/err_synth.pdf"
   
    results_regular = np.load(args.results_learned,  allow_pickle=True)
    space = np.array(results_regular['space_list'])
    true_counts = np.array(results_regular['true_values'])
    pred_counts = np.array(results_regular['oracle_predictions'])
    algo_predictions = np.array(results_regular['valid_algo_predictions'])
    count_sketch_predictions = np.array(results_regular['valid_count_sketch_predictions'])


        #     loss_per_partition = np.array(data['valid_loss_per_partition'])[space_index]

        # # ax = plt.figure().gca()
        # # ax.bar(range(len(loss_per_partition)), loss_per_partition)
        # # ax.set_ylim([0, np.max(loss_per_partition)+100])
        # # ax.set_xlabel('bucket id')
        # # ax.set_ylabel('# packets')
        # # plt.show()

    results_regular = np.load(args.results_learned,  allow_pickle=True)
    true_counts = np.array(results_regular['true_values'])
    pred_counts = np.array(results_regular['oracle_predictions'])
    space_list = results_regular['space_list']
    pred_counts = np.exp(pred_counts)

    if args.results_cutoff:
        results_cutoff = np.load(args.results_cutoff,  allow_pickle=True)
        cutoff_algo_predictions = np.array(results_cutoff['valid_cutoff_algo_predictions'])
        cutoff_count_sketch_predictions = np.array(results_cutoff['valid_cutoff_count_sketch_predictions'])
    else:
        cutoff_algo_predictions = algo_predictions
        cutoff_count_sketch_predictions = count_sketch_predictions


    # sort the estimates by true freuqency 
    sort = true_counts.argsort()
    true_counts = true_counts[sort][::-1]
    pred_counts = pred_counts[sort][::-1]

    # number of elements in the dataset total 
    num_data_points = len(true_counts)

    # plot absolute error and relative error 
    fig, (ax_abs, ax_rel, ax_true) = plt.subplots(3)

    if args.aol:
        fig.suptitle('AOL Dataset')
    elif args.synth:
        fig.suptitle('Synthetic Dataset')
    else:
        fig.suptitle('IP Dataset')


    N = args.N
    sample_size = len(true_counts) / N


    for space_index in range(len(space_list)): 
        if space_index != len(space_list)-4: # space_index != 0: # 
            continue

        print("Space: " + str(space_list[space_index]) + " MB")

        # sort the predictions according to the true frequency
        algo_predictions_for_space = algo_predictions[space_index][sort][::-1]
        count_sketch_predictions_for_space = count_sketch_predictions[space_index][sort][::-1]
        cutoff_algo_predictions_for_space = cutoff_algo_predictions[space_index][sort][::-1]
        cutoff_count_sketch_predictions_for_space = cutoff_count_sketch_predictions[space_index][sort][::-1]

        # compute the abs and relative erros for all items
        abs_error_oracle_raw = np.abs(true_counts - pred_counts) 
        rel_error_oracle_raw = abs_error_oracle_raw / true_counts

        abs_error_algo_raw = np.abs(true_counts - algo_predictions_for_space) 
        abs_error_algo_cutoff_raw = np.abs(true_counts - cutoff_algo_predictions_for_space) 
        rel_error_algo_raw = abs_error_algo_raw / true_counts
        rel_error_algo_cutoff_raw = abs_error_algo_cutoff_raw / true_counts
       
        print("L1 loss (learned) " + str(np.sum(abs_error_algo_raw)))
        print("L2 loss (learned) " + str(np.sum(abs_error_algo_raw**2)))

        abs_error_sketch_raw = np.abs(true_counts - count_sketch_predictions_for_space) 
        abs_error_sketch_cutoff_raw = np.abs(true_counts - cutoff_count_sketch_predictions_for_space) 
        rel_error_sketch_raw = abs_error_sketch_raw / true_counts
        rel_error_sketch_cutoff_raw = abs_error_sketch_cutoff_raw / true_counts
      
        print("L1 loss (sketch)  " + str(np.sum(abs_error_sketch_raw)))
        print("L2 loss (sketch)  " + str(np.sum(abs_error_sketch_raw**2)))

        # grouped abs and relative errors 
        abs_error_oracle = np.array_split(abs_error_oracle_raw, N)
        rel_error_oracle = np.array_split(rel_error_oracle_raw, N)
    
        abs_error_algo = np.array_split(abs_error_algo_raw, N)
        rel_error_algo = np.array_split(rel_error_algo_raw, N)
        abs_error_algo_cutoff = np.array_split(abs_error_algo_cutoff_raw, N)
        rel_error_algo_cutoff = np.array_split(rel_error_algo_cutoff_raw, N)

        abs_error_sketch = np.array_split(abs_error_sketch_raw, N)
        rel_error_sketch = np.array_split(rel_error_sketch_raw, N)
        abs_error_sketch_cutoff = np.array_split(abs_error_sketch_cutoff_raw, N)
        rel_error_sketch_cutoff = np.array_split(rel_error_sketch_cutoff_raw, N)

        grouped_abs_error_oracle = np.array([np.mean(x) for x in abs_error_oracle])
        grouped_rel_error_oracle = np.array([np.mean(x) for x in rel_error_oracle]) 
        grouped_abs_error_oracle_std = np.array([np.std(x) for x in abs_error_oracle])
        grouped_rel_error_oracle_std = np.array([np.std(x) for x in rel_error_oracle])
    
        grouped_abs_error_algo = np.array([np.mean(x) for x in abs_error_algo])
        grouped_rel_error_algo = np.array([np.mean(x) for x in rel_error_algo]) 
        grouped_abs_error_algo_std = np.array([np.std(x) for x in abs_error_algo])
        grouped_rel_error_algo_std = np.array([np.std(x) for x in rel_error_algo])

        grouped_abs_error_algo_cutoff = np.array([np.mean(x) for x in abs_error_algo_cutoff])
        grouped_rel_error_algo_cutoff = np.array([np.mean(x) for x in rel_error_algo_cutoff]) 
        grouped_abs_error_algo_cutoff_std = np.array([np.std(x) for x in abs_error_algo_cutoff])
        grouped_rel_error_algo_cutoff_std = np.array([np.std(x) for x in rel_error_algo_cutoff])

        grouped_abs_error_sketch = np.array([np.mean(x) for x in abs_error_sketch])
        grouped_rel_error_sketch = np.array([np.mean(x) for x in rel_error_sketch]) 
        grouped_abs_error_sketch_std = np.array([np.std(x) for x in abs_error_sketch])
        grouped_rel_error_sketch_std = np.array([np.std(x) for x in rel_error_sketch])

        grouped_abs_error_sketch_cutoff = np.array([np.mean(x) for x in abs_error_sketch_cutoff])
        grouped_rel_error_sketch_cutoff = np.array([np.mean(x) for x in rel_error_sketch_cutoff]) 
        grouped_abs_error_sketch_cutoff_std = np.array([np.std(x) for x in abs_error_sketch_cutoff])
        grouped_rel_error_sketch_cutoff_std = np.array([np.std(x) for x in rel_error_sketch_cutoff])

        grouped_true_counts = np.array([np.mean(x) for x in np.array_split(true_counts, N)])
        grouped_oracle_counts = np.array([np.mean(x) for x in np.array_split(pred_counts, N)])
        s = np.argsort(pred_counts)
        grouped_true_counts_sorted = np.array([np.mean(x) for x in np.array_split(true_counts[s][::-1], N)])
        grouped_sketch_counts = np.array([np.mean(x) for x in np.array_split(count_sketch_predictions, N)])

        ###################################################
        # make the absolute error plot 
        ###################################################
       
        # x axis for the plot
        x_range = (np.array(range(N)) / N) 

        #ax_abs.plot(x_range, grouped_abs_error_oracle, color=color_alt, label="Oracle Prediction", linestyle='--')
        ax_abs.plot(x_range, grouped_abs_error_algo, color=colors[1], label="Learned Count Sketch")
        ax_abs.plot(x_range, grouped_abs_error_sketch, color=colors[0], label="Count Sketch")
        ax_abs.plot(x_range, grouped_abs_error_algo_cutoff, color=colors[2], label="Learned Count Sketch + cutoff", linestyle='-')
        ax_abs.plot(x_range, grouped_abs_error_sketch_cutoff, color=colors[3], label="Count Sketch + cutoff", linestyle='-')

        # compute standard error std/sqrt(n)
        # https://en.wikipedia.org/wiki/Standard_error
        std_err_algo = 1.96 * grouped_abs_error_algo_std / math.sqrt(sample_size)
        std_err_sketch = 1.96 * grouped_abs_error_sketch_std / math.sqrt(sample_size)

        ax_abs.fill_between(
            x_range,
            grouped_abs_error_algo - std_err_algo, 
            grouped_abs_error_algo + std_err_algo, 
            color=colors[0],
            alpha=0.2)
            
        ax_abs.fill_between(
            x_range,
            grouped_abs_error_sketch - std_err_sketch, 
            grouped_abs_error_sketch + std_err_sketch,
            color=colors[1],
            alpha=0.2)

        ax_abs.yaxis.grid(color=gridcolor, linestyle=linestyle)
        ax_abs.xaxis.grid(color=gridcolor, linestyle=linestyle)
        ax_abs.set_axisbelow(True)
        ax_abs.set_yscale('log')
        ax_abs.set_ylabel("Absolute Error")



        ###################################################
        # make the relative error plot 
        ###################################################

        # plot relative error
        #ax_rel.plot(x_range, grouped_rel_error_oracle, color=color_alt, label="Oracle Prediction", linestyle='--')
        ax_rel.plot(x_range, grouped_rel_error_algo, color=colors[0], label="Learned Count Sketch")
        ax_rel.plot(x_range, grouped_rel_error_sketch, color=colors[1], label="Count Sketch")
        ax_rel.plot(x_range, grouped_rel_error_algo_cutoff, color=colors[2], label="Learned Count Sketch + cutoff", linestyle='-')
        ax_rel.plot(x_range, grouped_rel_error_sketch_cutoff, color=colors[3], label="Count Sketch + cutoff", linestyle='-')

        # compute standard error std/sqrt(n)
        # https://en.wikipedia.org/wiki/Standard_error
        std_err_algo = grouped_rel_error_algo_std / math.sqrt(sample_size)
        std_err_sketch = 1.96 * grouped_rel_error_sketch_std / math.sqrt(sample_size)
    
        ax_rel.fill_between(
            x_range,
            grouped_rel_error_algo - std_err_algo, 
            grouped_rel_error_algo + std_err_algo, 
            color=colors[0],
            alpha=0.2,
        )
        ax_rel.fill_between(
            x_range,
            grouped_rel_error_sketch - std_err_sketch, 
            grouped_rel_error_sketch + std_err_sketch,
            color=colors[1],
            alpha=0.2,
        )
    
        ax_rel.yaxis.grid(color=gridcolor, linestyle=linestyle)
        ax_rel.xaxis.grid(color=gridcolor, linestyle=linestyle)
        ax_rel.set_axisbelow(True)
        ax_rel.set_yscale('log')
        ax_rel.set_ylabel("Relative Error")


        ax_true.plot(x_range, grouped_true_counts, label="True Frequency")
        ax_true.plot(x_range, grouped_oracle_counts, color=colors[0], label="Oracle Prediction", linestyle='--')
        ax_true.plot(x_range, grouped_true_counts_sorted, color=colors[1], label="Sorted by Oracle")
        ax_true.yaxis.grid(color=gridcolor, linestyle=linestyle)
        ax_true.xaxis.grid(color=gridcolor, linestyle=linestyle)
        ax_true.set_axisbelow(True)
        ax_true.set_yscale('log')
        ax_true.set_ylabel("Frequency")
        ax_true.set_xlabel('Cumulative Distribution')

        ax_abs.legend(loc='best')

    plt.legend(loc='best')
    plt.savefig(file_name)
