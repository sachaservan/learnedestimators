import sys
import time
import argparse
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
###########################


# PLOT PARAMETERS
###########################
# font = {'family': 'sans-serif',
#         'sans-serif': ['Tahoma', 'DejaVu Sans', ' Lucida Grande', 'Verdana'],
#         'size': 14}

# matplotlib.rc('font', **font)

colors = ["#003f5c", "#ff6361", "#58508d", "#0091d4", "#D7005C"]
colors_alt = ['#003f5c', '#2f4b7c', '#665191' ,'#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
color_alt = '#000000'
color_alt2 = '#a05195'
color_alt3 = '#f95d6a'

# more colors:
# https://www.color-hex.com/color-palettes/
edgecolor = '#34495e'
gridcolor = '#2f3640'
linestyle = 'dotted'
opacity = 1

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--data", type=str, default='')
    argparser.add_argument("--aol", action='store_true', default=False)
    argparser.add_argument("--synth", action='store_true', default=False)
    argparser.add_argument("--N", type=int, default=100)
    args = argparser.parse_args()

    N = args.N

    file_name = "experiments/err_ip.pdf"
    if args.aol:
        file_name= "experiments/err_aol.pdf"
    if args.synth:
        file_name = "experiments/err_synth.pdf"


    if args.data:

        subset = -1 # show results only up to index; set to index value
        data = np.load(args.data,  allow_pickle=True)
        true_counts = np.array(data['true_values'])[:subset]
        pred_counts = np.array(data['oracle_predictions'])[:subset]
        space_list = data['space_list']
        pred_counts = np.exp(pred_counts)

 
        # sort the estimates by true freuqency 
        sort = true_counts.argsort()
        true_counts = true_counts[sort][::-1]
        pred_counts = pred_counts[sort][::-1]

        num_data_points = len(true_counts)

        # plot absolute error
        fig, (ax_abs, ax_rel) = plt.subplots(2)
        # ax_abs.invert_xaxis()
        # ax_rel.invert_xaxis()

        if args.aol:
            fig.suptitle('AOL Dataset')
        elif args.synth:
            fig.suptitle('Synthetic Dataset')
        else:
            fig.suptitle('IP Dataset')


        sample_size = len(true_counts) / N


        for space_index in range(len(space_list)): 

            if space_index != len(space_list)-1: # space_index != 0: # 
                continue

            # already sorted
            print(data['space_list'][space_index])
            algo_predictions = np.array(data['valid_algo_predictions'])[space_index][:subset]
            count_sketch_predictions = np.array(data['valid_count_sketch_predictions'])[space_index][:subset]
            loss_per_partition = np.array(data['valid_loss_per_partition'])[space_index][:subset]

            # ax = plt.figure().gca()
            # ax.bar(range(len(loss_per_partition)), loss_per_partition)
            # ax.set_ylim([0, np.max(loss_per_partition)+100])
            # ax.set_xlabel('bucket id')
            # ax.set_ylabel('# packets')
            # plt.show()

            algo_predictions_cutoff = np.zeros(len(algo_predictions))
            # if len(np.array(data['valid_algo_predictions_cutoff'])) != 0:
            #     algo_predictions_cutoff = np.array(data['valid_algo_predictions_cutoff'])[space_index][:subset]

            count_sketch_predictions_cutoff = np.zeros(len(algo_predictions))
            # if len(np.array(data['valid_count_sketch_predictions_cutoff'])) != 0:
            #     count_sketch_predictions_cutoff = np.array(data['valid_count_sketch_predictions_cutoff'])[space_index][:subset]

            # algo_predictions += 000.1 # avoid division by zero
            # count_sketch_predictions += 000.1 # avoid division by zero
            # count_sketch_predictions_cutoff += 000.1 # avoid division by zero
            # algo_predictions_cutoff += 000.1 # avoid dividion by zero

            # sort the predictions according to the true frequency
            algo_predictions = algo_predictions[sort][::-1]
            algo_predictions_cutoff = algo_predictions_cutoff[sort][::-1]
            count_sketch_predictions = count_sketch_predictions[sort][::-1]
            count_sketch_predictions_cutoff = count_sketch_predictions_cutoff[sort][::-1]

            # compute the abs and relative erros for all items
            abs_error_oracle_raw = np.abs(true_counts - pred_counts) 
            rel_error_oracle_raw = abs_error_oracle_raw / true_counts

            abs_error_algo_raw = np.abs(true_counts - algo_predictions) 
            print("L1 loss (learned) " + str(np.sum(abs_error_algo_raw)))
            print("L2 loss (learned) " + str(np.sum(abs_error_algo_raw**2)))

            abs_error_algo_cutoff_raw = np.abs(true_counts - algo_predictions_cutoff) 
            rel_error_algo_raw = abs_error_algo_raw / true_counts
            rel_error_algo_cutoff_raw = abs_error_algo_cutoff_raw / true_counts

            abs_error_sketch_raw = np.abs(true_counts - count_sketch_predictions) 
            abs_error_sketch_cutoff_raw = np.abs(true_counts - count_sketch_predictions_cutoff) 
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
            ax_abs.plot(x_range, grouped_abs_error_algo, color=color_alt2, label="Learned Count Sketch")
            ax_abs.plot(x_range, grouped_abs_error_sketch, color=color_alt3, label="Count Sketch")
            ax_abs.plot(x_range, grouped_abs_error_algo_cutoff, label="Learned Count Sketch + cutoff", linestyle='-')
            ax_abs.plot(x_range, grouped_abs_error_sketch_cutoff, label="Count Sketch + cutoff", linestyle='-')

            # compute standard error std/sqrt(n)
            # https://en.wikipedia.org/wiki/Standard_error
            std_err_algo = 1.96 * grouped_abs_error_algo_std / math.sqrt(sample_size)
            std_err_sketch = 1.96 * grouped_abs_error_sketch_std / math.sqrt(sample_size)

            ax_abs.fill_between(
                x_range,
                grouped_abs_error_algo - std_err_algo, 
                grouped_abs_error_algo + std_err_algo, 
                color=color_alt2,
                alpha=0.2)
                
            ax_abs.fill_between(
                x_range,
                grouped_abs_error_sketch - std_err_sketch, 
                grouped_abs_error_sketch + std_err_sketch,
                color=color_alt3,
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
            ax_rel.plot(x_range, grouped_rel_error_algo, color=color_alt2, label="Learned Count Sketch")
            ax_rel.plot(x_range, grouped_rel_error_sketch, color=color_alt3, label="Count Sketch")
            ax_rel.plot(x_range, grouped_rel_error_algo_cutoff, label="Learned Count Sketch + cutoff", linestyle='-')
            ax_rel.plot(x_range, grouped_rel_error_sketch_cutoff, label="Count Sketch + cutoff", linestyle='-')


            # compute standard error std/sqrt(n)
            # https://en.wikipedia.org/wiki/Standard_error
            std_err_algo = grouped_rel_error_algo_std / math.sqrt(sample_size)
            std_err_sketch = 1.96 * grouped_rel_error_sketch_std / math.sqrt(sample_size)
        
            ax_rel.fill_between(
                x_range,
                grouped_rel_error_algo - std_err_algo, 
                grouped_rel_error_algo + std_err_algo, 
                color=color_alt2,
                alpha=0.2,
            )
            ax_rel.fill_between(
                x_range,
                grouped_rel_error_sketch - std_err_sketch, 
                grouped_rel_error_sketch + std_err_sketch,
                color=color_alt3,
                alpha=0.2,
            )
        
            ax_rel.yaxis.grid(color=gridcolor, linestyle=linestyle)
            ax_rel.xaxis.grid(color=gridcolor, linestyle=linestyle)
            ax_rel.set_axisbelow(True)
            ax_rel.set_yscale('log')
            ax_rel.set_ylabel("Relative Error")


            # ax_true.plot(x_range, grouped_true_counts, label="True Frequency")
            # #ax_true.plot(x_range, grouped_oracle_counts, color=color_alt, label="Oracle Prediction", linestyle='--')
            # ax_true.plot(x_range, grouped_true_counts_sorted, color=color_alt3, label="Sorted by Oracle")
            # ax_true.yaxis.grid(color=gridcolor, linestyle=linestyle)
            # ax_true.xaxis.grid(color=gridcolor, linestyle=linestyle)
            # ax_true.set_axisbelow(True)
            # ax_true.set_yscale('log')
            ax_rel.set_ylabel("Frequency")
            ax_rel.set_xlabel('Cumulative Distribution')

            ax_abs.legend(loc='best')

        plt.legend(loc='best')
        plt.savefig(file_name)
