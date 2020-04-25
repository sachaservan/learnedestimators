import sys
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--data", type=str, default='')
    argparser.add_argument("--x_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--y_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--title", type=str, default='')
    argparser.add_argument("--algo", type=str, default='Alg')
    argparser.add_argument("--N", type=float, default=1, help='normalization factor (# item / total counts)')
    args = argparser.parse_args()
    
    N = args.N
    ax = plt.figure().gca()

    if args.data:
        data = np.load(args.data)

        space = data['space_list']
        loss_sketch = data['loss_vanilla']
        loss_learned = data['loss_learned_no_cutoff']
        loss_learned_cutoff = data['loss_learned_cutoff']
        loss_learned_perfect_oracle = data['loss_learned_perfect_oracle']

        ax.plot(space, loss_sketch*N, label="Count sketch")
        ax.plot(space, loss_learned*N, label=args.algo)
        ax.plot(space, loss_learned_perfect_oracle*N, label=args.algo + " (perfect oracle)", linestyle='dashed')
        ax.plot(space, loss_learned_cutoff*N, label=args.algo + " (with cutoff)", linestyle='dashdot')

    ax.set_yscale('log')
    ax.set_ylabel('Loss (L2)')
    ax.set_xlabel('Space (MB)')
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)

    title = 'Loss vs. Space (log scale)'
    ax.set_title(title)
    plt.legend(loc="upper right")


    fig, ax = plt.subplots(figsize=(12, 8))

    if args.data:
        data = np.load(args.data)
        space = data['space_list']
      
        percent_oracle_no_cutoff = data['percent_oracle_no_cutoff']
        percent_sketch_no_cutoff = np.subtract(np.ones(len(percent_oracle_no_cutoff)), np.array(percent_oracle_no_cutoff))
       
        percent_oracle_cutoff = data['percent_oracle_cutoff']
        percent_sketch_cutoff = np.subtract(np.ones(len(percent_oracle_cutoff)), np.array(percent_oracle_cutoff))
       
        percent_oracle_perfect = data['percent_oracle_perfect']
        percent_sketch_perfect = np.subtract(np.ones(len(percent_oracle_perfect)), np.array(percent_oracle_perfect))

        print(percent_oracle_no_cutoff)
        print(percent_oracle_cutoff)
        print(percent_oracle_perfect)

        x = np.arange(len(percent_oracle_no_cutoff))

        # Define bar width. We'll use this to offset the second bar.
        bar_width = 0.4

        # Same thing, but offset the x by the width of the bar.
        b2 = ax.bar(x + bar_width, percent_sketch_no_cutoff,
                    width=bar_width, label='Count Sketch')
                    
        # Note we add the `width` parameter now which sets the width of each bar.
        b1 = ax.bar(x, percent_oracle_no_cutoff,
                    width=bar_width, label='Oracle')


        ax.set_ylabel('Fraction of Predictions Selected')
        ax.set_xlabel('Space (MB)')
        plt.xticks(x + bar_width/2, space)
        plt.legend(loc='best')
        plt.show()
