import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def get_best_loss_space(data):
    rshape = (len(data['space_list']), -1)
    best_param_idx = np.argmin(data['loss_all'].reshape(rshape), axis=1)
    loss = data['loss_all'].reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    space_actual = data['spaec_actual'].reshape(rshape)[np.arange(rshape[0]), best_param_idx] / 1e6
    return loss, space_actual

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
       # loss_learned_cutoff = data['loss_learned_cutoff']

        ax.plot(space, loss_sketch*N, label="count sketch")
        ax.plot(space, loss_learned*N, label=args.algo)
        # ax.plot(space, loss_learned_cutoff*N, label=args.algo + " + cutoff")

    ax.set_yscale('log')
    ax.set_ylabel('loss')
    ax.set_xlabel('space (MB)')
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)

    title = 'loss vs space (log scale)'
    ax.set_title(title)
    plt.legend(loc="upper right")
    plt.show()

