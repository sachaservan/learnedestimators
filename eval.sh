
# === Internet Traffic ===

## count min

#python3 count_min_space_vs_loss.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 \
#    --save cmin_ip_1329 --n_workers 30 \
#    --data ./data/equinix-chicago.dirA.20160121-132900.ports.npy

## lookup table + count min

# python3 cutoff_count_min_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 --save cmin_ip_1329 --n_workers 40 \
#    --test_data  ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --lookup     ./data/equinix-chicago.dirA.20160121-130000.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130100.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130200.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130300.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130400.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130500.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130600.ports.npy

# learned count min with hashing

python3 learned_eval.py \
   --space_list 1 \
   --n_hashes 3 4 5 6 7 \
   --save cmin_ip_1329_ru64 --n_workers 30 \
   --test_data     ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
   --valid_data    ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
   --test_result   paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
   --valid_result  paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz


## learned count min

# python3 cutoff_count_min_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 \
#    --save cmin_ip_1329_ru64 --n_workers 30 \
#    --test_data     ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data    ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test_result   paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --valid_result  paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz

## learned count min + perfect oracle

# python3 cutoff_count_min_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 --save cmin_ip_1329_pcut --n_workers 30 \
#    --test_data  ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --perfect


## example command to get predictions from a trained model

# python3 run_nn_cls.py \
# 	--train ./data/equinix-chicago.dirA.20160121-130000.ports.npy \
#    --valid ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test  ./data/equinix-chicago.dirA.20160121-135900.ports.npy \
#    --save  pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1359 \
#    --rnn_hiddens 64 --port_hiddens 16 8 --hiddens 32 32 --batch_size 512 --n_epoch 10 --lr 0.0001 --regress_min 65 \
#    --resume model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1359_20191212-161307_ep0.69 --evaluate


## count sketch

#python3 count_sketch_space_vs_loss.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 \
#    --save csketch_ip_1329 --n_workers 30 \
#    --data ./data/equinix-chicago.dirA.20160121-132900.ports.npy

## lookup table + Count Sketch

#python3 cutoff_count_sketch_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 --save csketch_ip_1329 --n_workers 40 \
#    --test_data  ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --lookup     ./data/equinix-chicago.dirA.20160121-130000.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130100.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130200.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130300.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130400.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130500.ports.npy \
#                 ./data/equinix-chicago.dirA.20160121-130600.ports.npy

## learned count sketch

#python3 cutoff_count_sketch_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 \
#    --save csketch_ip_1329_ru64 --n_workers 30 \
#    --test_data     ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data    ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test_result   paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --valid_result  paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz

## learned count sketch + perfect oracle

#python3 cutoff_count_sketch_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 --save csketch_ip_1329_pcut --n_workers 30 \
#    --test_data  ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --perfect
#


# === AOL ===

## Note: replace "sketch" to "min" to generate results for algorithms with count-min

#python3 count_sketch_space_vs_loss.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#    --n_hashes 1 2 3 4 \
#    --save csketch_aol_tday50 --n_workers 30 \
#    --data ./aol/by_day/query_counts_day_0050.npz \
#    --aol
#
#
#python3 cutoff_count_sketch_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#    --n_hashes 1 2 3 4 --save csketch_aol_tday50 --n_workers 40 --aol \
#    --test_data     ./aol/by_day/query_counts_day_0050.npz \
#    --valid_data    ./aol/by_day/query_counts_day_0005.npz \
#    --lookup        ./aol/by_day/query_counts_day_0000.npz \
#                    ./aol/by_day/query_counts_day_0001.npz \
#                    ./aol/by_day/query_counts_day_0002.npz \
#                    ./aol/by_day/query_counts_day_0003.npz \
#                    ./aol/by_day/query_counts_day_0004.npz
#
#
#python3 cutoff_count_sketch_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#    --n_hashes 1 2 3 4 \
#    --save csketch_aol_tday50_u256 --n_workers 30 \
#    --test_data     ./aol/by_day/query_counts_day_0050.npz \
#    --valid_data    ./aol/by_day/query_counts_day_0005.npz \
#    --test_result   paper_predictions/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol
#
#
#python3 cutoff_count_sketch_param.py \
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#    --n_hashes 1 2 3 4 --save csketch_aol_tday50_pcut --n_workers 30 --aol \
#    --test_data  ./aol/by_day/query_counts_day_0050.npz \
#    --valid_data ./aol/by_day/query_counts_day_0050.npz \
#    --perfect
