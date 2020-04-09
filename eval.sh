
# === Internet Traffic ===
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 \
#    --save cmin_ip_1329_ru64 --n_workers 30 \
#    --test_data     ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data    ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test_result   paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --valid_result  paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz

# learned count sketch
python3 learned_eval.py \
   --space_list 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 2 4 \
   --n_hashes 5 \
   --save cmin_ip_1329_ru64 --n_workers 1 \
   --test_data    ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
   --valid_data   ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
   --test_result   paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
   --valid_result  paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz

# count min
# python3 learned_eval.py \
#    --space_list 1 \
#    --n_hashes 3 \
#    --save cmin_ip_1329_ru64 --n_workers 1 \
#    --test_data    ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data   ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test_result   paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --valid_result  paper_predictions/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --count_min

   
   
# === AOL ===

# learned count sketch
# python3 learned_eval.py \
#    --space_list 1 \
#    --n_hashes 5 \
#    --save cmin_ip_1329_ru64 --n_workers 1 \
#    --test_data     ./data/aol_0050_len60.npz \
#    --valid_data    ./data/aol_0005_len60.npz \
#    --test_result   paper_predictions/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol


#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#    --n_hashes 1 2 3 4 \
#    --save csketch_aol_tday50_u256 --n_workers 30 \
#    --test_data     ./aol/by_day/query_counts_day_0050.npz \
#    --valid_data    ./aol/by_day/query_counts_day_0005.npz \
#    --test_result   paper_predictions/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol

# python3 learned_eval.py \
#    --space_list 0.2 \
#    --n_hashes 3 4 5 6 7 \
#    --save cmin_ip_1329_ru64 --n_workers 30 \
#    --test_data     ./data/aol_0050_len60.npz \
#    --valid_data    ./data/aol_0050_len60.npz \
#    --test_result   paper_predictions/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol

