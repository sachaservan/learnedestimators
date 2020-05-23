# === INTERNET TRAFFIC ===

# find optimal parameters
python3 experiments.py \
   --space_list 0.2 0.3 0.5 0.6 0.8 0.9 1 1.5 2 3 4 4.3 \
   --test_dataset data/equinix-chicago.dirA.20160121-132900.ports.npy \
   --model paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
   --save_file ip_learned_sketch_optimal_params \
   --save_folder experiments \
   --n_workers 10 \
   --run_learned_version \
   --run_regular_count_sketch \
  #--run_cutoff_version \

# run validation with optimal test params 
python3 experiments.py \
   --valid_dataset data/equinix-chicago.dirA.20160121-130700.ports.npy \
   --model paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
   --optimal_params experiments/ip_learned_sketch_optimal_params.npz \
   --save_folder experiments \
   --save_file ip_learned_sketch_experiment_results \
   --n_workers 10 \
   --run_learned_version \
   --run_regular_count_sketch \
#  --run_cutoff_version \


# === AOL ===
# python3 experiments.py \
#    --aol_data \
#    --space_list 0.05 0.1 0.2 1 \
#    --n_hashes 5 \
#    --save_file aol_learned_sketch_experiment_results \
#    --save_folder experiments \
#    --n_workers 1 \
#    --dataset  ./data/aol_0005_len60.npz \
#    --model  paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --run_regular_count_sketch \
#    --run_learned_version \
   # --run_cutoff_version \


# === SYNTHETIC===
# python3 experiments.py \
#    --synth_data \
#    --space_list 0.1 0.2 \
#    --n_hashes 5 \
#    --save_file synth_learned_sketch_experiment_results \
#    --save_folder experiments \
#    --n_workers 1 \
#    --run_regular_count_sketch \
#    --run_learned_version \


# python3 plot_oracle_acc.py --data /Users/Sacha/Desktop/learnedsketch/experiments/aol_learned_sketch_experiment_results.npz --N 50 --aol
# python3 plot_oracle_acc.py --data /Users/Sacha/Desktop/learnedsketch/experiments/ip_learned_sketch_experiment_results.npz --N 100
# python3 plot_loss.py --data /Users/Sacha/Desktop/learnedsketch/experiments/aol_learned_sketch_experiment_results.npz --algo "learned cs" --aol
# python3 plot_loss.py --data /Users/Sacha/Desktop/learnedsketch/experiments/ip_learned_sketch_experiment_results.npz --algo "learned cs"



# === Internet Traffic ===
#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 \
#    --n_hashes 1 2 3 4 \
#    --save cmin_ip_1329_ru64 --n_workers 30 \
#    --test_data     ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data    ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test_result   paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --valid_result  paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz


# python3 experiments.py \
#    --aol_data \
#    --space_list 0.2 1.2 \
#    --n_hashes 5 \
#    --save_file aol_learned_sketch_experiment_results \
#    --save_folder experiments \
#    --n_workers 1 \
#    --dataset  ./data/aol_0005_len60.npz \
#    --model  paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --run_regular_count_sketch \
#    --run_learned_version \
  # --run_cutoff_version \

# count min
# python3 experiments.py \
#    --space_list 1 \
#    --n_hashes 3 \
#    --save cmin_ip_1329_ru64 --n_workers 1 \
#    --test_data    ./data/equinix-chicago.dirA.20160121-132900.ports.npy \
#    --valid_data   ./data/equinix-chicago.dirA.20160121-130700.ports.npy \
#    --test_result   paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --valid_result  paper_model/pred_exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep350_1329_res.npz \
#    --count_min

   
   
# === AOL ===

# learned count sketch
# python3 experiments.py \
#    --space_list 1 \
#    --n_hashes 5 \
#    --save cmin_ip_1329_ru64 --n_workers 1 \
#    --test_data     ./data/aol_0050_len60.npz \
#    --valid_data    ./data/aol_0005_len60.npz \
#    --test_result   paper_model/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol


#    --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#    --n_hashes 1 2 3 4 \
#    --save csketch_aol_tday50_u256 --n_workers 30 \
#    --test_data     ./aol/by_day/query_counts_day_0050.npz \
#    --valid_data    ./aol/by_day/query_counts_day_0005.npz \
#    --test_result   paper_model/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol

# python3 experiments.py \
#    --space_list 0.2 \
#    --n_hashes 3 4 5 6 7 \
#    --save cmin_ip_1329_ru64 --n_workers 30 \
#    --test_data     ./data/aol_0050_len60.npz \
#    --valid_data    ./data/aol_0050_len60.npz \
#    --test_result   paper_model/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --valid_result  paper_model/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#    --aol

