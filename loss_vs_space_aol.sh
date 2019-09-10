

tday=50
fname="50 th"

model="u256"

python3 plot_loss_vs_space.py \
    --algo          "Count-Sketch" \
    --count_min     param_results/count_sketch/csketch_aol_tday${tday}.npz \
    --learned       param_results/cutoff_csketch_param/csketch_aol_tday${tday}_${model}_test.npz \
    --perfect       param_results/cutoff_csketch_param_perfect/csketch_aol_tday${tday}_pcut_test.npz  \
    --lookup_table  param_results/lookup_table_csketch/csketch_aol_tday${tday}_test.npz \
    --model_names   "Learned Count-Sketch (NNet)" \
    --title         "AOL - @ ${fname} test day - model ${model}" \
    --N             0.56 \
    --model_sizes   0.0152 \
    --lookup_size   0.0009 \
    --x_lim         0.08 1.2 \
    --y_lim         0 4

