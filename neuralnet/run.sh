
# === Internet Traffic ===

# training

# python3 run_ip_model.py \
#    --train ../data/equinix-chicago.dirA.20160121-130000.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130100.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130200.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130300.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130400.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130500.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130600.ports.npy \
#    --valid ../data/equinix-chicago.dirA.20160121-130700.ports.npy \
#            ../data/equinix-chicago.dirA.20160121-130800.ports.npy \
#    --test  ../data/equinix-chicago.dirA.20160121-130900.ports.npy \
#    --save exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512 --rnn_hiddens 64 --port_hiddens 16 8 --hiddens 32 32 --batch_size 512 --n_epoch 500 --lr 0.0001 --regress_min 65

# resume training from an earlier model checkpoint

#python3 run_ip_model.py \
#   --train ../data/equinix-chicago.dirA.20160121-130000.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130100.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130200.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130300.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130400.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130500.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130600.ports.npy \
#   --valid ../data/equinix-chicago.dirA.20160121-130700.ports.npy \
#           ../data/equinix-chicago.dirA.20160121-130800.ports.npy \
#   --test  ../data/equinix-chicago.dirA.20160121-130900.ports.npy \
#   --save exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_ep180 --rnn_hiddens 64 --port_hiddens 16 8 --hiddens 32 32 --batch_size 512 --n_epoch 200 --lr 0.0001 --regress_min 65 \
#   --resume model/exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_20200621-153848_ep180.69 --start_epoch 181 --eval_n 10

# inference

python3 run_ip_model.py \
    --train ../data/equinix-chicago.dirA.20160121-130000.ports.npy \
    --valid ../data/equinix-chicago.dirA.20160121-130700.ports.npy \
    --test  ../data/equinix-chicago.dirA.20160121-130900.ports.npy \
    --save pred_relative_ip --rnn_hiddens 64 --port_hiddens 16 8 --hiddens 32 32 --batch_size 512 --n_epoch 200 --lr 0.0001 --regress_min 65 \
    --resume model/exp20_ip_rnn_10min_r1-p2-h2_rmin65_ru64_bs512_20201006-091450_ep220.69 --evaluate


# === Search Query ===

# training

# python3 ./run_aol_model.py \
#    --train ../data/aol_0000_len60.npz \
#            ../data/aol_0001_len60.npz \
#            ../data/aol_0002_len60.npz \
#            ../data/aol_0003_len60.npz \
#            ../data/aol_0004_len60.npz \
#    --valid ../data/aol_0005_len60.npz \
#    --test  ../data/aol_0006_len60.npz \
#    --save exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra  --embed_size 64 --rnn_hidden 256 --hiddens 32 --batch_size 128 --n_epoch 500 --lr 0.0001 --word_max_len 60 --regress_actual --eval_n 10


# resume training from an earlier model checkpoint

# python3 ./run_aol_model.py \
#    --train ../data/aol_0000_len60.npz \
#            ../data/aol_0001_len60.npz \
#            ../data/aol_0002_len60.npz \
#            ../data/aol_0003_len60.npz \
#            ../data/aol_0004_len60.npz \
#    --valid ../data/aol_0005_len60.npz \
#    --test  ../data/aol_0006_len60.npz \
#    --save exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra  --embed_size 64 --rnn_hidden 256 --hiddens 32 --batch_size 128 --n_epoch 500 --lr 0.0001 --word_max_len 60 --regress_actual --eval_n 10 \
#    --resume model/exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20201019-092449_ep140.69 --evaluate

# inference 
tday=06
python3 ./run_aol_model.py \
   --train ../data/aol_0000_len60.npz \
   --valid ../data/aol_0005_len60.npz \
   --test  ../data/aol_00${tday}_len60.npz \
   --save pred_relative_aol --embed_size 64 --rnn_hidden 256 --hiddens 32 --batch_size 128 \
   --n_epoch 2000 --lr 0.0001 --word_max_len 60 --regress_actual \
   --resume model/exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20201019-092449_ep140.69 --evaluate
