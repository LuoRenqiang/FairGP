

python run_models.py --gpu 0 --dataset "credit" --model "FairGP" --n_patch 50 --num_hidden 64 --nlayer 1 --nheads 2 --sens_attr region --pe_dim 2 --feat_norm "row" --label_number 6000 --metric 4

python run_models.py --gpu 0 --dataset "pokec_z" --model "FairGP" --n_patch 300 --num_hidden 64 --nlayer 2 --nheads 1 --sens_attr region --pe_dim 8 --feat_norm "row" --label_number 1000 --metric 4

python run_models.py --gpu 0 --dataset "pokec_z" --model "FairGP" --n_patch 100 --num_hidden 64 --nlayer 2 --nheads 1 --sens_attr gender --pe_dim 1 --feat_norm "row" --label_number 1000 --metric 4

python run_models.py --gpu 0 --dataset "pokec_n" --model "FairGP" --n_patch 250 --num_hidden 64 --nlayer 2 --nheads 1 --sens_attr region --pe_dim 16 --feat_norm "row" --label_number 1000 --metric 4

python run_models.py --gpu 0 --dataset "pokec_n" --model "FairGP" --n_patch 200 --num_hidden 32 --nlayer 2 --nheads 2 --sens_attr gender --pe_dim 9 --feat_norm "row" --label_number 1000 --metric 4

python run_models.py --gpu 0 --dataset "aminer_l" --model "FairGP" --n_patch 50 --num_hidden 16 --nlayer 2 --nheads 1 --sens_attr gender --pe_dim 8 --feat_norm "none" --label_number 5000 --metric 4