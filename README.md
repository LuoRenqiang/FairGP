FairGP 

see 
```
./run.sh
```

or

```
python run_models.py --gpu 0 --dataset "credit" --model "FairGP" --n_patch 50 --num_hidden 64 --nlayer 1 --nheads 2 --sens_attr region --pe_dim 2 --feat_norm "row" --label_number 6000 --metric 4
```