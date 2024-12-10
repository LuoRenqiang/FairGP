## FairGP: A Scalable and Fair Graph Transformer Using Graph Partitioning 

see how to run FairGP
```
./run.sh
```

or try an example

```
python run_models.py --gpu 0 --dataset "credit" --model "FairGP" --n_patch 50 --num_hidden 64 --nlayer 1 --nheads 2 --sens_attr region --pe_dim 2 --feat_norm "row" --label_number 6000 --metric 4
```

### Cite
```
@INPROCEEDINGS{fairgp2025luo,
  author={Luo, Renqiang and Huang, Huafei and Lee, Ivan and Xu, Chengpei and Qi, Jianzhong and Xia, Feng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)}, 
  title={FairGP: A Scalable and Fair Graph Transformer Using Graph Partitioning}, 
  year={2025},
}
```