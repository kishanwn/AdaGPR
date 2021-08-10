python -u train_semi.py --data cora  --test --layer 64 --hidden 64 --GPR_coeff 3 --lamda 0.5 --alpha 0.1 --dropout 0.5 --wd1 1.0 --wd2 1e-4 --wd3 1e-1
python -u train_semi.py --data citeseer --layer 32 --GPR_coeff 16  --hidden 256 --lamda 0.5 --alpha 0.1 --dropout 0.1 --test --wd1 1.0 --wd2 1e-4 --wd3 1e-1


