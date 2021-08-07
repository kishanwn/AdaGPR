python -u train_full.py --data cora --layer 64 --GPR_coeffs 3 --lamda 0.5 --alpha 0.1 --dropout 0.5 --weight_decay 1e-4
python -u train_full.py --data citeseer --layer 64 --GPR_coeffs 2 --lamda 0.5 --alpha 0.4 --dropout 0.7 --weight_decay 1e-4
python -u train_full.py --data chameleon --layer 2 --GPR_coeffs 3 --lamda 1.5 --alpha 0.6  --dropout 0.6 --weight_decay 1e-3
python -u train_full.py --data cornell --layer 4 --GPR_coeffs 2 --lamda 1.0 --alpha 0.9 --dropout 0.4  --weight_decay 1e-4
python -u train_full.py --data texas --layer 4 --GPR_coeffs 4 --lamda 1.0 --alpha 0.5 --dropout 0.5  --weight_decay 5e-4
python -u train_full.py --data wisconsin --layer 16 --GPR_coeffs 3 --lamda 1.5 --alpha 0.6  --dropout 0.3 --weight_decay 5e-5
