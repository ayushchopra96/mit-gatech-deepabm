DOSAGE_VALUE=08

python -u main.py --params Data/generated_params-trainable.yaml --seed 1234 --num_runs 1 
python -u main.py --params Data/generated_params-trainable.yaml --seed 1234 --num_runs 1 --results_file_postfix VAC_100_1FEXPT08  > logs/tmp_fig1_1f_expt_100vac.txt &