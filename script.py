import os

if not os.path.exists('output_simple'):
    os.makedirs('output_simple')

distrs = ['IID', 'OOD']
DGPs = ['anticasual', 'linear', 'plusminus', 'fortest', 'normal']
n = 1000

for distr in distrs:
    for DGP in DGPs:
        # Set the output filename
        filename = f"output_simple/{distr}_{DGP}_{n}.txt"

        cmd = f"python run_exp.py --model ERM --distr {distr} --DGP {DGP} -n {n} -d 10 --num_epochs 1000 --seed 0 > {filename}"
        print(f"Running command: {cmd}")

        os.system(cmd)
