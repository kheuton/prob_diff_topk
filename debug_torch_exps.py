import os 



if __name__ == '__main__':

    print(os.getcwd())
    if os.path.exists('run_neg_binom.slurm'):
        print(os.path.abspath('run_neg_binom.slurm'))
    else:
        print('no path')
    