import numpy as np
import sys

sys.path.insert(0,'./src/')
sib_folder = "../sib" # sib path
abm_folder = "../OpenABM-Covid19/src" #Open ABM path
sys.path.insert(0,sib_folder)
sys.path.insert(0,abm_folder)

from pathlib import Path
import log, logging
from importlib import reload
import loop_abm, abm_utils, scipy
from rankers import dotd_rank, tracing_rank, mean_field_rank
import matplotlib.pyplot as plt
import plot_utils
import time
from collections import defaultdict
import pickle
import os


output_dir = "./output/"
fold_out = Path(output_dir)
if not fold_out.exists():
    fold_out.mkdir(parents=True)


reload(log)
logger = log.setup_logger()



N=5000 #Number of individuals 50000
T=15 #Total time of simulations  100
n_seed_infection = 10 #number of patient zero

fraction_SM_obs = 0.5 #fraction of Symptomatic Mild tested positive
fraction_SS_obs = 1 #fraction of Symptomatic Severe tested positive
initial_steps = 12 #starting time of intervention
quarantine_HH = True #Households quarantine
test_HH = True #Tests the households when quarantined
adoption_fraction = 1 #app adoption (fraction)
num_test_random = 0 #number of random tests per day
num_test_algo = int(sys.argv[1]) #number of tests using by the ranker per day
fp_rate = 0.0 #test false-positive rate
fn_rate = 0.0 #test false-negative rate

n_repeats = 2
delays = [1] # test delayed MF with a delays from 1 to max_delay days

prob_seed = 1/N
prob_sus = 0.55
pseed = prob_seed / (2 - prob_seed)
psus = prob_sus * (1 - pseed)
pautoinf = 1/N

rankers = {
    "RG": dotd_rank.DotdRanker()
'''
    "MF": mean_field_rank.MeanFieldRanker(
        tau=5,
        delta=10,
        mu=1 / 30,
        lamb=0.014,
        delay=0
    ),
    '''
    "CT": tracing_rank.TracingRanker(
        tau=5,
        lamb=0.014
    )
}
'''
_delayed_MF_rankers = {
    f"MF-D{d}": mean_field_rank.MeanFieldRanker(
        tau = 5,
        delta = 10,
        mu = 1/30,
        lamb = 0.014,
        delay = d)
    for d in delays
}

rankers.update(_delayed_MF_rankers)
'''
_delayed_CT_rankers = {
    f"CT-D{d}": mean_field_rank.MeanFieldRanker(
        tau = 5,
        lamb = 0.014,
        delay = d)
    for d in delays
}

rankers.update(_delayed_CT_rankers)

if not os.path.exists(f'/home/ghio/epidemic_mitigation/partial_data_{num_test_algo}'):
    os.makedirs(f'/home/ghio/epidemic_mitigation/partial_data_{num_test_algo}')

repeats = []
for i in range(n_repeats):
    seed = np.random.randint(10000)
    repeat = defaultdict(list)

    for s, ranker in rankers.items():
        params_model = {
            "rng_seed": seed,
            "end_time": T,
            "n_total": N,
            "days_of_interactions": T,
            "n_seed_infection": n_seed_infection,
        }

        plots = plot_utils.plot_style(N, T)
        save_path_fig = f"./output/plot_run_N_{N}_SM_{fraction_SM_obs}_test_{num_test_algo}_n_seed_infection_{n_seed_infection}_seed_{seed}_fp_{fp_rate}_fn_{fn_rate}.png"
        fig, callback = plot_utils.plotgrid(rankers, plots, initial_steps, save_path=save_path_fig)

        data = {"algo": s}

        loop_abm.loop_abm(
            params_model,
            ranker,
            seed=seed,
            logger=logging.getLogger(f"iteration.{s}"),
            data=data,
            callback=callback,
            initial_steps=initial_steps,
            num_test_random=num_test_random,
            num_test_algo=num_test_algo,
            fraction_SM_obs=fraction_SM_obs,
            fraction_SS_obs=fraction_SS_obs,
            quarantine_HH=quarantine_HH,
            test_HH=test_HH,
            adoption_fraction=adoption_fraction,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            name_file_res= s + f"_N_{N}_T_{T}_obs_{num_test_algo}_SM_obs_{fraction_SM_obs}_seed_{seed}"
        )

        repeat[s] = data
        current_idx = max([0] + [int(p.split("_")[1].split(".")[0]) for p in os.listdir(f"/home/ghio/epidemic_mitigation/partial_data_{num_test_algo}") if s in p])
        with open(f"/home/ghio/epidemic_mitigation/partial_data_{num_test_algo}/{s}_{current_idx+1}.pkl", "wb+") as f_out:
            f_out.write(pickle.dumps(data))
    repeats.append(repeat)

to_plot = 'I'
plt.clf()

for i, s in enumerate(rankers.keys()):
    avg = np.mean([r[s][to_plot] for r in repeats], axis=0)
    plt.plot(avg, label=s, color=plt.cm.Set1(i), linewidth=1)
    for r in repeats:
        plt.plot(r[s][to_plot], color=plt.cm.Pastel1(i), alpha=0.7, linewidth=0.5)

plt.semilogy()
plt.ylabel("Infected")
plt.xlabel("days")
plt.legend()
plt.savefig(f"Delayed_Intervention_{num_test_algo}.png")