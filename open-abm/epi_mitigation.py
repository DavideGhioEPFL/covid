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

#logging
output_dir = "./output/"
fold_out = Path(output_dir)
if not fold_out.exists():
    fold_out.mkdir(parents=True)

reload(log)
logger = log.setup_logger()

num_threads = 40 #number of threads used for sib



N=5000 #Number of individuals 50000
T=20 #Total time of simulations  100
seed = np.random.randint(1000) #seed of the random number generator
n_seed_infection = 10 #number of patient zero

params_model = {
    "rng_seed" : seed,
    "end_time" : T,
    "n_total"  : N,
    "days_of_interactions" : T,
    "n_seed_infection" : n_seed_infection,
}


fraction_SM_obs = 0.5 #fraction of Symptomatic Mild tested positive
fraction_SS_obs = 1 #fraction of Symptomatic Severe tested positive
initial_steps = 12 #starting time of intervention
quarantine_HH = True #Households quarantine
test_HH = True #Tests the households when quarantined
adoption_fraction = 1 #app adoption (fraction)
num_test_random = 0 #number of random tests per day
num_test_algo = 200 #number of tests using by the ranker per day
fp_rate = 0.0 #test false-positive rate
fn_rate = 0.0 #test false-negative rate
n_repeats = 2

from rankers import dotd_rank, tracing_rank, mean_field_rank


prob_seed = 1/N
prob_sus = 0.55
pseed = prob_seed / (2 - prob_seed)
psus = prob_sus * (1 - pseed)
pautoinf = 1/N


dotd = dotd_rank.DotdRanker()


tracing = tracing_rank.TracingRanker(
                 tau=5,
                 lamb=0.014
)

MF = mean_field_rank.MeanFieldRanker(
                tau = 5,
                delta = 10,
                mu = 1/30,
                lamb = 0.014,
                delay = 0
                )

ress = {}

rankers = {
    # "RG" : dotd,
    "CT": tracing,
    "MF": MF
}

import matplotlib.pyplot as plt
import plot_utils
import time

import imp
imp.reload(loop_abm)

repeats = []
for i in range(n_repeats):
    tracing = tracing_rank.TracingRanker(
        tau=5,
        lamb=0.014
    )

    MF = mean_field_rank.MeanFieldRanker(
        tau=5,
        delta=10,
        mu=1 / 30,
        lamb=0.014,
        delay=0
    )
    rankers = {
        # "RG" : dotd,
        #"CT": tracing,
        "MF": MF
    }
    '''
    rankers.update({f"MF-D{i}": mean_field_rank.MeanFieldRanker(
        tau = 5,
        delta = 10,
        mu = 1/30,
        lamb = 0.014,
        delay = i) for i in range(1, 2)})
    '''

    plots = plot_utils.plot_style(N, T)
    save_path_fig = f"./output/plot_run_N_{N}_SM_{fraction_SM_obs}_test_{num_test_algo}_n_seed_infection_{n_seed_infection}_seed_{seed}_fp_{fp_rate}_fn_{fn_rate}.png"
    fig, callback = plot_utils.plotgrid(rankers, plots, initial_steps, save_path=save_path_fig)

    for s in rankers:
        new_seed = np.random.randint(1000)
        params_model['rng_seed'] = new_seed
        data = {"algo":s}
        imp.reload(loop_abm)
        loop_abm.loop_abm(
            params_model,
            rankers[s],
            seed=new_seed,
            logger = logging.getLogger(f"iteration.{s}"),
            data = data,
            callback = callback,
            initial_steps = initial_steps,
            num_test_random = num_test_random,
            num_test_algo = num_test_algo,
            fraction_SM_obs = fraction_SM_obs,
            fraction_SS_obs = fraction_SS_obs,
            quarantine_HH = quarantine_HH,
            test_HH = test_HH,
            adoption_fraction = adoption_fraction,
            fp_rate = fp_rate,
            fn_rate = fn_rate,
            name_file_res = s + f"_N_{N}_T_{T}_obs_{num_test_algo}_SM_obs_{fraction_SM_obs}_seed_{seed}"
        )
        ress[s] = data
        # saves a bit of memory: rankers[s] = {}
    repeats.append(ress)


plt.clf()
to_plot = "I"
for i, s in enumerate(repeats[0].keys()):
    avg = np.mean([ress[s][to_plot] for ress in repeats], axis=0)
    plt.plot(avg, label = s, color=plt.cm.Set1(i))
    for ress in repeats:
        print(ress[s][to_plot])
        plt.plot(ress[s][to_plot], color=plt.cm.Pastel1(i))

plt.semilogy()
plt.ylabel("Infected")
plt.xlabel("days")
plt.legend()
plt.savefig(f"Delayed_Intervention_{num_test_algo}.png")