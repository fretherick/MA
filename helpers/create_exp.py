# region ========== PACKAGES
import os
import time
from itertools import product
from copy import deepcopy
from pprint import pprint
# endregion

add_to_queue = True
SLEEP_SEC = 0
pipeline = 'simulation_experiments'

# region ========== DICTIONARY OF EXPERIMENT Table (in paper monthly)
dict_exp = {
    'common.verbose': [True],
    'common.dump': [True],
    'common.plot': [False],
    'common.save_plot': [True],

    # Simulation HMM parameters
    'simulation_experiments.hmm.params.k': [2],
    'simulation_experiments.hmm.params.nb_paths': [1000],
    'simulation_experiments.hmm.params.len_path': [500], # [250, 500, 1000, 2000],
    'simulation_experiments.hmm.params.nb_states': [2],
    'simulation_experiments.hmm.params.more_states': [True],
    'simulation_experiments.hmm.params.dt': [1], #[1, 5, 20],

    # Means/variances (you could also grid search these)
    'simulation_experiments.hmm.params.means': ['[0.0123, -0.0157]'],
    'simulation_experiments.hmm.params.variances': ['[0.00120409, 0.00605284]'],
    'simulation_experiments.hmm.params.initial_state_prob': ['[0.85, 0.15]'],
    'simulation_experiments.hmm.params.transition_matrix': ['[[0.9629, 0.0371], [0.2101, 0.7899]]'],

    # Simulation params
    'simulation_experiments.simulation.params.windows': ['[5, 13]'],
    'simulation_experiments.simulation.params.nb_states': [2],
    'simulation_experiments.simulation.params.eff_timesteps': [224],

    # Clustering method
    'simulation_experiments.clustering.model': ["jump"], #['hmm', 'som', 'jump'],

    # SOM jump parameters
    'simulation_experiments.clustering.som_jump.params.map_size': ['[2, 1]'],
    'simulation_experiments.clustering.som_jump.params.Tmax': [4.33752676631246],
    'simulation_experiments.clustering.som_jump.params.Tmin': [0.4980629901238591],
    'simulation_experiments.clustering.som_jump.params.batch_size': [512],
    'simulation_experiments.clustering.som_jump.params.iterations': [5010],
    'simulation_experiments.clustering.som_jump.params.lr': [0.00612170606578564],
    'simulation_experiments.clustering.som_jump.params.decay': ['exponential'],

    # Output paths
    'simulation_experiments.clustering.som_jump.params.dump_path': [
        './artifacts/simulation_experiments/'
    ]
}


def apply_rules(dict_cmd):
    return deepcopy(dict_cmd)  # Placeholder if you want to auto-sync/transform params


def main():
    # parse dict_exp
    list_params, list_values = zip(*dict_exp.items())

    # build list of experiments
    list_cmd = []
    for i, comb in enumerate(product(*list_values)):
        dict_cmd = {p: v for p, v in zip(list_params, comb)}
        dict_cmd = apply_rules(dict_cmd)
        list_cmd.append(dict_cmd)

    # submit to queue:
    for doc in list_cmd:
        cmd = ["dvc exp run"]

        for k, v in doc.items():
            cmd.append(f'-S "{k}={v}"')

        cmd.append("--queue")
        if pipeline:
            cmd.append(f"--pipeline {pipeline}")

        _cmd = " ".join(cmd)

        if add_to_queue:
            os.system(_cmd)
            time.sleep(SLEEP_SEC)
        else:
            print("\n", _cmd)

    print(f"\nTotal number of simulations to run: {len(list_cmd)}")


if __name__ == '__main__':
    main()
