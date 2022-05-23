import pickle
import pandas as pd
from os import walk
from ast import literal_eval
import numpy as np

base_path = "/work3/s174437/replays/"

replays = list(range(6,12))
rp = []

def generate_combined_trajectories(): # format data as [(p1,p2),(p1,p2),...]
    for i in replays:
        for (dirpath, dirnames, filenames) in walk(f"{base_path}/{i}/"):
            for file in filenames:
                if ".gz" in file:
                    d = {}
                    df = pd.read_csv(f"{base_path}/{i}/{file}", compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)
                    d["rewards"] = np.array([literal_eval(x)[0] for x in df["rewards"][1:]])
                    d["actions"] = np.array([literal_eval(x)[0] for x in df["actions"][1:]])
                    d["observations"] = np.array([literal_eval(x)[0] for x in df["obs"][:-1]])
                    d["next_observations"] = np.array([literal_eval(x)[0] for x in df["obs"][1:]])
                    d["states"] = np.array([literal_eval(x) for x in df["state"][:-1]])
                    d["next_states"] = np.array([literal_eval(x) for x in df["state"][1:]])
                    d["terminals"] = np.array([x for x in df["terminated"][1:]])
                    rp.append(d)

    with open('gym/data/lbforaging_old.pkl', 'wb') as handle:
        pickle.dump(rp, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_separate_trajectories(): # format data as [[p1],[p2]]
    for i in replays:
        for (_, _, filenames) in walk(f"{base_path}/{i}/"):
            for file in filenames:
                if ".gz" in file:
                    d = {}
                    df = pd.read_csv(f"{base_path}/{i}/{file}", compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)
                    n_players = len(literal_eval(df["rewards"][0]))
                    d["rewards"] = np.array([[literal_eval(x)[j] for x in df["rewards"][1:]] for j in range(n_players)])
                    d["actions"] = np.array([[literal_eval(x)[j] for x in df["actions"][1:]] for j in range(n_players)])
                    d["observations"] = np.array([[literal_eval(x)[j] for x in df["obs"][:-1]] for j in range(n_players)])
                    d["next_observations"] = np.array([[literal_eval(x)[j] for x in df["obs"][1:]] for j in range(n_players)])
                    d["states"] = np.array([literal_eval(x) for x in df["state"][:-1]])
                    d["next_states"] = np.array([literal_eval(x) for x in df["state"][1:]])
                    d["terminals"] = np.array([x for x in df["terminated"][1:]])
                    rp.append(d)

    with open('gym/data/lbforaging_separate-medium-v2.pkl', 'wb') as handle:
        pickle.dump(rp, handle, protocol=pickle.HIGHEST_PROTOCOL)

generate_combined_trajectories()