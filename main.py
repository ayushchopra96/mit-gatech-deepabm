import argparse
import random
import numpy as np
import torch
import yaml
import os
import pandas as pd

import initialize_agents_networks
import model
from utils import get_time_stamped_outdir


# Parsing command line arguments
parser = argparse.ArgumentParser(description='PyTorch ABM Model for Covid 19.')
parser.add_argument('-p', '--params', help='Name of the yaml file with the parameters.')
parser.add_argument('-s', '--seed', type=int, help='Seed for python random, numpy and torch', default = 6666)
parser.add_argument('-n', '--num_runs', type=int, help='Number of runs', default = 1)
parser.add_argument('-f', '--results_file_postfix', help='Postfix to be appended to output dir for ease of interpretation', default = '')
args = parser.parse_args()


# Setting seed
print('Seed used for python random, numpy and torch is {}'.format(args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed) 

#Reading params
with open(args.params, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print('Error in reading parameters file')
        print(exc)

params['seed'] = args.seed
params['num_runs'] = args.num_runs
params['results_file_postfix'] = args.results_file_postfix

if params['use_gpu']:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#Optionally re-initializing agents and re-building networks
if params['type'] == "initialization":
    print("Calling initialization..")
    initialize_agents_networks.initialize_agents_networks(params)

    print("Initialization done.. ", " exiting")
    exit()
    print(" --- I SHOULD NOT BE HERE ----")

if params['use_quarantine_logic']:
    print('This is a case with quarantine logic')
else:
    print('This is a case without quarantine logic')

if params['use_den_logic']:
    print('This is a case with den logic')
else:
    print('This is a case without den logic')

outdir = get_time_stamped_outdir(
                [params['output_location']['parent_dir'],
                    params['output_location']['results_dir']], params['results_file_postfix']
        )
outfile = os.path.join(
        outdir, 
        params['genrerated_params_file_name']
    )
with open(outfile, 'w') as stream:
    try:
        yaml.dump(params, stream)
    except yaml.YAMLError as exc:
        print(exc)
for n in range(params['num_runs']):
    abm = model.TorchABMCovid(params, device)

    # for i in range(params['num_steps']):
    #     abm.step()
    # df = abm.collect_results()
    # outfile = os.path.join(outdir, '{}_agents_stages_summary_seed_{}.csv'.format(n, params['seed']))
    # df.to_csv(outfile, index=False)

    """
        Trainable
    """ 
    gt_df = pd.read_csv('./Data/data.csv',header=0)
    gt_df = gt_df.iloc[:params['num_steps'],:] # subset
    gt_new_infected = torch.tensor(gt_df['Confirmed_Cases'].values)
    gt_new_hosp = torch.tensor(gt_df['Hospitalizations'].values)
    gt_new_death = torch.tensor(gt_df['Deaths'].values)
    # import pdb
    # pdb.set_trace()

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, abm.net.parameters()))
    loss = torch.nn.MSELoss()
    for epoch in range(100):
        opt.zero_grad()
        for i in range(params['num_steps']):
            abm.step()
        results = abm.collect_results_trainable()
        import pdb
        loss = loss(gt_new_infected.float(),results['new_infected'])# +\
        pdb.set_trace()
            # loss(gt_new_hosp,results['new_hosp']) +\
            # loss(gt_new_death,results['new_death'])
        loss.backward()
    

    print(" ENDED RUN {}".format(n))
    print("-"*60)
