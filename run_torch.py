from src.dataset import create_multiple_circles
from src.torch_ripsnet import DenseNestedTensors, PermopNestedTensors

import logging
import gudhi as gd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from matplotlib import pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from gudhi.representations import DiagramSelector
from gudhi.representations import Landscape, PersistenceImage




def run():
    #### Create dataset ####
    N_sets_train = 900  # Number of train point clouds
    N_sets_test = 300  # Number of test  point clouds
    N_points = 600  # Point cloud cardinality
    N_noise = 200  # Number of corrupted points

    data_train, label_train = create_multiple_circles(N_sets_train, N_points, noisy=0, N_noise=N_noise)
    # clean_data_test, clean_label_test = create_multiple_circles(N_sets_test,  N_points, noisy=0, N_noise=N_noise)
    # noisy_data_test, noisy_label_test = create_multiple_circles(N_sets_test,  N_points, noisy=1, N_noise=N_noise)
    ds = [pairwise_distances(X).flatten() for X in data_train[:30]]
    maxd = np.max(np.concatenate(ds))

    # train PD
    PD_train = []
    for X in tqdm(data_train):
        st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
        st.persistence()
        dg = st.persistence_intervals_in_dimension(1)
        if len(dg) == 0:
            dg = np.empty([0,2])
        PD_train.append(dg)

    # # test PD
    # clean_PD_test = []
    # for X in tqdm(clean_data_test):
    #     st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
    #     st.persistence()
    #     dg = st.persistence_intervals_in_dimension(1)
    #     if len(dg) == 0:
    #         dg = np.empty([0,2])
    #     clean_PD_test.append(dg)


    # # noisy PD
    # noisy_PD_test = []
    # for X in tqdm(noisy_data_test):
    #     st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
    #     st.persistence()
    #     dg = st.persistence_intervals_in_dimension(1)
    #     if len(dg) == 0:
    #         dg = np.empty([0,2])
    #     noisy_PD_test.append(dg)

    PVs_train, clean_PVs_test, noisy_PVs_test, PVs_params = [], [], [], []
    pds_train = DiagramSelector(use=True).fit_transform(PD_train)
    # clean_pds_test = DiagramSelector(use=True).fit_transform(clean_PD_test)
    # noisy_pds_test = DiagramSelector(use=True).fit_transform(noisy_PD_test)

    vpdtr = np.vstack(pds_train)
    pers = vpdtr[:,1]-vpdtr[:,0]
    bps_pairs = pairwise_distances(np.hstack([vpdtr[:,0:1],vpdtr[:,1:2]-vpdtr[:,0:1]])[:200]).flatten()
    ppers = bps_pairs[np.argwhere(bps_pairs > 1e-5).ravel()]
    sigma = np.quantile(ppers, .2)
    im_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,0],1.), np.quantile(pers,0.), np.quantile(pers,1.)]
    sp_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,1],1.)]

    ### Persistence Images
    PI_params = {'bandwidth': sigma, 
                'weight': lambda x: 10*np.tanh(x[1]), 
                'resolution': [50,50], 
                'im_range': im_bnds
                }
    PI_train = PersistenceImage(**PI_params).fit_transform(pds_train)
    # clean_PI_test = PersistenceImage(**PI_params).fit_transform(clean_pds_test)
    # noisy_PI_test = PersistenceImage(**PI_params).fit_transform(noisy_pds_test)


    MPI = np.max(PI_train)
    PI_train /= MPI
    # clean_PI_test /= MPI
    # noisy_PI_test /= MPI

    ### Persistence Landscapes
    PL_params = {'num_landscapes': 5, 'resolution': 300, 'sample_range': sp_bnds}
    PL_train = Landscape(**PL_params).fit_transform(pds_train)
    # clean_PL_test = Landscape(**PL_params).fit_transform(clean_pds_test)
    # noisy_PL_test = Landscape(**PL_params).fit_transform(noisy_pds_test)
    MPL = np.max(PL_train)
    PL_train /= MPL
    # clean_PL_test /= MPL
    # noisy_PL_test /= MPL

    ##### TRAINING #####  

    ## convert to torch  
    train_data = torch.nested.nested_tensor(data_train, dtype=torch.float32)
    # clean_test_set = torch.nested.nested_tensor(clean_data_test)
    # noisy_test_set = torch.nested.nested_tensor(noisy_data_test)

    PI_train = torch.from_numpy(PI_train).float()

    ripsnet = nn.Sequential(
        DenseNestedTensors(30, last_dim=2, use_bias=True),
        DenseNestedTensors(20, last_dim=30, use_bias=True),
        DenseNestedTensors(10, last_dim=20,use_bias=True),
        PermopNestedTensors(),
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, PI_train.shape[1]),
        nn.Sigmoid()
        )
    optim = torch.optim.Adam(ripsnet.parameters(), lr=1e-3)
    loss = nn.MSELoss()

    ## train loop 
    for epoch in range(10):
        optim.zero_grad()
        out = ripsnet(train_data)
        l = loss(out, PI_train)
        l.backward()
        optim.step()
        print(f"Epoch {epoch} - Loss: {l.item()}")


if __name__ == "__main__":
    run()