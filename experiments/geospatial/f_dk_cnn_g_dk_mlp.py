import sys
sys.path.insert(0, '../../src/')
import argparse
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_generator import train_val_test_split
from models import NonlinearSCI
from trainers import Trainer

np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True


def main(args):
    with open('../../data/dataset_1.pkl', 'rb') as f:
        data = pkl.load(f)
    ndvi = [np.array([x[3] for x in data])]   # intervention
    nlcd = np.array([x[2] for x in data])     # confounder
    spatial_features = np.array([abs(x[0]-x[1]) for x in data])
    targets = np.array([x[4] for x in data])

    train_dataset, val_dataset, test_dataset = train_val_test_split(
        ndvi, nlcd, spatial_features, targets, train_size=0.6, val_size=0.2, test_size=0.2, shuffle=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    min_dist, max_dist = np.min(spatial_features), np.max(spatial_features)
    model = NonlinearSCI(
        num_interventions=1, 
        window_size=ndvi[0].shape[-1], 
        confounder_dim=nlcd.shape[-1], 
        f_network_type="dk_convnet", 
        g_network_type="dk_mlp", 
        unobserved_confounder=True, 
        nys_space=[[min_dist,max_dist]]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = args.optim_name
    optim_params = {
        'lr': args.lr,
        'momentum': args.momentum
    }
    epochs, patience = args.n_epochs, args.patience
    trainer = Trainer(
        model=model, 
        data_generators=dataloaders, 
        optim=optim, 
        optim_params=optim_params, 
        device=device,
        epochs=epochs,
        patience=patience
    )
    trainer.train()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='f: DeepKriging with CNN, g: DeepKriging with MLP')
    arg_parser.add_argument('--batch_size', type=int, default=16)
    arg_parser.add_argument('--optim_name', type=str, default="sgd")
    arg_parser.add_argument('--lr', type=float, default=1e-6)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--n_epochs', type=int, default=1000)
    arg_parser.add_argument('--patience', type=int, default=10)
    args = arg_parser.parse_known_args()[0]
    main(args)