import sys
sys.path.insert(0, '../../src/')
sys.path.insert(0, '../../data/geospatial_data/')
import argparse
import wandb
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_generator import train_val_test_split
from models import NonlinearSCI
from trainers import Trainer
from spatial_dataset_semisynthetic import SpatialDataset

# Experimental tracking
wandb.login()

np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True


def main(args):
    # Load and re-format data
    with open('../../data/geospatial_data/pickle_files/durham_semisynthetic.pkl', 'rb') as f:
        files = pkl.load(f)
    data, data_generator = files["data"], files["data_generator"]
    interventions = [np.array([x[3] for x in data])]
    confounder = np.array([x[2] for x in data])
    spatial_features = np.array([[x[0],x[1]] for x in data])
    targets = np.array([x[4] for x in data])

    train_dataset, val_dataset, test_dataset, test_indices = train_val_test_split(
        interventions, confounder, spatial_features, targets, 
        train_size=0.6, val_size=0.2, test_size=0.2, shuffle=False, 
        block_sampling=True, return_test_indices=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    
    # Model definition
    model = NonlinearSCI(
        num_interventions=1, 
        window_size=interventions[0].shape[-1],  
        confounder_dim=confounder.shape[-1], 
        f_network_type="dk_convnet", 
        f_kernel_size=3,
        g_network_type="dk_mlp", 
        g_hidden_dims=[128],
        unobserved_confounder=False
    )
    
    # Experimental tracking
    wandb.init(
        project="deep_sci",
        name="geospatial_f_dk_cnn_g_dk_mlp_without_U", 
        allow_val_change=True
    )
    config = wandb.config
    
    # Training
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
        patience=patience, 
        wandb=wandb
    )
    trainer.train()
    
    # Prediction
    y_pred_00 = trainer.predict(window_size=interventions[0].shape[-1], direct=False, indirect=False)
    y_pred_01 = trainer.predict(window_size=interventions[0].shape[-1], direct=False, indirect=True)
    y_pred_10 = trainer.predict(window_size=interventions[0].shape[-1], direct=True, indirect=False)
    y_pred_11 = trainer.predict(window_size=interventions[0].shape[-1], direct=True, indirect=True)
    de_pred = np.mean(y_pred_11 - y_pred_01)
    ie_pred = np.mean(y_pred_01 - y_pred_00)
    te_pred = np.mean(y_pred_11 - y_pred_00)
    
    # Evaluatioon
    de_true, ie_true, te_true = data_generator.calc_causal_effects(test_indices)
    print(f"Error on direct effect: {np.abs(de_true - de_pred):.5f}")
    print(f"Error on indirect effect: {np.abs(ie_true - ie_pred):.5f}")
    print(f"Error on total effect: {np.abs(te_true - te_pred):.5f}")

    # Update wandb
    wandb.config.update({
        "de_error": np.abs(de_true - de_pred),
        "ie_error": np.abs(ie_true - ie_pred),
        "te_error": np.abs(te_true - te_pred)
    })

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='f: DeepKriging with CNN, g: DeepKriging with MLP, without unobserved confounder')
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--optim_name', type=str, default="sgd")
    arg_parser.add_argument('--lr', type=float, default=1e-5)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--n_epochs', type=int, default=1000)
    arg_parser.add_argument('--patience', type=int, default=20)
    args = arg_parser.parse_known_args()[0]
    main(args)