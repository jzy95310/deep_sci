import sys
sys.path.insert(0, '../../src/')
sys.path.insert(0, '../../data/geospatial_data/')
import argparse
import wandb
import dill as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import torch
from torch.utils.data import DataLoader

from data_generator import train_val_test_split, train_val_test_split_gps
from trainers import Trainer, GPSModelTrainer
from gps import GeneralizedPropensityScoreModel
from models import NonlinearSCI

# Experimental tracking
wandb.login()

np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
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
    scaler = StandardScaler()
    targets = scaler.fit_transform(targets.reshape(-1,1))
    window_size = interventions[0].shape[-1]
    intervention_min, intervention_max, num_bins = np.min(interventions[0]), np.max(interventions[0]), 100

    train_dataset, val_dataset, test_dataset, test_indices = train_val_test_split(
        interventions, confounder, spatial_features, targets, 
        train_size=0.6, val_size=0.2, test_size=0.2, shuffle=False, 
        block_sampling=True, return_test_indices=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    
    # Load data for generalized propensity score (GPS) model
    train_dataset_gps, val_dataset_gps, _ = train_val_test_split_gps(
        interventions[0][:,window_size//2,window_size//2][:,np.newaxis], confounder, spatial_features, 
        train_size=0.8, val_size=0.2, test_size=0.0, shuffle=True
    )
    train_loader_gps = DataLoader(train_dataset_gps, batch_size=128, shuffle=True)
    val_loader_gps = DataLoader(val_dataset_gps, batch_size=128, shuffle=False)
    dataloaders_gps = {'train': train_loader_gps, 'val': val_loader_gps, 'test': None}
    
    # GPS model
    gps_model = GeneralizedPropensityScoreModel(
        input_dim=confounder.shape[1]+spatial_features.shape[1],
        num_hidden_layers=1,
        hidden_dims=128
    )
    
    # Train GPS model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = "sgd"
    optim_params = {
        'lr': 1e-5, 
        'momentum': 0.99
    }
    epochs, patience = args.n_epochs, args.patience
    trainer = GPSModelTrainer(
        model=gps_model, 
        data_generators=dataloaders_gps, 
        optim=optim, 
        optim_params=optim_params, 
        device=device,
        epochs=epochs,
        patience=patience
    )
    trainer.train()
    
    # Stabilized weights model
    sw_model = gaussian_kde(interventions[0][:,window_size//2,window_size//2])
    
    # Model definition
    model = NonlinearSCI(
        num_interventions=len(interventions), 
        window_size=interventions[0].shape[-1],  
        confounder_dim=confounder.shape[-1], 
        f_network_type="unet", 
        f_depth=3, 
        f_batch_norm=False, 
        f_padding=1, 
        f_dropout_ratio=0.0, 
        g_network_type="dk_mlp",
        g_dropout_ratio=0.0, 
        g_num_basis=4, 
        g_hidden_dims=[128],
        unobserved_confounder=False
    )
    model.initialize_weights(method="xavier")
    
    # Experimental tracking
    wandb.init(
        project="deep_sci",
        name="geospatial_semi_f_unet_g_dk_mlp_without_U_with_GPS", 
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
        window_size=window_size, 
        gps_model=gps_model, 
        sw_model=sw_model, 
        device=device,
        epochs=epochs,
        patience=patience, 
        wandb=wandb
    )
    trainer.train()
    
    # Prediction
    y_direct_pred = trainer.predict(mode='direct', t_min=intervention_min, t_max=intervention_max, 
                                    num_bins=num_bins, weighting='snipw')
    y_indirect_pred = trainer.predict(mode='indirect', t_min=intervention_min, t_max=intervention_max, 
                                      num_bins=num_bins, weighting='snipw')
    y_total_pred = trainer.predict(mode='total', t_min=intervention_min, t_max=intervention_max, 
                                     num_bins=num_bins, weighting='snipw')
    y_direct_pred = scaler.inverse_transform(y_direct_pred).squeeze()
    y_indirect_pred = scaler.inverse_transform(y_indirect_pred).squeeze()
    y_total_pred = scaler.inverse_transform(y_total_pred).squeeze()
    de_pred = np.mean(np.mean(y_direct_pred, axis=1), axis=0)
    ie_pred = np.mean(np.mean(y_indirect_pred, axis=1), axis=0)
    te_pred = np.mean(np.mean(y_total_pred, axis=1), axis=0)
    
    # Evaluatioon
    de_true, ie_true, te_true = data_generator.calc_causal_effects(
        test_indices, intervention_min, intervention_max, num_bins
    )
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
    arg_parser = argparse.ArgumentParser(description='f: U-Net, g: DeepKriging with MLP, without unobserved confounder')
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--optim_name', type=str, default="sgd")
    arg_parser.add_argument('--lr', type=float, default=1e-6)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--weight_decay', type=float, default=0.0)
    arg_parser.add_argument('--n_epochs', type=int, default=1000)
    arg_parser.add_argument('--patience', type=int, default=20)
    args = arg_parser.parse_known_args()[0]
    main(args)
