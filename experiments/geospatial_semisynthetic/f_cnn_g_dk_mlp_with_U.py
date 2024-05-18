import sys
sys.path.insert(0, '../../src/')
sys.path.insert(0, '../../data/geospatial_data/')
import argparse
import wandb
import dill as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from data_generator import train_val_test_split
from models import NonlinearSCI
from trainers import Trainer
from spatial_dataset_semisynthetic import SpatialDataset

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
    
    # Model definition
    model = NonlinearSCI(
        num_interventions=len(interventions), 
        window_size=interventions[0].shape[-1],  
        confounder_dim=confounder.shape[-1], 
        f_network_type="convnet", 
        f_batch_norm=False, 
        f_dropout_ratio=0.0, 
        g_network_type="dk_mlp",
        g_dropout_ratio=0.0, 
        g_num_basis=4, 
        g_hidden_dims=[128],
        unobserved_confounder=True, 
        kernel_func="rbf", 
        kernel_param_vals=[1.,5e-3,0.1], 
        inducing_point_space=[[0.,1.],[0.,1.]]
    )
    model.initialize_weights(method="xavier")
    
    # Experimental tracking
    wandb.init(
        project="deep_sci",
        name="geospatial_semi_f_cnn_g_dk_mlp_with_U", 
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
        device=device,
        epochs=epochs,
        patience=patience, 
        wandb=wandb
    )
    trainer.train()
    
    # Prediction
    y_direct_pred = trainer.predict(mode='direct', t_min=intervention_min, t_max=intervention_max, 
                                    num_bins=num_bins)
    y_indirect_pred = trainer.predict(mode='indirect', t_min=intervention_min, t_max=intervention_max, 
                                      num_bins=num_bins)
    y_total_pred = trainer.predict(mode='total', t_min=intervention_min, t_max=intervention_max, 
                                     num_bins=num_bins)
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
    arg_parser = argparse.ArgumentParser(description='f: CNN, g: DeepKriging with MLP, with unobserved confounder')
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--optim_name', type=str, default="sgd")
    arg_parser.add_argument('--lr', type=float, default=1e-6)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--weight_decay', type=float, default=0.0)
    arg_parser.add_argument('--n_epochs', type=int, default=1000)
    arg_parser.add_argument('--patience', type=int, default=20)
    args = arg_parser.parse_known_args()[0]
    main(args)
