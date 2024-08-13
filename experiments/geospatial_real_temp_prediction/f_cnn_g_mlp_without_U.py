import sys, os
sys.path.insert(0, '../../src/')
sys.path.insert(0, '../../data/geospatial_data/')
import argparse
from tqdm import tqdm
import dill as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from data_generator import train_val_test_split
from models import NonlinearSCI
from trainers import Trainer

np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True


def main(args):
    # Load and re-format data
    with open('../../data/geospatial_data/pickle_files/durham_real_train.pkl', 'rb') as f:
        data = pkl.load(f)
    interventions = [np.array([x[3] for x in data]), np.array([x[4] for x in data])]
    confounder = np.array([x[2].mean(axis=(0,1)) for x in data])
    spatial_features = np.array([[x[0],x[1]] for x in data])
    targets = np.array([x[5] for x in data])
    scaler = StandardScaler()
    targets = scaler.fit_transform(targets.reshape(-1,1))
    window_size = interventions[0].shape[-1]

    train_dataset, val_dataset, test_dataset = train_val_test_split(
        interventions, confounder, spatial_features, targets, 
        train_size=0.7, val_size=0.3, test_size=0.0, shuffle=False, 
        block_sampling=True
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
        g_network_type="mlp", 
        g_hidden_dims=[128], 
        unobserved_confounder=False
    )
    model.initialize_weights(method="xavier")
    
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
        patience=patience
    )
    trainer.train()
    
    # Inference
    with open('../../data/geospatial_data/pickle_files/durham_real_test.pkl', 'rb') as f:
        test_data = pkl.load(f)
        data_generator = test_data['data_generator']
        height, width = test_data['height'], test_data['width']
        print(f"Map height: {height}, map width: {width}")
    
    # Visualization of predicted change in temperature over distance
    assert args.inference_x - args.inference_window_size >= 0 and \
        args.inference_x + args.inference_window_size < height
    assert args.inference_y - args.inference_window_size >= 0 and \
        args.inference_y + args.inference_window_size < width
    
    y_map = []
    with torch.no_grad():
        for i in tqdm(range(200,2201), position=0, leave=True):
            batch = [[torch.empty(0).to(device)]*len(interventions),torch.empty(0).to(device),torch.empty(0).to(device)]
            for j in range(200,2201):
                sample = data_generator.get_item_by_coords(i,j)
                s = torch.tensor([sample[0],sample[1]]).float().view(1,-1).to(device)
                nlcd = torch.tensor(sample[2]).mean(dim=(0,1)).float().view(1,-1).to(device)
                ndvi = torch.tensor(sample[3]).float().view(1,sample[3].shape[0],-1).to(device)
                albedo = torch.tensor(sample[4]).float().view(1,sample[3].shape[0],-1).to(device)
                batch[0][0] = torch.cat((batch[0][0],ndvi),dim=0)
                batch[0][1] = torch.cat((batch[0][1],albedo),dim=0)
                batch[1] = torch.cat((batch[1],nlcd),dim=0)
                batch[2] = torch.cat((batch[2],s),dim=0)
                del sample, s, nlcd, ndvi, albedo
            y_pred = model.predict(*batch).cpu().numpy()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1,1)).squeeze()
            y_map.append(y_pred)
            
    y_map = np.array(y_map)
    if os.path.exists('./results_geospatial_real.pkl'):
        with open('./results_geospatial_real.pkl', 'rb') as fp:
            res = pkl.load(fp)
        res['f_cnn_g_mlp_without_U'] = y_map
    else:
        res = {'f_cnn_g_mlp_without_U': y_map}
    with open('./results_geospatial_real.pkl', 'wb') as fp:
        pkl.dump(res, fp)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='f: CNN, g: MLP, without unobserved confounder')
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--optim_name', type=str, default="sgd")
    arg_parser.add_argument('--lr', type=float, default=1e-6)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--weight_decay', type=float, default=0.0)
    arg_parser.add_argument('--n_epochs', type=int, default=1000)
    arg_parser.add_argument('--patience', type=int, default=20)
    arg_parser.add_argument('--inference_x', type=int, default=1000)
    arg_parser.add_argument('--inference_y', type=int, default=1000)
    arg_parser.add_argument('--inference_window_size', type=int, default=50)
    args = arg_parser.parse_known_args()[0]
    main(args)