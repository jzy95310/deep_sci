# models.py: a file containing all models for nonlinear spatial causal inference
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn

from typing import List
from nn import MLP, ConvNet, DeepKrigingMLP, DeepKrigingConvNet
from unet import UNet
from ick.kernels.kernel_fn import *
from ick.model.ick import ICK

KERNEL_FUNCS = {
    'rbf': sq_exp_kernel_nys, 
    'periodic': periodic_kernel_nys, 
    'exp': exp_kernel_nys,
    'rq': rq_kernel_nys, 
    'matern_type1': matern_type1_kernel_nys,
    'matern_type2': matern_type2_kernel_nys, 
    'linear': linear_kernel_nys,
    'spectral_mixture': spectral_mixture_kernel_1d_nys
}

KERNEL_PARAMS = {
    'rbf': ['std','lengthscale','noise'], 
    'periodic': ['std','period','lengthscale','noise'], 
    'exp': ['std','lengthscale','noise'],
    'rq': ['std','lengthscale','scale_mixture','noise'], 
    'matern_type1': ['std','lengthscale','noise'],
    'matern_type2': ['std','lengthscale','noise'], 
    'linear': ['std','c','noise'],
    'spectral_mixture': ['weight','mean','cov','noise']
}


class LinearSCI(nn.Module):
    """
    Defines a linear model for spatial causal inference
    y = beta_1 * T1 + ... + beta_m + Tm + gamma_1 * T1_bar + ... + gamma_m * Tm_bar + alpha * X + U + epsilon
    where T is the intervention variable, T_bar is the "spatially weighted" neighboring interventions, 
    X is the confounder, U is the unobserved confounder, epsilon is the noise term, m is the number of interventions,
    and betas, gammas, and alpha are the coefficients for the linear model. The weight function for the neighboring
    interventions is the Gaussian kernel function with a trainable lengthscale parameter.

    Arguments
    --------------
    num_interventions: int, number of interventions
    window_size: int, grid size for neighboring interventions T_bar
    confounder_dim: int, dimension of the confounder X
    unobserved_confounder: bool, whether to include the unobserved confounder U, default to False
    dimensionality: int, the dimensionality of the window, either 1 or 2
    **kwargs: dict, additional keyword arguments for U
        - kernel_func: str, the kernel function to use for the ICK model
        - kernel_param_vals: List[float], the initial values for the kernel parameters
        - inducing_point_space: List[List[float]], the space of the inducing points for Nystrom approximation
    """
    def __init__(self, num_interventions: int, window_size: int, confounder_dim: int = None, 
                 unobserved_confounder: bool = False, dimensionality: int = 1, **kwargs) -> None:
        super(LinearSCI, self).__init__()
        self.num_interventions: int = num_interventions
        self.window_size: int = window_size
        self.confounder_dim: int = confounder_dim
        self.unobserved_confounder: bool = unobserved_confounder
        self.dimensionality: int = dimensionality
        self.kwargs: dict = kwargs
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Build the linear spatial causal inference model, where the interventions and confounders are modeled 
        using a linear relationship with the outcome and the unobserved confounder is modeled using a pseudo 
        Gaussian process, which is implemented by the Implicit Composite Kernel (ICK) model.

        References:
        Jiang, Ziyang, et al. "Incorporating prior knowledge into neural networks through an implicit 
        composite kernel." arXiv preprint arXiv:2205.07384 (2022).
        """
        assert self.dimensionality in [1,2], "The dimensionality should be either 1 or 2."
        for i in range(1,self.num_interventions+1):
            setattr(self, f"beta_{i}", nn.Parameter(torch.randn(1)))
            setattr(self, f"gamma_{i}", nn.Parameter(torch.randn(1)))
        if isinstance(self.confounder_dim, int):
            self.alpha = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(self.confounder_dim)])
        else:
            self.alpha = nn.Parameter(torch.randn(1))
        if self.dimensionality == 1:
            self.weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.window_size, 1)))
        else:
            self.weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.window_size, self.window_size)))
        if self.unobserved_confounder:
            kernel_assignment = ['ImplicitNystromKernel']
            kernel_params = {
                'ImplicitNystromKernel': {
                    'kernel_func': KERNEL_FUNCS[self.kwargs.get('kernel_func', 'rbf')], 
                    'params': KERNEL_PARAMS[self.kwargs.get('kernel_func', 'rbf')], 
                    'vals': self.kwargs.get('kernel_param_vals', [1.,1.,0.5]), 
                    'trainable': [True] * len(KERNEL_PARAMS[self.kwargs.get('kernel_func', 'rbf')]), 
                    'alpha': self.kwargs.get('alpha', 1e-5),
                    'num_inducing_points': self.kwargs.get('num_inducing_points', 32),
                    'nys_space': self.kwargs.get('inducing_point_space', [[0.,1.]])
                }
            }
            self.gp_unobserved_confounder = ICK(kernel_assignment, kernel_params)
    
    def forward(self, t: List[torch.Tensor], x: torch.Tensor, s: torch.Tensor = None, 
                return_components: bool = False) -> torch.Tensor:
        """
        t: List[torch.Tensor], a list of tensors containing the intervention variables with shape 
            (batch_size, window_size) or (batch_size, window_size, window_size)
        x: torch.Tensor, a tensor containing the confounder variable with shape (batch_size, confounder_dim) or
            (batch_size, window_size) or (batch_size, window_size, window_size)
        s: torch.Tensor, a tensor containing the spatial information (e.g., coordinates or distance) 
            of the training data, only used when unobserved_confounder is True
        """
        assert self.window_size % 2 == 1, "The window size should be odd."
        y_t, y_t_bar, y_x = [], [], []
        for i in range(self.num_interventions):
            if len(t[i].shape) == 2:
                ti = t[i][:,self.window_size//2]
            elif len(t[i].shape) == 3:
                ti = t[i][:,self.window_size//2,self.window_size//2]
            else:
                raise Exception(f"Invalid shape for intervention variable: {t[i].shape}")
            y_t.append(ti * getattr(self, f"beta_{i+1}"))
            if self.dimensionality == 1:
                # Weighted sum of neighboring interventions
                ti_bar = torch.sum(t[i][:,:self.window_size//2] * self.weights.view(-1)[:self.window_size//2], dim=1) + \
                    torch.sum(t[i][:,self.window_size//2+1:] * self.weights.view(-1)[self.window_size//2+1:], dim=1)
            elif self.dimensionality == 2:
                t_sq = t[i].reshape(t[i].shape[0],-1)
                mid_idx = t_sq.shape[-1] // 2
                ti_bar = torch.sum(t_sq[:,:mid_idx] * self.weights.view(-1)[:mid_idx], dim=1) + \
                    torch.sum(t_sq[:,mid_idx+1:] * self.weights.view(-1)[mid_idx+1:], dim=1)
            else:
                raise Exception(f"Invalid shape for weight: {self.weights.shape}")
            y_t_bar.append(ti_bar * getattr(self, f"gamma_{i+1}"))
        if len(x.shape) == 2:
            assert self.confounder_dim == x.shape[1], "The dimension of the confounder should match the input."
            for i in range(self.confounder_dim):
                y_x.append(x[:,i] * self.alpha[i])
        else:
            y_x.append(torch.sum(x,dim=tuple(i for i in range(1,len(x.shape)))) * self.alpha)
        output = torch.sum(torch.stack(y_t + y_t_bar + y_x), dim=0)
        if self.unobserved_confounder and len(s.shape) == 1:
            s = s.unsqueeze(1)
        if not return_components:
            return output if not self.unobserved_confounder else (output + self.gp_unobserved_confounder([s]))
        else:
            return (y_t,) + (y_t_bar,) + (y_x,) + ([self.gp_unobserved_confounder([s])],) if self.unobserved_confounder \
                else (y_t,) + (y_t_bar,) + (y_x,)
    
    def predict(self, t: List[torch.Tensor], x: torch.Tensor, s: torch.Tensor = None, 
                return_components: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(t, x, s, return_components)


class NonlinearSCI(nn.Module):
    """
    Defines a nonlinear model for spatial causal inference
    y = beta_1 * T1 + ... + beta_m + Tm + f_1(T1_bar) + ... + f_m(Tm_bar) + g(X) + U + epsilon
    where T is the intervention variable, T_bar is the neighboring interventions, X is the confounder,
    U is the unobserved confounder, epsilon is the noise term, and m is the number of interventions.
    Here both f and g are deep models, and U is modeled using the Implicit Composite Kernel (ICK).

    Arguments
    --------------
    num_interventions: int, number of interventions
    window_size: int, grid size for neighboring interventions T_bar
    confounder_dim: int, dimension of the confounder X
    f_network_type: str, the model type for f, default to "convnet", can be any of {"mlp", "convnet", 
        "dk_convnet", "unet"}
    g_network_type: str, the model type for g, default to "mlp", can be any of {"mlp", "convnet",
        "dk_mlp", "dk_convnet"}
    unobserved_confounder: bool, whether to include the unobserved confounder U, default to False
    **kwargs: dict, additional keyword arguments for f, g, and U
        arguments for f:
            - f_hidden_dims: List[int], the dimensions of hidden layers for the MLP for f, default to [128,64]
            - f_dense_hidden_dims: int, the dimensions of hidden dense layers for the convolutional neural 
            network for f, default to 128
            - f_kernel_size: int, the kernel size for the convolutional neural network for f, default to 7
            - f_stride: int, the stride for the convolutional neural network for f, default to 3
            - f_channels: int, number of input channels for the convolutional neural network for f, default to 1
            - f_activation: str, the activation function to use for the MLP for f, default to "relu"
            - f_dropout_ratio: float, the dropout ratio for the MLP for f, default to 0.0
            - f_num_basis: int, number of basis functions for the deep kriging model for f, default to 4
            - f_depth: int, the depth of the U-Net model for f, default to 2
            - f_padding: int, the padding size for the U-Net model for f, default to 1
        arguments for g:
            - g_hidden_dims: List[int], the dimensions of hidden layers for the MLP for g, default to [128,64]
            - g_dense_hidden_dims: List[int], the dimensions of hidden dense layers for the convolutional neural
            network for g, default to 128
            - g_kernel_size: int, the kernel size for the convolutional neural network for g, default to 7
            - g_channels: int, number of input channels for the convolutional neural network for g, default to 1
            - g_activation: str, the activation function to use for the MLP for g, default to "relu"
            - g_dropout_ratio: float, the dropout ratio for the MLP for g, default to 0.0
            - g_num_basis: int, number of basis functions for the deep kriging model for g, default to 4
        arguments for U:
            - kernel_func: str, the kernel function to use for the ICK model
            - kernel_param_vals: List[float], the initial values for the kernel parameters
            - inducing_point_space: List[List[float]], the space of the inducing points for Nystrom approximation
    """
    def __init__(self, num_interventions: int, window_size: int, confounder_dim: int = None, f_network_type: str = "mlp", 
                 g_network_type: str = "mlp", unobserved_confounder: bool = False, **kwargs) -> None:
        super(NonlinearSCI, self).__init__()
        self.num_interventions: int = num_interventions
        self.window_size: int = window_size
        self.confounder_dim: int = confounder_dim
        self.f_network_type: str = f_network_type
        self.g_network_type: str = g_network_type
        self.unobserved_confounder: bool = unobserved_confounder
        self.kwargs: dict = kwargs
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Build the nonlinear spatial causal inference model, where the intervention variables are modeled
        using a linear relationship, the spatial/non-spatial confounders are modeled using convolutional 
        neural networks, and the unobserved confounder is modeled using a pseudo Gaussian process, which
        is implemented using the Implicit Composite Kernel (ICK) model.

        References:
        Jiang, Ziyang, et al. "Incorporating prior knowledge into neural networks through an implicit 
        composite kernel." arXiv preprint arXiv:2205.07384 (2022).
        """
        for i in range(1,self.num_interventions+1):
            setattr(self, f"beta_{i}", nn.Parameter(torch.randn(1)))
        # model for f(T_bar)
        f_batch_norm = self.kwargs.get('f_batch_norm', False)
        f_activation = self.kwargs.get('f_activation', 'relu')
        f_dropout_ratio = self.kwargs.get('f_dropout_ratio', 0.0)
        if self.f_network_type == "mlp":
            f_hidden_dims = self.kwargs.get('f_hidden_dims', [128,64])
            for i in range(1,self.num_interventions+1):
                setattr(self, f"f_{i}", MLP(
                    self.window_size, len(f_hidden_dims), f_hidden_dims, 
                    batch_norm=f_batch_norm, p_dropout=f_dropout_ratio, activation=f_activation
                ))
        elif self.f_network_type == "convnet":
            f_dense_hidden_dims = self.kwargs.get('f_dense_hidden_dims', 128)
            f_channels = self.kwargs.get('f_channels', 1)
            f_kernel_size = self.kwargs.get('f_kernel_size', 7)
            f_stride = self.kwargs.get('f_stride', 3)
            for i in range(1,self.num_interventions+1):
                setattr(self, f"f_{i}", ConvNet(
                    self.window_size, self.window_size, f_channels, f_kernel_size, f_stride, 
                    dense_hidden_dim=f_dense_hidden_dims, batch_norm=f_batch_norm, 
                    p_dropout=f_dropout_ratio, activation=f_activation
                ))
        elif self.f_network_type == "dk_convnet":
            f_dense_hidden_dims = self.kwargs.get('f_dense_hidden_dims', 128)
            f_channels = self.kwargs.get('f_channels', 1)
            f_kernel_size = self.kwargs.get('f_kernel_size', 7)
            f_stride = self.kwargs.get('f_stride', 3)
            f_num_basis = self.kwargs.get('f_num_basis', 4)
            for i in range(1,self.num_interventions+1):
                setattr(self, f"f_{i}", DeepKrigingConvNet(
                    self.window_size, self.window_size, f_num_basis, f_channels, f_kernel_size, f_stride, 
                    dense_hidden_dim=f_dense_hidden_dims, batch_norm=f_batch_norm, p_dropout=f_dropout_ratio, 
                    activation=f_activation
                ))
        # elif self.f_network_type == "gcn":
        #     f_hidden_dims = self.kwargs.get('f_hidden_dims', 16)
        #     f_num_hidden_layers = self.kwargs.get('f_num_hidden_layers', 1)
        #     setattr(self, "f", GCN(self.num_interventions, f_num_hidden_layers, f_hidden_dims, 
        #                            activation=f_activation))
        elif self.f_network_type == "unet":
            f_depth = self.kwargs.get('f_depth', 2)
            f_padding = self.kwargs.get('f_padding', 1)
            for i in range(1,self.num_interventions+1):
                setattr(self, f"f_{i}", UNet(1, 1, depth=f_depth, padding=f_padding, batch_norm=f_batch_norm, 
                                             p_dropout=f_dropout_ratio))
        else:
            raise Exception(f"Invalid network type for f: {self.f_network_type}")
        # model for g(X)
        g_batch_norm = self.kwargs.get('g_batch_norm', False)
        g_activation = self.kwargs.get('g_activation', 'relu')
        g_dropout_ratio = self.kwargs.get('g_dropout_ratio', 0.0)
        if self.g_network_type == "mlp":
            g_hidden_dims = self.kwargs.get('g_hidden_dims', [128,64])
            self.g = MLP(self.confounder_dim, len(g_hidden_dims), g_hidden_dims, 
                         batch_norm=g_batch_norm, p_dropout=g_dropout_ratio, activation=g_activation)
        elif self.g_network_type == "convnet":
            g_dense_hidden_dims = self.kwargs.get('g_dense_hidden_dims', 128)
            g_channels = self.kwargs.get('g_channels', 1)
            g_kernel_size = self.kwargs.get('g_kernel_size', 7)
            self.g = ConvNet(
                self.window_size, self.window_size, g_channels, g_kernel_size, 
                dense_hidden_dim=g_dense_hidden_dims, batch_norm=g_batch_norm, 
                p_dropout=g_dropout_ratio, activation=g_activation
            )
        elif self.g_network_type == "dk_mlp":
            g_hidden_dims = self.kwargs.get('g_hidden_dims', [128,64])
            g_num_basis = self.kwargs.get('g_num_basis', 4)
            self.g = DeepKrigingMLP(self.confounder_dim, len(g_hidden_dims), g_hidden_dims, 
                                    g_num_basis, batch_norm=g_batch_norm, 
                                    p_dropout=g_dropout_ratio, activation=g_activation)
        elif self.g_network_type == "dk_convnet":
            g_dense_hidden_dims = self.kwargs.get('g_dense_hidden_dims', 128)
            g_channels = self.kwargs.get('g_channels', 1)
            g_kernel_size = self.kwargs.get('g_kernel_size', 7)
            g_num_basis = self.kwargs.get('g_num_basis', 4)
            self.g = DeepKrigingConvNet(
                self.window_size, self.window_size, g_num_basis, g_channels, g_kernel_size, 
                dense_hidden_dim=g_dense_hidden_dims, batch_norm=g_batch_norm, 
                p_dropout=g_dropout_ratio, activation=g_activation
            )
        else:
            raise Exception(f"Invalid network type for g: {self.g_network_type}")
        # model for U
        if self.unobserved_confounder:
            kernel_assignment = ['ImplicitNystromKernel']
            kernel_params = {
                'ImplicitNystromKernel': {
                    'kernel_func': KERNEL_FUNCS[self.kwargs.get('kernel_func', 'rbf')], 
                    'params': KERNEL_PARAMS[self.kwargs.get('kernel_func', 'rbf')], 
                    'vals': self.kwargs.get('kernel_param_vals', [1.,1.,0.5]), 
                    'trainable': [True] * len(KERNEL_PARAMS[self.kwargs.get('kernel_func', 'rbf')]), 
                    'alpha': self.kwargs.get('alpha', 1e-5),
                    'num_inducing_points': self.kwargs.get('num_inducing_points', 32),
                    'nys_space': self.kwargs.get('inducing_point_space', [[0.,1.]])
                }
            }
            self.gp_unobserved_confounder = ICK(kernel_assignment, kernel_params)
    
    def forward_direct(self, t: List[torch.Tensor]) -> List:
        """
        Calculate the resulting outcome directly from the interventions

        t: List[torch.Tensor], a list of tensors containing the intervention variables with shape 
            (batch_size, window_size) or (batch_size, window_size, window_size)
        """
        y_t = []
        for i in range(self.num_interventions):
            if len(t[i].shape) == 2:
                ti = t[i][:,self.window_size//2]
            else:
                ti = t[i][:,self.window_size//2,self.window_size//2]
            y_t.append(ti * getattr(self, f"beta_{i+1}"))
        return y_t
    
    def forward_residual(self, t: List[torch.Tensor], x: torch.Tensor, s: torch.Tensor = None, 
                         graph_features: torch.Tensor = None, edge_indices: torch.Tensor = None) -> Tuple:
        """
        Calculate the residuals from the neighboring interventions and the confounder variables

        t: List[torch.Tensor], a list of tensors containing the intervention variables with shape 
            (batch_size, window_size) or (batch_size, window_size, window_size)
        x: torch.Tensor, a tensor containing the confounder variable with shape (batch_size, confounder_dim) or
            (batch_size, window_size) or (batch_size, num_channels, window_size, window_size)
        s: torch.Tensor, a tensor containing the spatial information (e.g., coordinates or distance) 
            of the training data
        graph_features: torch.Tensor, a tensor containing the graph features for the GCN model
        edge_indices: torch.Tensor, a tensor containing the edge indices for the GCN model
        """
        assert self.window_size % 2 == 1, "The window size should be odd."
        if self.f_network_type == "gcn":
            assert graph_features is not None and edge_indices is not None, \
                "Graph features and edge indices are required for the GCN model."
        y_t_bar, y_x = [], []
        t_mask = torch.ones_like(t[0])
        # Masking the center of the intervention variables to be zero
        # TBD: or should we freeze the gradients of the center of the intervention variables?
        if len(t[0].shape) == 2:
            t_mask[:,self.window_size//2] = 0
        else:
            t_mask[:,self.window_size//2,self.window_size//2] = 0
        for i in range(self.num_interventions):
            ti_bar = (t[i] * t_mask).unsqueeze(1)
            if self.f_network_type == "dk_convnet":
                y_t_bar_i = getattr(self, f"f_{i+1}")(ti_bar, s).squeeze()
            # elif self.f_network_type == "gcn":
            #     y_t_bar_i = getattr(self, "f")(graph_features, edge_indices).squeeze()
            #     y_t_bar_i = y_t_bar_i[y_t_bar_i.shape[0]//2].unsqueeze(0)
            else:
                y_t_bar_i = getattr(self, f"f_{i+1}")(ti_bar).squeeze()
            y_t_bar.append(y_t_bar_i if len(y_t_bar_i.shape) else y_t_bar_i.unsqueeze(0))
        if self.g_network_type == "mlp":
            assert len(x.shape) == 2, "MLP only supports 1D input."
            y_x_i = self.g(x).squeeze()
        elif self.g_network_type == "convnet":
            y_x_i = self.g(x).squeeze()
        elif self.g_network_type == "dk_mlp":
            assert len(x.shape) == 2, "MLP only supports 1D input."
            y_x_i = self.g(x, s).squeeze()
        elif self.g_network_type == "dk_convnet":
            y_x_i = self.g(x, s).squeeze()
        y_x.append(y_x_i if len(y_x_i.shape) else y_x_i.unsqueeze(0))
        res = (y_t_bar, y_x)
        if self.unobserved_confounder:
            if len(s.shape) == 1:
                s = s.unsqueeze(1)
            res += ([self.gp_unobserved_confounder([s])],)
        return res
    
    def forward(self, t: List[torch.Tensor], x: torch.Tensor, s: torch.Tensor = None, 
                graph_features: torch.Tensor = None, edge_indices: torch.Tensor = None, 
                return_components: bool = False) -> torch.Tensor:
        y_t = self.forward_direct(t)
        if not self.unobserved_confounder:
            y_t_bar, y_x = self.forward_residual(t, x, s, graph_features, edge_indices)
            output = torch.sum(torch.stack(y_t + y_t_bar + y_x), dim=0)
        else:
            y_t_bar, y_x, u = self.forward_residual(t, x, s, graph_features, edge_indices)
            output = torch.sum(torch.stack(y_t + y_t_bar + y_x + u), dim=0)
        if not return_components:
            return output
        else:
            return (y_t,) + (y_t_bar,) + (y_x,) + (u,) if self.unobserved_confounder \
                else (y_t,) + (y_t_bar,) + (y_x,)
    
    def predict(self, t: List[torch.Tensor], x: torch.Tensor, s: torch.Tensor = None, 
                graph_features: torch.Tensor = None, edge_indices: torch.Tensor = None, 
                return_components: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(t, x, s, graph_features, edge_indices, return_components)
    
    def initialize_weights(self, method: str = 'kaiming') -> None:
        """
        Initialize weights of the model

        Arguments
        --------------
        method: str, the method to use for weight initialization, default to "kaiming"
        """
        assert method in {'kaiming','xavier'}, "Invalid method for weight initialization."
        for i in range(1,self.num_interventions+1):
            if self.f_network_type in {"mlp","convnet","dk_convnet","unet"}:
                getattr(self, f"f_{i}").initialize_weights(method=method)
            else:
                self.f.initialize_weights(method=method)
        self.g.initialize_weights(method=method)
    
    def freeze_direct_weights(self) -> None:
        """
        Freeze the weights for direct interventions
        """
        for i in range(1,self.num_interventions+1):
            getattr(self, f"beta_{i}").requires_grad = False
    
    def unfreeze_direct_weights(self) -> None:
        """
        Unfreeze the weights for direct interventions
        """
        for i in range(1,self.num_interventions+1):
            getattr(self, f"beta_{i}").requires_grad = True
    
    def freeze_residual_weights(self) -> None:
        """
        Freeze the weights for the residual interventions
        """
        if self.f_network_type == "gcn":
            for param in self.f.parameters():
                param.requires_grad = False
        else:
            for i in range(1,self.num_interventions+1):
                for param in getattr(self, f"f_{i}").parameters():
                    param.requires_grad = False
        for param in self.g.parameters():
            param.requires_grad = False
        if self.unobserved_confounder:
            for param in self.gp_unobserved_confounder.parameters():
                param.requires_grad = False
    
    def unfreeze_residual_weights(self) -> None:
        """
        Unfreeze the weights for the residual interventions
        """
        if self.f_network_type == "gcn":
            for param in self.f.parameters():
                param.requires_grad = True
        else:
            for i in range(1,self.num_interventions+1):
                for param in getattr(self, f"f_{i}").parameters():
                    param.requires_grad = True
        for param in self.g.parameters():
            param.requires_grad = True
        if self.unobserved_confounder:
            for param in self.gp_unobserved_confounder.parameters():
                param.requires_grad = True

        
# ########################################################################################
# MIT License

# Copyright (c) 2023 Ziyang Jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:

# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
# OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################################