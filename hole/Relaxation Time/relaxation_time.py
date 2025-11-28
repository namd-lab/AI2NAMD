import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import sys
from scipy.optimize import curve_fit
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch

sys.setrecursionlimit(100000)

# -------------------------- Global Settings: Ensure Normal Plot Display --------------------------
def set_plot_style():
    """Set plot style for better visualization (remove Chinese font settings)"""
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
    plt.rcParams['font.size'] = 10  # Set default font size
    plt.rcParams['figure.dpi'] = 100  # Default DPI for plots


# -------------------------- 1. Gaussian Curve Fitting Tool (Core: Get τ_D) --------------------------
def gaussian_decay(t, A, tau_D):
    """Gaussian decay function model: y = A * exp(-0.5*(t/tau_D)^2)"""
    return A * np.exp(-0.5 * (t / tau_D)**2)

def fit_gaussian_decay(x_data, y_data):
    """Fit Gaussian decay function and return τ_D (focus on lifetime parameter)"""
    try:
        initial_guess = [np.max(y_data), len(x_data) / 4]  # A=curve max, τ=1/4 time length
        params, params_covariance = curve_fit(
            gaussian_decay, 
            x_data, 
            y_data, 
            p0=initial_guess,
            maxfev=10000,
            bounds=([0, 1e-3], [np.inf, np.inf])  # τ_D must be positive
        )
        A, tau_D = params
        y_fit = gaussian_decay(x_data, A, tau_D)
        return {
            'A': A,
            'tau_D': tau_D,
            'y_fit': y_fit,
            'success': True,
            'covariance': params_covariance
        }
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        return {
            'A': np.nan,
            'tau_D': np.nan,
            'y_fit': np.zeros_like(x_data),
            'success': False,
            'covariance': None
        }


# -------------------------- 2. Graph Structure Dataset Class (Target: τ_D) --------------------------
class GraphEnergyDataset(Dataset):
    """Graph-structured dataset (Target: Predict Gaussian-fitted lifetime τ_D, keep raw curves for visualization)"""
    def __init__(self, config, ham_count_per_group=400):
        self.config = config
        self.ham_count_per_group = ham_count_per_group
        
        # 1. Load Energy data and fit τ_D (Core modification: Target from curve to τ_D)
        self.tau_data, self.size_to_meta = self._load_energy_and_fit_tau()
        self.valid_sizes = sorted(self.size_to_meta.keys())  # Only include sizes with successful fitting
        self.max_size = max(self.valid_sizes) if self.valid_sizes else 0
        self.energy_curve_len = self.size_to_meta[self.valid_sizes[0]]["curve_length"] if self.valid_sizes else 0
        
        # 2. Output dimension: τ_D is a single value, so output_dim=1
        self.output_dim = 1
        
        # 3. Preload Hamiltonians and build graph structures
        self.shared_full_hams = self._load_shared_full_hamiltonians()
        self.graph_groups = self._build_graph_groups_by_size()
        
        # 4. Validate data consistency
        self._validate_data()
        print(f"✅ Graph dataset initialized successfully:")
        print(f"  - Max Size: {self.max_size}×{self.max_size} (Graph nodes: {self.max_size})")
        print(f"  - Output dimension (τ_D as single value): {self.output_dim}")
        print(f"  - Number of valid sizes: {len(self.valid_sizes)} (Filtered only failed fittings)")

    def _load_energy_and_fit_tau(self):
        """Load Energy curves and fit τ_D as model target"""
        tau_data = {}  # Store true τ_D for each size
        size_to_meta = {}  # Store metadata for each size (curve, time steps, fitting results, etc.)
        # Core modification 1: en_num range adapted to 161×161 large matrix (76~132 corresponds to size=86~30, ensure size validity)
        start_en_num = 1  # corresponds to size=161-76+1=86
        end_en_num = 120   # corresponds to size=161-132+1=30
        
        for en_num in range(start_en_num, end_en_num + 1):
            # Core modification 2: Submatrix size calculation logic (size=161 - en_num + 1, 161 is fixed size of large matrix)
            size = 152 - en_num + 1
            if size <= 0:
                print(f"⚠️ Skip size={size}×{size} (invalid size, en_num={en_num})")
                continue
            
            en_filename = f"jj-en{en_num}.dat"
            en_filepath = os.path.join(self.config["res_target_energy"], en_filename)
            
            try:
                # Load raw energy data
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                energy_curve = en_raw[:, 1]
                
                # Fit τ_D (Core step)
                fit_result = fit_gaussian_decay(time_steps, energy_curve)
                if not fit_result["success"] or np.isnan(fit_result["tau_D"]) or fit_result["tau_D"] <= 0:
                    print(f"⚠️ Skip size={size}×{size} (τ_D fitting failed or invalid)")
                    continue
                
                # Save τ_D and metadata
                tau_data[size] = fit_result["tau_D"]
                size_to_meta[size] = {
                    "energy_num": en_num,
                    "filename": en_filename,
                    "time_steps": time_steps,
                    "energy_curve": energy_curve,
                    "curve_length": len(energy_curve),
                    "tau_D": fit_result["tau_D"],
                    "fit_A": fit_result["A"],
                    "fit_y": fit_result["y_fit"],
                    "fit_success": True
                }
                print(f"✅ Loaded successfully: size={size}×{size} (Energy_num={en_num}, τ_D={fit_result['tau_D']:.2f} fs)")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath}")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load {en_filename}: {str(e)} (Skipped)")
                continue
        
        # Check valid data
        if not tau_data:
            raise ValueError("All Energy data τ_D fitting failed, no valid training data! Check data or fitting parameters")
        
        return tau_data, size_to_meta

    def _load_shared_full_hamiltonians(self):
        """Load independent real/imaginary part Hamiltonian matrices"""
        shared_hams = []
        filename_template = "0_Ham_{idx}_{part}"
        # Core modification 3: Hamiltonian size adapted to 161×161 large matrix
        hamiltonian_size = 152  # Must match actual Hamiltonian size (161×161)
        
        for idx in range(self.ham_count_per_group):
            re_path = os.path.join(self.config["res"], filename_template.format(idx=idx, part="re"))
            im_path = os.path.join(self.config["res"], filename_template.format(idx=idx, part="im"))
            
            try:
                # Load and reshape to matrix
                re_data = np.loadtxt(re_path).astype(np.float32)
                im_data = np.loadtxt(im_path).astype(np.float32)
                re_mat = re_data.reshape(hamiltonian_size, hamiltonian_size)
                im_mat = im_data.reshape(hamiltonian_size, hamiltonian_size)
                
                # Validate dimensions
                if re_mat.shape != im_mat.shape:
                    raise ValueError(f"Hamiltonian {idx}: Real part shape({re_mat.shape}) != Imaginary part shape({im_mat.shape})")
                if hamiltonian_size < self.max_size:
                    raise ValueError(f"Hamiltonian size({hamiltonian_size}) < Max required size({self.max_size})")
                
                shared_hams.append( (re_mat, im_mat) )
            
            except FileNotFoundError:
                raise FileNotFoundError(f"Hamiltonian file missing: {re_path} or {im_path}")
            except ValueError as e:
                raise ValueError(f"Hamiltonian {idx} processing failed: {str(e)}")
        
        return shared_hams

    def _build_graph_from_hamiltonian(self, re_mat, im_mat):
        """Build graph structure from Hamiltonian (Node feature=row mean, Edge weight=interaction strength)"""
        size = re_mat.shape[0]
        
        # Node features: 2D (real part row mean + imaginary part row mean)
        node_feat = np.hstack([
            np.mean(re_mat, axis=1).reshape(-1, 1),  # Real part row mean
            np.mean(im_mat, axis=1).reshape(-1, 1)   # Imaginary part row mean
        ]).astype(np.float32)
        
        # Edge weights: √(real² + imaginary²) (interaction strength)
        adj_matrix = np.sqrt( re_mat**2 + im_mat**2 ).astype(np.float32)
        
        return (
            torch.tensor(node_feat, dtype=torch.float32),
            torch.tensor(adj_matrix, dtype=torch.float32)
        )

    def _build_graph_groups_by_size(self):
        """Crop Hamiltonians by size and build graph structure groups"""
        graph_groups = {}
        for size in self.valid_sizes:
            # Core modification 4: Get en_num corresponding to current size, calculate bottom-right crop start index
            en_num = self.size_to_meta[size]["energy_num"]
            start_idx = en_num - 1  # 0-based start index (en_num=76 → start_idx=75, corresponds to bottom-right 86×86 submatrix of 161×161 matrix)
            
            # Validate crop validity
            if start_idx + size > 152:
                raise ValueError(f"Invalid crop for size={size}×{size} (en_num={en_num}): start_idx={start_idx}, start_idx+size={start_idx+size} > 152")
            
            graph_list = []
            for (full_re, full_im) in self.shared_full_hams:
                # Core modification 5: Crop bottom-right submatrix (replace original top-left crop logic)
                sub_re = full_re[start_idx:, start_idx:]
                sub_im = full_im[start_idx:, start_idx:]
                # Build graph
                node_feat, adj = self._build_graph_from_hamiltonian(sub_re, sub_im)
                graph_list.append( (node_feat, adj) )
            graph_groups[size] = graph_list
        return graph_groups

    def _validate_data(self):
        """Validate data consistency (τ_D validity, graph structure dimensions, etc.)"""
        # Validate Hamiltonian count
        if len(self.shared_full_hams) != self.ham_count_per_group:
            raise ValueError(f"Hamiltonian count error: Expected {self.ham_count_per_group}, Actual {len(self.shared_full_hams)}")
        
        # Validate graph structure and τ_D for each size
        for size in self.valid_sizes:
            # Validate τ_D validity
            if np.isnan(self.tau_data[size]) or self.tau_data[size] <= 0:
                raise ValueError(f"size={size}×{size} has invalid τ_D ({self.tau_data[size]})")
            
            # Validate graph structure dimensions
            for node_feat, adj in self.graph_groups[size]:
                if node_feat.shape[0] != size or adj.shape != (size, size):
                    raise ValueError(f"size={size}×{size} graph dimension error (Nodes={node_feat.shape[0]}, Adjacency={adj.shape})")

    def __len__(self):
        """Total dataset samples = Number of valid sizes × Hamiltonians per size"""
        return len(self.valid_sizes) * self.ham_count_per_group

    def __getitem__(self, idx):
        """Return: Graph structure + True τ_D + Metadata (keep curves for visualization)"""
        group_idx = idx // self.ham_count_per_group
        ham_idx = idx % self.ham_count_per_group
        size = self.valid_sizes[group_idx]
        
        # Graph structure
        node_feat, adj = self.graph_groups[size][ham_idx]
        # True τ_D (target value, shape [1])
        true_tau = torch.tensor([self.tau_data[size]], dtype=torch.float32)
        # Metadata (for later visualization and saving)
        meta = self.size_to_meta[size]
        
        return (node_feat, adj, true_tau, size, meta["energy_num"], 
                meta["time_steps"], meta["energy_curve"], meta["fit_y"])


# -------------------------- 3. Graph Data Batching Tool (Adapt to τ_D Target) --------------------------
def collate_graphs(batch):
    """Batch process graph data, ensure edge weight dimension is correct ([num_edges, 1])"""
    # Unpack batch samples (correspond to __getitem__ return values)
    node_feats, adjs, true_taus, sizes, energy_nums, time_steps, energy_curves, fit_ys = zip(*batch)
    
    # Build PyG Data list
    data_list = []
    for nf, adj, tau in zip(node_feats, adjs, true_taus):
        # Extract non-zero edges (sparse graph representation)
        edges = adj.nonzero().t()  # [2, num_edges]
        # Edge weights: Extract from adjacency matrix and reshape to [num_edges, 1]
        edge_weights = adj[edges[0], edges[1]].view(-1, 1)
        # Build Data object
        data = Data(
            x=nf,          # Node features [size, 2]
            edge_index=edges,  # Edge indices [2, num_edges]
            edge_weight=edge_weights,  # Edge weights [num_edges, 1]
            y=tau          # Target τ_D [1]
        )
        data_list.append(data)
    
    # Batch concatenation
    batch_data = Batch.from_data_list(data_list)
    # Stack target τ_D ([batch_size, 1])
    stacked_taus = torch.stack(true_taus)
    
    return (batch_data, stacked_taus, sizes, energy_nums, 
            time_steps, energy_curves, fit_ys)


# -------------------------- 4. TransformerGNN Model (Output: τ_D) --------------------------
class EnergyPredTransformerGNN(nn.Module):
    """Transformer-based Graph Neural Network (Target: Predict τ_D, Output dimension=1)"""
    def __init__(self, input_dim, hidden_dim, config, heads=4):
        super().__init__()
        self.config = config
        self.heads = heads  # Number of multi-head attention
        self.single_head_dim = hidden_dim  # Output dimension per attention head
        
        # TransformerConv layers (Explicit edge feature dimension=1)
        self.transformer1 = TransformerConv(
            in_channels=input_dim,
            out_channels=self.single_head_dim,
            heads=self.heads,
            edge_dim=1  # Edge weight is 1D feature
        )
        self.transformer2 = TransformerConv(
            in_channels=self.heads * self.single_head_dim,  # Input = multi-head concatenation dimension
            out_channels=self.single_head_dim,
            heads=self.heads,
            edge_dim=1
        )
        self.transformer3 = TransformerConv(
            in_channels=self.heads * self.single_head_dim,
            out_channels=self.single_head_dim,
            heads=self.heads,
            edge_dim=1
        )
        
        # Batch normalization (Match multi-head output dimension)
        self.bn1 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        self.bn2 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        self.bn3 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        
        # Fully connected layers (Output dimension=1, predict τ_D)
        self.fc = nn.Sequential(
            nn.Linear(self.heads * self.single_head_dim, self.heads * self.single_head_dim),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(self.heads * self.single_head_dim, 1)  # Core modification: Output τ_D (1D)
        )

    def forward(self, batch_data):
        """Forward propagation: Graph data → Transformer attention → Global pooling → τ_D prediction"""
        x, edge_index, edge_weight, batch = (
            batch_data.x, 
            batch_data.edge_index, 
            batch_data.edge_weight,
            batch_data.batch
        )
        
        # First Transformer layer
        x = self.transformer1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second Transformer layer
        x = self.transformer2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third Transformer layer
        x = self.transformer3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling (Compress graph-level features to 1D)
        x = global_mean_pool(x, batch)  # [batch_size, heads*single_head_dim]
        
        # Predict τ_D
        return self.fc(x)  # [batch_size, 1]


# -------------------------- 5. Utility Functions (Add Average CSV + Combined Plot) --------------------------
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R² score for numerical prediction (Adapt to τ_D prediction)"""
    # Convert to numpy array and flatten
    y_true = y_true.detach().cpu().numpy().squeeze()  # [batch_size]
    y_pred = y_pred.detach().cpu().numpy().squeeze()  # [batch_size]
    
    # Calculate R² (Avoid division by zero)
    mean_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_true) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total < 1e-8:
        return 0.0 if ss_residual < 1e-8 else -np.inf
    
    return 1 - (ss_residual / ss_total)

def save_training_history_to_csv(history, save_path):
    """Save training history (loss, R²) to CSV file"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    history_df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": history["train_loss"],
        "test_loss": history["test_loss"],
        "train_r2": history["train_r2"],
        "test_r2": history["test_r2"]
    })
    history_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"📊 Training history CSV saved to: {save_path}")
    return save_path

def save_tau_sample_results_to_csv(true_taus, pred_taus, sizes, energy_nums, 
                                  dataset_type, save_dir, sample_indices):
    """Save true/predicted τ_D of each Hamiltonian sample to CSV (Sample-level CSV)"""
    # Calculate error metrics
    true_np = np.array(true_taus)
    pred_np = np.array(pred_taus)
    abs_errors = np.abs(true_np - pred_np)
    rel_errors = (abs_errors / true_np) * 100  # Relative error (%)
    
    # Build CSV data
    csv_data = {
        "dataset_type": [dataset_type.upper()] * len(true_taus),
        "sample_index": sample_indices,
        "size": [f"{s}×{s}" for s in sizes],
        "energy_num": energy_nums,
        "true_tau_fs": true_np.round(4),
        "pred_tau_fs": pred_np.round(4),
        "abs_error_fs": abs_errors.round(4),
        "relative_error_pct": rel_errors.round(4)
    }
    
    # Save path
    csv_dir = os.path.join(save_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{dataset_type}_tau_sample_results_{timestamp}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    # Save CSV
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📋 Sample-level τ_D prediction results CSV saved to: {csv_path}")
    return csv_filename, csv_path

def save_tau_mean_results_to_csv(size_list, mean_true_list, mean_pred_list, sample_count_list,
                                 dataset_type, save_dir):
    """Save size-averaged true/predicted τ_D to CSV (Average-level CSV, raw data for scatter plot)"""
    # Calculate average errors
    mean_true_np = np.array(mean_true_list)
    mean_pred_np = np.array(mean_pred_list)
    abs_errors = np.abs(mean_true_np - mean_pred_np)
    rel_errors = (abs_errors / mean_true_np) * 100  # Relative error (%)
    
    # Calculate overall metrics for this dataset
    overall_mean_true = np.mean(mean_true_np)
    overall_ss_total = np.sum((mean_true_np - overall_mean_true)**2)
    overall_ss_res = np.sum((mean_true_np - mean_pred_np)**2)
    overall_r2 = 1 - (overall_ss_res / overall_ss_total) if overall_ss_total > 1e-8 else 0.0
    overall_mae = np.mean(abs_errors)
    overall_rmse = np.sqrt(np.mean(abs_errors**2))
    
    # Build CSV data
    csv_data = {
        "dataset_type": [dataset_type.upper()] * len(size_list),
        "size": [f"{s}×{s}" for s in size_list],
        "mean_true_tau_fs": mean_true_np.round(4),
        "mean_pred_tau_fs": mean_pred_np.round(4),
        "sample_count_per_size": sample_count_list,
        "mean_abs_error_fs": abs_errors.round(4),
        "mean_relative_error_pct": rel_errors.round(4),
        "overall_r2": [overall_r2.round(4)] * len(size_list),
        "overall_mae_fs": [overall_mae.round(4)] * len(size_list),
        "overall_rmse_fs": [overall_rmse.round(4)] * len(size_list)
    }
    
    # Save path
    csv_dir = os.path.join(save_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{dataset_type}_tau_mean_results_{timestamp}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    # Save CSV
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📋 Average-level τ_D prediction results CSV (scatter plot data) saved to: {csv_path}")
    return csv_filename, csv_path

def plot_training_history(history, save_path):
    """Plot training history (Adapt to English, keep only loss and R² curves)"""
    set_plot_style()  # Set plot style
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss curves
    plt.figure(figsize=(12, 5),dpi=600)
    plt.plot(history["train_loss"], label="Training Loss", color="#2E86AB", linewidth=2)
    plt.plot(history["test_loss"], label="Test Loss", color="#A23B72", linewidth=2)

    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel(" MSE Loss", fontsize=10)
    plt.legend(fontsize=10)
    
    plt.savefig(save_path, dpi=600)
    print(f"📈 Training history plot saved to: {save_path}")
    plt.close()

def plot_single_tau_visualization(size_list, mean_true_list, mean_pred_list, 
                                 dataset_type, save_dir, r2, mae, rmse):
    """Plot τ_D prediction scatter plot for single dataset (train/test) (Remove energy curve example, adapt to English)"""
    set_plot_style()  # Set plot style
    plt.figure(figsize=(10, 8))  # Adjust figure size, keep only scatter plot
    ax1 = plt.gca()
    
    # Plot scatter plot (size-averaged data)
    scatter = ax1.scatter(mean_true_list, mean_pred_list, c='#A23B72', s=100, alpha=0.8, label=f'{dataset_type} Set')
    
    # Plot ideal prediction line (y=x)
    min_tau = min(min(mean_true_list), min(mean_pred_list)) * 0.9
    max_tau = max(max(mean_true_list), max(mean_pred_list)) * 1.1
    ax1.plot([min_tau, max_tau], [min_tau, max_tau], 'k--', linewidth=2)
    
    # Annotate size for each point
    for i, s in enumerate(size_list):
        ax1.annotate(f"{s}×{s}", (mean_true_list[i], mean_pred_list[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Title and axis labels (English)
    ax1.set_title(f'{dataset_type} Set τ_D Prediction Results\nR²={r2:.4f} | MAE={mae:.2f}fs | RMSE={rmse:.2f}fs',
                 fontsize=14, pad=20)
    ax1.set_xlabel('True τ_D (fs)', fontsize=12)
    ax1.set_ylabel('Predicted τ_D (fs)', fontsize=12)
    ax1.legend(fontsize=11)
    
    # Save plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{dataset_type}_tau_scatter_{timestamp}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    print(f"📊 {dataset_type} set τ_D scatter plot saved to: {plot_path}")
    plt.close()
    return plot_filename, plot_path

def plot_combined_tau_visualization(train_data, test_data, save_dir):
    """Plot combined τ_D prediction scatter plot for train+test sets (Adapt to English, comparative display)"""
    set_plot_style()  # Set plot style
    plt.figure(figsize=(12, 9))  # Larger figure size for better comparison
    ax = plt.gca()
    
    # Extract averaged data for train and test sets
    # train_data structure: (size_list, mean_true_list, mean_pred_list, r2, mae, rmse)
    train_size, train_true, train_pred, train_r2, train_mae, train_rmse = train_data
    test_size, test_true, test_pred, test_r2, test_mae, test_rmse = test_data
    
    # Plot training set scatter
    ax.scatter(train_true, train_pred, c='#2E86AB', s=120, alpha=0.8, label=f'Training Set (R²={train_r2:.4f})', marker='o')
    # Plot test set scatter
    ax.scatter(test_true, test_pred, c='#E74C3C', s=120, alpha=0.8, label=f'Test Set (R²={test_r2:.4f})', marker='s')
    
    # Plot ideal prediction line (y=x)
    all_true = train_true + test_true
    min_tau = min(all_true) * 0.9
    max_tau = max(all_true) * 1.1
    ax.plot([min_tau, max_tau], [min_tau, max_tau], 'k--', linewidth=2.5)
    plt.tick_params(axis='both', labelsize=20)  # Adjust X and Y axes simultaneously
    
    # Annotate size for each point (train + test sets)
    for i, s in enumerate(train_size):
        ax.annotate(f"{s}×{s}", (train_true[i], train_pred[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, color='#2E86AB')
    for i, s in enumerate(test_size):
        ax.annotate(f"{s}×{s}", (test_true[i], test_pred[i]), 
                    xytext=(5, -15), textcoords='offset points', fontsize=10, color='#E74C3C')
    
    # Title and axis labels (English, include metrics for both sets)
    ax.set_title(f'Combined τ_D Prediction Results\n'
                 f'Training Set: MAE={train_mae:.2f}fs | RMSE={train_rmse:.2f}fs | '
                 f'Test Set: MAE={test_mae:.2f}fs | RMSE={test_rmse:.2f}fs',
                 fontsize=20, pad=20)
    ax.set_xlabel('True τ_D (fs)', fontsize=20)
    ax.set_ylabel('Predicted τ_D (fs)', fontsize=20)
    ax.legend(fontsize=20)
    
    # Save plot (save to prediction root directory for easy access)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"combined_tau_scatter_{timestamp}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    print(f"📊 Combined train+test set τ_D scatter plot saved to: {plot_path}")
    plt.close()
    return plot_filename, plot_path

def predict_all_taus(model, dataset, dataset_type, device, save_dir):
    """Batch predict τ_D and save [Sample-level + Average-level CSV] + plot [Single dataset scatter plot]"""
    model.eval()
    torch.manual_seed(42)  # Fix random seed for reproducibility
    
    # 1. Collect raw prediction data for each sample
    all_true_taus = []
    all_pred_taus = []
    all_sizes = []
    all_energy_nums = []
    all_sample_indices = []
    
    print(f"\n=== {dataset_type.upper()} Set τ_D Prediction Started ===")
    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            # Load sample
            node_feat, adj, true_tau, size, en_num, _, _, _ = dataset[sample_idx]  # No need for curve data
            true_tau_val = true_tau.item()
            
            # Build graph data
            edges = adj.nonzero().t()
            edge_weights = adj[edges[0], edges[1]].view(-1, 1)
            data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
            batch_data = Batch.from_data_list([data])
            
            # Predict τ_D
            pred = model(batch_data)
            pred_tau_val = pred.cpu().item()
            
            # Collect sample-level data
            all_true_taus.append(true_tau_val)
            all_pred_taus.append(pred_tau_val)
            all_sizes.append(size)
            all_energy_nums.append(en_num)
            all_sample_indices.append(sample_idx)
            
            # Progress prompt
            if (sample_idx + 1) % 100 == 0 or (sample_idx + 1) == len(dataset):
                print(f"Progress: {sample_idx + 1}/{len(dataset)} samples processed")
    
    # 2. Calculate size-averaged data (for scatter plot and average-level CSV)
    size_groups = {}
    for t, p, s in zip(all_true_taus, all_pred_taus, all_sizes):
        if s not in size_groups:
            size_groups[s] = {"true": [], "pred": []}
        size_groups[s]["true"].append(t)
        size_groups[s]["pred"].append(p)
    
    # Organize averaged data
    size_list = sorted(size_groups.keys())
    mean_true_list = [np.mean(size_groups[s]["true"]) for s in size_list]
    mean_pred_list = [np.mean(size_groups[s]["pred"]) for s in size_list]
    sample_count_list = [len(size_groups[s]["true"]) for s in size_list]  # Number of samples per size
    
    # 3. Calculate overall metrics for this dataset
    mean_true_np = np.array(mean_true_list)
    mean_pred_np = np.array(mean_pred_list)
    overall_mean_true = np.mean(mean_true_np)
    overall_ss_total = np.sum((mean_true_np - overall_mean_true)**2)
    overall_ss_res = np.sum((mean_true_np - mean_pred_np)**2)
    overall_r2 = 1 - (overall_ss_res / overall_ss_total) if overall_ss_total > 1e-8 else 0.0
    overall_mae = np.mean(np.abs(mean_true_np - mean_pred_np))
    overall_rmse = np.sqrt(np.mean((mean_true_np - mean_pred_np)**2))
    
    # 4. Save sample-level CSV (prediction for each Hamiltonian)
    sample_csv_name, sample_csv_path = save_tau_sample_results_to_csv(
        true_taus=all_true_taus,
        pred_taus=all_pred_taus,
        sizes=all_sizes,
        energy_nums=all_energy_nums,
        dataset_type=dataset_type,
        save_dir=save_dir,
        sample_indices=all_sample_indices
    )
    
    # 5. Save average-level CSV (size-averaged, raw data for scatter plot)
    mean_csv_name, mean_csv_path = save_tau_mean_results_to_csv(
        size_list=size_list,
        mean_true_list=mean_true_list,
        mean_pred_list=mean_pred_list,
        sample_count_list=sample_count_list,
        dataset_type=dataset_type,
        save_dir=save_dir
    )
    
    # 6. Plot single dataset scatter plot (Remove energy curve example)
    plot_name, plot_path = plot_single_tau_visualization(
        size_list=size_list,
        mean_true_list=mean_true_list,
        mean_pred_list=mean_pred_list,
        dataset_type=dataset_type,
        save_dir=save_dir,
        r2=overall_r2,
        mae=overall_mae,
        rmse=overall_rmse
    )
    
    # Output statistics
    print(f"\n=== {dataset_type.upper()} Set Prediction Statistics ===")
    print(f"Total samples: {len(all_true_taus)}")
    print(f"Number of size groups: {len(size_list)}")
    print(f"Overall R²: {overall_r2:.4f}")
    print(f"Mean Absolute Error (MAE): {overall_mae:.2f} fs")
    print(f"Root Mean Square Error (RMSE): {overall_rmse:.2f} fs")
    
    # Return averaged data for later combined plot
    mean_data = (size_list, mean_true_list, mean_pred_list, overall_r2, overall_mae, overall_rmse)
    return {
        "sample_csv_path": sample_csv_path,
        "mean_csv_path": mean_csv_path,
        "plot_path": plot_path,
        "mean_data": mean_data,  # For combined plot
        "overall_r2": overall_r2,
        "mae": overall_mae
    }


# -------------------------- 6. Main Workflow (Dataset Generation + Model Training + Prediction + Combined Plot) --------------------------
def regenerate_graph_dataset(train_energy_count=7):
    """Generate graph-structured dataset for τ_D prediction (Run once only)
    Data division logic consistent with before: divide by energy index, training set contains minimum and maximum energy indices, customize number of training set energy indices
    """
    config = {
        "res": "./res",  # Hamiltonian file path
        "res_target_energy": "./res_target_energy",  # Energy data file path
        "ham_count_per_group": 1000,  # Number of Hamiltonian samples per size
        "save_dir": "./10processed_graph_tau_data",  # Dataset save path
        "random_state": 1314
    }
    
    # Create directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["res"], exist_ok=True)
    os.makedirs(config["res_target_energy"], exist_ok=True)
    
    # Fix random seed
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    
    # Initialize dataset
    full_dataset = GraphEnergyDataset(config=config, ham_count_per_group=config["ham_count_per_group"])
    
    # Divide training and test sets by energy index (keep original logic)
    # 1. Collect all valid energy indices
    all_energy_nums = []
    for size in full_dataset.valid_sizes:
        en_num = full_dataset.size_to_meta[size]["energy_num"]
        all_energy_nums.append(en_num)
    all_energy_nums = sorted(list(set(all_energy_nums)))  # Deduplicate and sort
    
    if len(all_energy_nums) < train_energy_count:
        raise ValueError(f"Insufficient available energy indices! {len(all_energy_nums)} available, need {train_energy_count} for training set")
    
    # 2. Ensure training set contains minimum and maximum energy indices
    min_en = all_energy_nums[0]  # Minimum energy index (must include)
    max_en = all_energy_nums[-1]  # Maximum energy index (must include)
    
    # Initialize training set indices, force include minimum and maximum
    train_energy_indices = [min_en, max_en]
    
    # Select supplementary training set from remaining indices
    remaining_ens = [en for en in all_energy_nums if en not in train_energy_indices]
    
    # Number needed to supplement
    need = train_energy_count - 2
    if need > 0:
        if len(remaining_ens) < need:
            raise ValueError(f"Insufficient remaining energy indices! Need {need} supplements, but only {len(remaining_ens)} found")
        
        # Uniformly sample remaining indices
        step = max(1, len(remaining_ens) // need)
        additional_ens = remaining_ens[::step][:need]
        
        # If sampling is insufficient, randomly supplement from remaining
        if len(additional_ens) < need:
            remaining_after = [en for en in remaining_ens if en not in additional_ens]
            additional_ens += random.sample(remaining_after, need - len(additional_ens))
        
        train_energy_indices.extend(additional_ens)
        train_energy_indices = sorted(list(set(train_energy_indices)))  # Deduplicate and sort
    
    # 3. Test set is remaining energy indices (excluding training set indices)
    test_energy_indices = [en for en in all_energy_nums if en not in train_energy_indices]
    
    print(f"\n=== Dataset Division ===")
    print(f"All energy indices: {all_energy_nums}")
    print(f"Minimum energy index: {min_en}, Maximum energy index: {max_en}")
    print(f"Training set energy indices({len(train_energy_indices)}): {sorted(train_energy_indices)}")
    print(f"Test set energy indices({len(test_energy_indices)}): {sorted(test_energy_indices)}")
    
    # 4. Build global indices for training and test sets
    train_indices = []
    test_indices = []
    current_idx = 0
    
    for size in full_dataset.valid_sizes:
        en_num = full_dataset.size_to_meta[size]["energy_num"]
        count = full_dataset.ham_count_per_group  # Samples per size = number of Hamiltonians
        
        # Determine if current energy index belongs to training or test set
        if en_num in train_energy_indices:
            train_indices.extend(range(current_idx, current_idx + count))
        else:
            test_indices.extend(range(current_idx, current_idx + count))
            
        current_idx += count
    
    # Create training and test sets
    train_data = [full_dataset[i] for i in train_indices]
    test_data = [full_dataset[i] for i in test_indices]
    
    # Save dataset and parameters
    try:
        # Save dataset
        with open(os.path.join(config["save_dir"], "train_graph_tau_dataset.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        with open(os.path.join(config["save_dir"], "test_graph_tau_dataset.pkl"), "wb") as f:
            pickle.dump(test_data, f)
        
        # Save dataset parameters
        data_params = {
            "output_dim": full_dataset.output_dim,
            "train_energy_indices": sorted(train_energy_indices),
            "test_energy_indices": sorted(test_energy_indices),
            "train_sizes": [full_dataset.valid_sizes[group_idx] for group_idx in range(len(full_dataset.valid_sizes)) 
                          if full_dataset.size_to_meta[full_dataset.valid_sizes[group_idx]]["energy_num"] in train_energy_indices],
            "test_sizes": [full_dataset.valid_sizes[group_idx] for group_idx in range(len(full_dataset.valid_sizes)) 
                         if full_dataset.size_to_meta[full_dataset.valid_sizes[group_idx]]["energy_num"] in test_energy_indices],
            "node_feat_dim": 2,  # Real part mean + Imaginary part mean
            "ham_count_per_group": config["ham_count_per_group"],
            "valid_sizes": full_dataset.valid_sizes,
            "max_size": full_dataset.max_size
        }
        with open(os.path.join(config["save_dir"], "graph_tau_data_params.pkl"), "wb") as f:
            pickle.dump(data_params, f)
        
        print(f"\n✅ Dataset saved successfully:")
        print(f"  - Training set: {len(train_data)} samples | Test set: {len(test_data)} samples")
        print(f"  - Node feature dimension: {data_params['node_feat_dim']} | Output dimension: {data_params['output_dim']}")
        print(f"  - Training set sizes: {sorted(data_params['train_sizes'])}")
        print(f"  - Test set sizes: {sorted(data_params['test_sizes'])}")
        print(f"  - Save path: {config['save_dir']}")
    
    except Exception as e:
        print(f"❌ Failed to save dataset: {str(e)}")
        raise


def main_training(train_energy_count=7):
    """Train TransformerGNN model for τ_D prediction + Generate dual CSV + Single/Combined plots"""
    config = {
        "processed_data_dir": "./10processed_graph_tau_data",
        "model_save_dir": "./10transformer_gnn_tau_models",
        "prediction_save_dir": "./transformer_gnn_tau_predictions",  # Prediction results root directory
        "batch_size": 32,
        "hidden_dim": 32,  # Single head dimension
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "epochs": 100,  # Adjustable training epochs
        "print_interval": 1,
        "random_state": 42,
        "transformer_heads": 6  # Number of Transformer multi-head attention
    }
    
    # Fix random seed
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    torch.manual_seed(config["random_state"])
    torch.cuda.manual_seed_all(config["random_state"])
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directories (Prediction root + train/test subdirectories)
    os.makedirs(config["model_save_dir"], exist_ok=True)
    os.makedirs(config["prediction_save_dir"], exist_ok=True)
    train_pred_dir = os.path.join(config["prediction_save_dir"], "train")  # Training set prediction results
    test_pred_dir = os.path.join(config["prediction_save_dir"], "test")    # Test set prediction results
    os.makedirs(train_pred_dir, exist_ok=True)
    os.makedirs(test_pred_dir, exist_ok=True)
    
    # Load dataset
    print("\n=== Loading Graph Dataset for τ_D Prediction ===")
    required_files = [
        "train_graph_tau_dataset.pkl", 
        "test_graph_tau_dataset.pkl", 
        "graph_tau_data_params.pkl"
    ]
    for file in required_files:
        file_path = os.path.join(config["processed_data_dir"], file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file missing: {file_path}\nPlease run regenerate_graph_dataset() first")
    
    with open(os.path.join(config["processed_data_dir"], "train_graph_tau_dataset.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(config["processed_data_dir"], "test_graph_tau_dataset.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(config["processed_data_dir"], "graph_tau_data_params.pkl"), "rb") as f:
        data_params = pickle.load(f)
    
    # Wrap dataset
    class WrappedTauDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = WrappedTauDataset(train_data)
    test_dataset = WrappedTauDataset(test_data)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    # Initialize TransformerGNN model
    print("\n=== Initializing EnergyPredTransformerGNN Model ===")
    model = EnergyPredTransformerGNN(
        input_dim=data_params["node_feat_dim"],
        hidden_dim=config["hidden_dim"],
        config=config,
        heads=config["transformer_heads"]
    ).to(device)
    print(model)
    
    # Training configuration
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )
    
    # Training process
    print("\n=== Starting TransformerGNN Training for τ_D Prediction ===")
    history = {
        "train_loss": [], "train_r2": [],
        "test_loss": [], "test_r2": []
    }
    best_test_loss = float("inf")
    best_model_path = os.path.join(config["model_save_dir"], "best_transformer_gnn_tau_model.pth")
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_total_r2 = 0.0
        
        for batch in train_loader:
            batch_data, targets, _, _, _, _, _ = batch
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_data)
            loss = F.mse_loss(preds, targets)  # Use MSE loss for τ_D prediction
            batch_r2 = r2_score(targets, preds)
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item() * preds.size(0)
            train_total_r2 += batch_r2 * preds.size(0)
        
        avg_train_loss = train_total_loss / len(train_dataset)
        avg_train_r2 = train_total_r2 / len(train_dataset)
        history["train_loss"].append(avg_train_loss)
        history["train_r2"].append(avg_train_r2)
        
        # Testing phase
        model.eval()
        test_total_loss = 0.0
        test_total_r2 = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_data, targets, _, _, _, _, _ = batch
                batch_data = batch_data.to(device)
                targets = targets.to(device)
                
                preds = model(batch_data)
                loss = F.mse_loss(preds, targets)
                batch_r2 = r2_score(targets, preds)
                
                test_total_loss += loss.item() * preds.size(0)
                test_total_r2 += batch_r2 * preds.size(0)
        
        avg_test_loss = test_total_loss / len(test_dataset)
        avg_test_r2 = test_total_r2 / len(test_dataset)
        history["test_loss"].append(avg_test_loss)
        history["test_r2"].append(avg_test_r2)
        
        scheduler.step(avg_test_loss)
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"📌 Epoch {epoch+1}: Saved best TransformerGNN model (Test loss: {best_test_loss:.6f})")
        
        if (epoch + 1) % config["print_interval"] == 0:
            print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
            print(f"  Training Set: Loss={avg_train_loss:.6f}, R²={avg_train_r2:.4f}")
            print(f"  Test Set: Loss={avg_test_loss:.6f}, R²={avg_test_r2:.4f}")
    
    # Save training history to CSV
    history_csv_path = os.path.join(config["model_save_dir"], "transformer_gnn_tau_training_history.csv")
    save_training_history_to_csv(history, history_csv_path)
    
    # Plot training history (Adapt to English)
    print("\n=== Plotting Training History ===")
    history_plot_path = os.path.join(config["model_save_dir"], "transformer_gnn_tau_training_history.png")
    plot_training_history(history, history_plot_path)
    
    # Batch predict τ_D (Generate dual CSV + single dataset scatter plot)
    print("\n=== Batch Predicting τ_D ===")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Training set prediction (Return averaged data for combined plot)
    print("\n--- Training Set Prediction ---")
    train_results = predict_all_taus(
        model=model,
        dataset=train_dataset,
        dataset_type="Train",
        device=device,
        save_dir=train_pred_dir
    )
    train_mean_data = train_results["mean_data"]  # (size, true, pred, r2, mae, rmse)
    
    # Test set prediction (Return averaged data for combined plot)
    print("\n--- Test Set Prediction ---")
    test_results = predict_all_taus(
        model=model,
        dataset=test_dataset,
        dataset_type="Test",
        device=device,
        save_dir=test_pred_dir
    )
    test_mean_data = test_results["mean_data"]
    
    # Plot combined train+test set scatter plot (Save to prediction root directory)
    print("\n--- Plotting Combined Train+Test Set Scatter Plot ---")
    combined_plot_name, combined_plot_path = plot_combined_tau_visualization(
        train_data=train_mean_data,
        test_data=test_mean_data,
        save_dir=config["prediction_save_dir"]  # Save to root directory for easy comparison
    )
    
    print(f"\n🎉 All Tasks Completed!")
    print(f"1. Best TransformerGNN Model: {best_model_path}")
    print(f"2. Training History CSV: {history_csv_path}")
    print(f"3. Training History Plot: {history_plot_path}")
    print(f"4. Training Set Results:")
    print(f"   - Sample-level CSV: {train_results['sample_csv_path']}")
    print(f"   - Average-level CSV (scatter plot data): {train_results['mean_csv_path']}")
    print(f"   - Single Set Scatter Plot: {train_results['plot_path']}")
    print(f"5. Test Set Results:")
    print(f"   - Sample-level CSV: {test_results['sample_csv_path']}")
    print(f"   - Average-level CSV (scatter plot data): {test_results['mean_csv_path']}")
    print(f"   - Single Set Scatter Plot: {test_results['plot_path']}")
    print(f"6. Combined Comparison Plot: {combined_plot_path}")


if __name__ == "__main__":
    # Customize number of training set energy indices (ensure ≥2, as must include minimum and maximum)
    custom_train_count = 12
    
    # Step 1: Generate dataset (Run once only)
    regenerate_graph_dataset(train_energy_count=custom_train_count)
    
    # Step 2: Train TransformerGNN model and predict τ_D
    main_training(train_energy_count=custom_train_count)