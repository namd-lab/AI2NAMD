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
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

sys.setrecursionlimit(100000)


# -------------------------- Configuration class (replaces dictionary config for enhanced type safety) --------------------------
@dataclass
class Config:
    res: str = "./res"
    res_target_energy: str = "./res_target_energy"
    save_dir: str = "./shi_mo_data"
    model_save_dir: str = "./shi_mo_models"
    prediction_save_dir: str = "./Si_shi_mo_predictions"
    features_save_dir: str = "./Si_shi_mo_features"
    
    # Data parameters
    ham_count_per_group: int = 1000
    train_energy_count: int = 7
    random_state: int = 42
    
    # Model parameters
    batch_size: int = 32
    hidden_dim: int = 32
    dropout: float = 0.2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    print_interval: int = 1
    transformer_heads: int = 6
    node_feat_dim: int = 4  # Extended node feature dimension


# -------------------------- 1. Improved Gaussian curve fitting tools (eliminates global variables) --------------------------
def gaussian_decay(t: np.ndarray, A: float, tau_D: float, t0: float = 0) -> np.ndarray:
    """Gaussian decay function model"""
    return A * np.exp(-0.5 * ((t - t0) / tau_D) ** 2)

def create_constrained_gaussian_decay(initial_value: float):
    """Create constrained Gaussian decay function with initial value constraint (using closure to eliminate global variables)"""
    def constrained_func(t: np.ndarray, tau_D: float) -> np.ndarray:
        return initial_value * np.exp(-0.5 * (t / tau_D) ** 2)
    return constrained_func

def fit_gaussian_decay(
    x_data: np.ndarray, 
    y_data: np.ndarray, 
    enforce_initial: bool = True
) -> Dict[str, Union[float, np.ndarray, bool]]:
    """Fit Gaussian decay function (improved version without global variables)"""
    if len(y_data) == 0:
        return {
            'A': np.nan, 'tau_D': np.nan, 'y_fit': np.zeros_like(x_data),
            'success': False, 'initial_value': np.nan, 'fit_initial_value': np.nan,
            'initial_diff': np.nan
        }
    
    initial_value = y_data[0]
    
    try:
        if enforce_initial:
            constrained_func = create_constrained_gaussian_decay(initial_value)
            initial_guess = [len(x_data) / 4]
            params, params_covariance = curve_fit(
                constrained_func, x_data, y_data, p0=initial_guess, maxfev=10000,
                bounds=([1e-3], [len(x_data)])  # Add parameter boundary constraints
            )
            tau_D = params[0]
            A = initial_value
            y_fit = constrained_func(x_data, tau_D)
        else:
            initial_guess = [initial_value, len(x_data) / 4]
            params, params_covariance = curve_fit(
                gaussian_decay, x_data, y_data, p0=initial_guess, maxfev=10000,
                bounds=([1e-3, 1e-3], [np.max(y_data)*2, len(x_data)])
            )
            A, tau_D = params
            y_fit = gaussian_decay(x_data, A, tau_D)
        
        fit_initial = y_fit[0]
        initial_diff = abs(fit_initial - initial_value)
        
        return {
            'A': A, 'tau_D': tau_D, 'y_fit': y_fit, 'success': True,
            'initial_value': initial_value, 'fit_initial_value': fit_initial,
            'initial_diff': initial_diff
        }
    except (RuntimeError, ValueError) as e:
        print(f"Fitting failed: {str(e)}")
        return {
            'A': np.nan, 'tau_D': np.nan, 'y_fit': np.zeros_like(x_data),
            'success': False, 'initial_value': initial_value, 
            'fit_initial_value': np.nan, 'initial_diff': np.nan
        }


# -------------------------- 2. Enhanced graph structure dataset class --------------------------
class GraphEnergyDataset(Dataset):
    def __init__(
        self, 
        config: Config, 
        ham_count_per_group: int = 400, 
        load_initial_from_csv: bool = False, 
        initial_csv_path: Optional[str] = None
    ):
        self.config = config
        self.ham_count_per_group = ham_count_per_group
        self.load_initial_from_csv = load_initial_from_csv
        self.initial_csv_path = initial_csv_path
        self.initial_save_csv = os.path.join(config.save_dir, "energy_initial_values.csv")
        
        # Load energy data
        if self.load_initial_from_csv:
            if not initial_csv_path or not os.path.exists(initial_csv_path):
                raise ValueError(f"Failed to load initial values: File does not exist → {initial_csv_path}")
            self.energy_data, self.size_to_energy, self.sample_to_initial = self._load_energy_from_initial_csv()
        else:
            self.energy_data, self.size_to_energy, self.sample_to_initial = self._load_energy_and_save_initial_csv()
        
        self.valid_sizes = sorted(self.size_to_energy.keys())
        self.max_size = max(self.valid_sizes) if self.valid_sizes else 0
        
        # Preload Hamiltonians and build enhanced graph structures
        self.shared_full_hams = self._load_shared_full_hamiltonians()
        self.graph_groups = self._build_enhanced_graph_groups_by_size()  # Enhanced graph construction
        
        # Validate curve lengths
        self._validate_curve_lengths()
        self.output_dim = self.energy_curve_len if hasattr(self, 'energy_curve_len') else 0
        
        # Data validation
        self._validate_data()
        
        print(f"✅ Dataset initialization completed:")
        print(f"  - Max Size: {self.max_size}×{self.max_size} (Graph nodes: {self.max_size})")
        print(f"  - Output dimension (Energy curve length): {self.output_dim}")
        print(f"  - Valid size list: {self.valid_sizes}")

    def _save_initial_to_csv(self, initial_data: List[Dict]) -> str:
        """Save initial energy values to CSV"""
        df = pd.DataFrame(initial_data)
        os.makedirs(os.path.dirname(self.initial_save_csv), exist_ok=True)
        df.to_csv(self.initial_save_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Initial energy saved successfully: {self.initial_save_csv} ({len(df)} samples total)")
        return self.initial_save_csv

    def _load_energy_and_save_initial_csv(self) -> Tuple[Dict, Dict, Dict]:
        """Load energy data and save initial values"""
        energy_data = {}
        size_to_energy = {}
        sample_to_initial = {}
        initial_data_list = []
        start_en_num, end_en_num = 30, 100
        global_sample_idx = 0
        
        for en_num in range(start_en_num, end_en_num + 1):
            size = en_num
            en_filename = f"jj-en{en_num}.dat"
            en_filepath = os.path.join(self.config.res_target_energy, en_filename)
            
            try:
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                base_energy_curve = en_raw[:, 1]
                base_initial_energy = base_energy_curve[0]
                
                size_curves = []
                for ham_idx in range(self.ham_count_per_group):
                    # Improved noise generation (fixed random seed)
                    rng = np.random.default_rng(self.config.random_state + ham_idx + en_num)
                    noise = rng.normal(0, 0.001, size=len(base_energy_curve))
                    single_curve = (base_energy_curve + noise).astype(np.float32)
                    size_curves.append(single_curve)
                    
                    sample_initial = single_curve[0]
                    sample_key = (size, ham_idx)
                    sample_to_initial[sample_key] = sample_initial
                    
                    initial_data_list.append({
                        "global_sample_idx": global_sample_idx,
                        "size": size,
                        "size_str": f"{size}×{size}",
                        "energy_num": en_num,
                        "ham_idx": ham_idx,
                        "initial_energy_eV": sample_initial,
                        "curve_first_point_eV": single_curve[0]
                    })
                    global_sample_idx += 1
                
                energy_data[size] = size_curves
                size_to_energy[size] = {
                    "energy_num": en_num,
                    "filename": en_filename,
                    "time_steps": time_steps,
                    "curve_count": len(size_curves)
                }
                print(f"✅ Processed: Energy_num={en_num} (size={size}×{size}, {len(size_curves)} samples)")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath}")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load {en_filename}: {str(e)} (skipped)")
                continue
        
        if initial_data_list:
            self._save_initial_to_csv(initial_data_list)
        
        return energy_data, size_to_energy, sample_to_initial

    def _load_energy_from_initial_csv(self) -> Tuple[Dict, Dict, Dict]:
        """Load initial energy values from CSV"""
        df = pd.read_csv(self.initial_csv_path)
        required_cols = ["size", "ham_idx", "energy_num", "initial_energy_eV", "curve_first_point_eV"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Initial energy CSV missing required columns! Need: {required_cols}")
        
        energy_data = {}
        size_to_energy = {}
        sample_to_initial = {}
        
        size_groups = df.groupby("size")
        for size, size_df in size_groups:
            size = int(size)
            energy_num = size_df["energy_num"].iloc[0]
            en_filename = f"jj-en{energy_num}.dat"
            en_filepath = os.path.join(self.config.res_target_energy, en_filename)
            
            try:
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                base_energy_curve = en_raw[:, 1]
                
                size_curves = []
                for _, row in size_df.iterrows():
                    ham_idx = int(row["ham_idx"])
                    target_initial = float(row["initial_energy_eV"])
                    
                    # Fixed seed for noise generation
                    rng = np.random.default_rng(self.config.random_state + ham_idx + energy_num)
                    noise = rng.normal(0, 0.001, size=len(base_energy_curve))
                    noise[0] = target_initial - base_energy_curve[0]
                    single_curve = (base_energy_curve + noise).astype(np.float32)
                    
                    size_curves.append(single_curve)
                    sample_to_initial[(size, ham_idx)] = target_initial
                
                energy_data[size] = size_curves
                size_to_energy[size] = {
                    "energy_num": energy_num,
                    "filename": en_filename,
                    "time_steps": time_steps,
                    "curve_count": len(size_curves)
                }
                print(f"✅ Loaded: size={size}×{size} ({len(size_curves)} samples)")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath} (skipped size={size})")
                continue
        
        return energy_data, size_to_energy, sample_to_initial

    def _load_shared_full_hamiltonians(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load Hamiltonian matrices (added exception handling)"""
        shared_hams = []
        filename_template = "0_Ham_{idx}_{part}"
        hamiltonian_size = 100
        
        for idx in range(self.ham_count_per_group):
            re_path = os.path.join(self.config.res, filename_template.format(idx=idx, part="re"))
            im_path = os.path.join(self.config.res, filename_template.format(idx=idx, part="im"))
            
            try:
                if not (os.path.exists(re_path) and os.path.exists(im_path)):
                    raise FileNotFoundError(f"Real or imaginary part file missing")
                
                re_data = np.loadtxt(re_path).astype(np.float32)
                im_data = np.loadtxt(im_path).astype(np.float32)
                re_mat = re_data.reshape(hamiltonian_size, hamiltonian_size)
                im_mat = im_data.reshape(hamiltonian_size, hamiltonian_size)
                
                shared_hams.append((re_mat, im_mat))
            
            except Exception as e:
                raise RuntimeError(f"Failed to load Hamiltonian {idx}: {str(e)}")
        
        return shared_hams

    def _build_enhanced_graph_from_hamiltonian(self, re_mat: np.ndarray, im_mat: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build enhanced graph structure (extended node features)"""
        size = re_mat.shape[0]
        
        # Enhanced node features (4D: mean, variance, max, min)
        node_feat = np.hstack([
            np.mean(re_mat, axis=1).reshape(-1, 1),
            np.var(re_mat, axis=1).reshape(-1, 1),
            np.mean(im_mat, axis=1).reshape(-1, 1),
            np.var(im_mat, axis=1).reshape(-1, 1)
        ]).astype(np.float32)
        
        # Improved edge weight calculation (added absolute value sum)
        adj_matrix = (np.sqrt(re_mat**2 + im_mat**2) + np.abs(re_mat) + np.abs(im_mat)) / 3
        adj_matrix = adj_matrix.astype(np.float32)
        
        return (
            torch.tensor(node_feat, dtype=torch.float32),
            torch.tensor(adj_matrix, dtype=torch.float32)
        )

    def _build_enhanced_graph_groups_by_size(self) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Build enhanced graph structure groups by size"""
        graph_groups = {}
        for size in self.valid_sizes:
            graph_list = []
            ham_count = min(len(self.shared_full_hams), len(self.energy_data[size]))
            for ham_idx in range(ham_count):
                full_re, full_im = self.shared_full_hams[ham_idx]
                sub_re = full_re[:size, :size]
                sub_im = full_im[:size, :size]
                node_feat, adj = self._build_enhanced_graph_from_hamiltonian(sub_re, sub_im)
                graph_list.append((node_feat, adj))
            graph_groups[size] = graph_list
        return graph_groups

    def _validate_curve_lengths(self):
        """Validate all curve lengths are consistent"""
        if not self.valid_sizes:
            self.energy_curve_len = 0
            return
        
        sample_size = self.valid_sizes[0]
        self.energy_curve_len = len(self.energy_data[sample_size][0])
        
        for size in self.valid_sizes:
            for curve in self.energy_data[size]:
                if len(curve) != self.energy_curve_len:
                    raise ValueError(f"Energy curve length mismatch! Size={size} has curve with length {len(curve)}, standard is {self.energy_curve_len}")

    def _validate_data(self):
        """Enhanced data validation"""
        for size in self.valid_sizes:
            ham_count = len(self.graph_groups[size])
            curve_count = len(self.energy_data[size])
            if ham_count != curve_count:
                raise ValueError(f"Size={size}×{size}: Hamiltonian count {ham_count} does not match curve count {curve_count}")
            
            for ham_idx in range(ham_count):
                node_feat, adj = self.graph_groups[size][ham_idx]
                if node_feat.shape[0] != size or node_feat.shape[1] != self.config.node_feat_dim:
                    raise ValueError(f"{size}×{size} node feature dimension error (index {ham_idx})")
                if adj.shape != (size, size):
                    raise ValueError(f"{size}×{size} adjacency matrix dimension error (index {ham_idx})")
                
                curve = self.energy_data[size][ham_idx]
                if len(curve) != self.energy_curve_len:
                    raise ValueError(f"{size}×{size} curve length error (index {ham_idx})")
                
                # Initial value consistency validation
                recorded_initial = self.sample_to_initial[(size, ham_idx)]
                curve_initial = curve[0]
                if not np.isclose(recorded_initial, curve_initial, atol=1e-6):
                    raise ValueError(
                        f"{size}×{size} sample {ham_idx} initial value mismatch!"
                        f" Recorded={recorded_initial:.6f}, Curve first point={curve_initial:.6f}"
                    )

    def __len__(self) -> int:
        return sum(len(self.energy_data[size]) for size in self.valid_sizes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, np.ndarray, int]:
        """Get data item"""
        cumulative = 0
        target_size = None
        inner_idx = None
        
        for size in self.valid_sizes:
            count = len(self.energy_data[size])
            if cumulative + count > idx:
                target_size = size
                inner_idx = idx - cumulative
                break
            cumulative += count
        
        if target_size is None or inner_idx is None:
            raise IndexError(f"Index {idx} out of range")
        
        node_feat, adj = self.graph_groups[target_size][inner_idx]
        energy_curve = self.energy_data[target_size][inner_idx]
        meta = self.size_to_energy[target_size]
        energy_num = meta["energy_num"]
        time_steps = meta["time_steps"]
        initial_energy = energy_curve[0]
        
        target_energy = torch.tensor(energy_curve, dtype=torch.float32)
        initial_energy_tensor = torch.tensor([[initial_energy]], dtype=torch.float32)
        
        return (node_feat, adj, target_energy, initial_energy_tensor, 
                target_size, energy_num, time_steps, inner_idx)


# -------------------------- 3. Improved graph data batching tool --------------------------
def collate_graphs(batch: List[Tuple]) -> Tuple[Batch, torch.Tensor, torch.Tensor, Tuple, Tuple, Tuple, Tuple]:
    """Batching function (improved edge weight handling)"""
    (node_feats, adjs, targets, initial_energies, 
     sizes, energy_nums, time_steps, ham_indices) = zip(*batch)
    
    data_list = []
    for nf, adj, t in zip(node_feats, adjs, targets):
        # Sparsify edges (only keep non-zero weight edges)
        edge_mask = adj > 1e-6
        edges = edge_mask.nonzero().t()
        if edges.size(1) == 0:
            # Handle all-zero adjacency matrix case
            edges = torch.tensor([[0], [0]], dtype=torch.long)
            edge_weights = torch.tensor([[0.0]], dtype=torch.float32)
        else:
            edge_weights = adj[edges[0], edges[1]].view(-1, 1)
        
        data = Data(
            x=nf,
            edge_index=edges,
            edge_weight=edge_weights,
            y=t
        )
        data_list.append(data)
    
    batch_data = Batch.from_data_list(data_list)
    return (batch_data, 
            torch.stack(targets), 
            torch.cat(initial_energies, dim=0),
            sizes, 
            energy_nums, 
            time_steps,
            ham_indices)


# -------------------------- 4. Enhanced TransformerGNN model --------------------------
class EnergyPredTransformerGNN(nn.Module):
    """Enhanced TransformerGNN model"""
    def __init__(self, config: Config, input_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        
        # Input projection layer (adapts to different node feature dimensions)
        self.input_proj = nn.Linear(input_dim, config.transformer_heads * config.hidden_dim)
        
        # TransformerConv layers (with residual connections)
        self.transformer1 = TransformerConv(
            in_channels=config.transformer_heads * config.hidden_dim,
            out_channels=config.hidden_dim,
            heads=config.transformer_heads,
            edge_dim=1,
            concat=True
        )
        self.transformer2 = TransformerConv(
            in_channels=config.transformer_heads * config.hidden_dim,
            out_channels=config.hidden_dim,
            heads=config.transformer_heads,
            edge_dim=1,
            concat=True
        )
        self.transformer3 = TransformerConv(
            in_channels=config.transformer_heads * config.hidden_dim,
            out_channels=config.hidden_dim,
            heads=config.transformer_heads,
            edge_dim=1,
            concat=True
        )
        
        # Layer normalization (replaces BatchNorm, better for graph data)
        self.ln1 = nn.LayerNorm(config.transformer_heads * config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.transformer_heads * config.hidden_dim)
        self.ln3 = nn.LayerNorm(config.transformer_heads * config.hidden_dim)
        
        # Initial value fusion layer (with normalization)
        self.fc_initial = nn.Sequential(
            nn.Linear(1, config.transformer_heads * config.hidden_dim),
            nn.LayerNorm(config.transformer_heads * config.hidden_dim),
            nn.ReLU()
        )
        
        # Final prediction head (deeper network)
        self.fc = nn.Sequential(
            nn.Linear(2 * config.transformer_heads * config.hidden_dim, config.transformer_heads * config.hidden_dim),
            nn.LayerNorm(config.transformer_heads * config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_heads * config.hidden_dim, config.transformer_heads * config.hidden_dim // 2),
            nn.LayerNorm(config.transformer_heads * config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_heads * config.hidden_dim // 2, output_dim)
        )

    def forward(
        self, 
        batch_data: Batch, 
        initial_energies: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass (with residual connections)"""
        x, edge_index, edge_weight, batch = (
            batch_data.x, 
            batch_data.edge_index, 
            batch_data.edge_weight,
            batch_data.batch
        )
        
        # Input projection
        x = self.input_proj(x)
        
        # Transformer layers (with residual)
        x1 = self.transformer1(x, edge_index, edge_weight)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x = x + x1  # Residual connection
        
        x2 = self.transformer2(x, edge_index, edge_weight)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x = x + x2  # Residual connection
        
        x3 = self.transformer3(x, edge_index, edge_weight)
        x3 = self.ln3(x3)
        x3 = F.relu(x3)
        x = x + x3  # Residual connection
        
        # Global pooling
        graph_features = global_mean_pool(x, batch)
        
        # Initial value processing
        initial_energies = initial_energies.view(-1, 1)
        initial_features = self.fc_initial(initial_energies)
        
        # Feature fusion
        combined_features = torch.cat([graph_features, initial_features], dim=1)
        
        # Prediction output
        output = self.fc(combined_features)
        
        if return_features:
            return output, combined_features
        return output


# -------------------------- 5. Evaluation and visualization utility functions (maintain compatibility) --------------------------
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate R² score"""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    r2_list = []
    for t, p in zip(y_true, y_pred):
        mean_t = np.mean(t)
        ss_total = np.sum((t - mean_t) ** 2)
        ss_residual = np.sum((t - p) ** 2)
        r2 = 1 - (ss_residual / (ss_total + 1e-8))
        r2_list.append(r2)
    
    return np.mean(r2_list) if r2_list else 0.0


def mean_energy_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean sequence R²"""
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_pred, axis=0)
    
    mean_t = np.mean(y_true_mean)
    ss_total = np.sum((y_true_mean - mean_t) ** 2)
    ss_residual = np.sum((y_true_mean - y_pred_mean) ** 2)
    return 1 - (ss_residual / (ss_total + 1e-8)) if ss_total > 0 else 0.0


def save_energy_pred_to_csv(
    y_true_mean: np.ndarray, y_pred_mean: np.ndarray, true_fit: Dict, pred_fit: Dict,
    size: int, energy_num: int, time_steps: np.ndarray, dataset_type: str,
    save_dir: str, num_samples: int
) -> Tuple[str, str]:
    """Save prediction results to CSV"""
    csv_data = {
        "time_step_fs": time_steps,
        "true_energy_eV": y_true_mean,
        "predicted_energy_eV": y_pred_mean,
        "true_fit_energy_eV": true_fit["y_fit"],
        "pred_fit_energy_eV": pred_fit["y_fit"],
        "energy_num": [energy_num] * len(time_steps),
        "size": [f"{size}×{size}"] * len(time_steps),
        "true_tau_fs": [true_fit["tau_D"]] * len(time_steps),
        "pred_tau_fs": [pred_fit["tau_D"]] * len(time_steps),
        "true_initial_eV": [true_fit["initial_value"]] * len(time_steps),
        "true_fit_initial_eV": [true_fit["fit_initial_value"]] * len(time_steps),
        "true_initial_diff": [true_fit["initial_diff"]] * len(time_steps),
        "pred_initial_eV": [pred_fit["initial_value"]] * len(time_steps),
        "pred_fit_initial_eV": [pred_fit["fit_initial_value"]] * len(time_steps),
        "pred_initial_diff": [pred_fit["initial_diff"]] * len(time_steps),
        "mean_samples_count": [num_samples] * len(time_steps),
        "dataset_type": [dataset_type.upper()] * len(time_steps)
    }
    
    df = pd.DataFrame(csv_data)
    csv_save_dir = os.path.join(save_dir, "csv")
    os.makedirs(csv_save_dir, exist_ok=True)
    csv_filename = f"{dataset_type}_energy{energy_num}_{size}x{size}_meanN{num_samples}_with_fit.csv"
    csv_save_path = os.path.join(csv_save_dir, csv_filename)
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    
    return csv_filename, csv_save_path


def plot_mean_energy_prediction(
    y_true_mean: np.ndarray, y_pred_mean: np.ndarray, true_fit: Dict, pred_fit: Dict,
    size: int, energy_num: int, time_steps: np.ndarray, dataset_type: str,
    save_dir: str, num_samples: int
) -> Tuple[str, str]:
    """Visualize prediction results"""
    avg_r2 = mean_energy_r2(np.array([y_true_mean]), np.array([y_pred_mean]))
    
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, y_true_mean, label="True Energy Curve", color="#2E86AB", linewidth=3.0)
    plt.plot(time_steps, y_pred_mean, label="Averaged Predicted Curve", color="#A23B72", linewidth=3.0)
    plt.plot(time_steps, true_fit["y_fit"], label="True Gaussian Fit", color="#2E86AB", linewidth=2.5, linestyle=':')
    plt.plot(time_steps, pred_fit["y_fit"], label="Predicted Gaussian Fit", color="#A23B72", linewidth=2.5, linestyle=':')
    
    plt.title(
        f"Energy Prediction - Energy_num={energy_num} (Size={size}×{size})\n"
        f"{dataset_type.upper()} | Mean Samples={num_samples} | R²={avg_r2:.4f}\n"
        f"True τ={true_fit['tau_D']:.2f} fs | Pred τ={pred_fit['tau_D']:.2f} fs",
        fontsize=12, pad=15
    )
    plt.xlabel("Time (fs)", fontsize=10)
    plt.ylabel("Energy (eV)", fontsize=10)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_filename = f"{dataset_type}_energy{energy_num}_{size}x{size}_pred_with_fit.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return save_filename, save_path


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Visualize training history"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="#2E86AB", linewidth=2)
    plt.plot(history["test_loss"], label="Test Loss", color="#A23B72", linewidth=2)
    plt.title("MSE Loss", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_r2"], label="Train R²", color="#2E86AB", linewidth=2)
    plt.plot(history["test_r2"], label="Test R²", color="#A23B72", linewidth=2)
    plt.title("R² Score", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("R²", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history saved to: {save_path}")
    plt.close()


def predict_all_energies_with_mean(
    model: nn.Module, dataset: Dataset, dataset_type: str, 
    device: torch.device, save_dir: str, num_samples_for_mean: Union[str, int] = "all"
) -> Tuple[int, int]:
    """Batch prediction and visualization"""
    model.eval()
    random.seed(42)
    all_sizes = sorted(list(set([dataset[i][4] for i in range(len(dataset))])))
    total_plots = 0
    total_csvs = 0
    
    print(f"\n=== {dataset_type.upper()} Set Prediction ===")
    for size in all_sizes:
        size_indices = [i for i in range(len(dataset)) if dataset[i][4] == size]
        if not size_indices:
            print(f"⚠️ Skipping size={size}×{size} (no data)")
            continue
        
        selected_indices = size_indices if num_samples_for_mean == "all" else random.sample(size_indices, min(num_samples_for_mean, len(size_indices)))
        num_samples = len(selected_indices)
        if num_samples < 1:
            print(f"⚠️ Skipping size={size}×{size} (insufficient samples)")
            continue
        
        all_preds = []
        true_curve = None
        energy_num = None
        time_steps = None
        
        with torch.no_grad():
            for idx in selected_indices:
                (node_feat, adj, target_energy, initial_energy, 
                 _, en_num, t_steps, _) = dataset[idx]
                if true_curve is None:
                    true_curve = target_energy.numpy().copy()
                    energy_num = en_num
                    time_steps = t_steps
                
                edges = adj.nonzero().t()
                edge_weights = adj[edges[0], edges[1]].view(-1, 1)
                data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
                batch_data = Batch.from_data_list([data])
                initial_energy = initial_energy.to(device)
                
                pred = model(batch_data, initial_energy).cpu().numpy().flatten()
                all_preds.append(pred)
        
        pred_mean = np.mean(np.array(all_preds), axis=0)
        
        # Improved fitting call
        true_fit_result = fit_gaussian_decay(time_steps, true_curve, enforce_initial=True)
        pred_fit_result = fit_gaussian_decay(time_steps, pred_mean, enforce_initial=True)
        
        # Save results
        plot_name, _ = plot_mean_energy_prediction(
            y_true_mean=true_curve, y_pred_mean=pred_mean,
            true_fit=true_fit_result, pred_fit=pred_fit_result,
            size=size, energy_num=energy_num, time_steps=time_steps,
            dataset_type=dataset_type, save_dir=save_dir, num_samples=num_samples
        )
        
        csv_name, _ = save_energy_pred_to_csv(
            y_true_mean=true_curve, y_pred_mean=pred_mean,
            true_fit=true_fit_result, pred_fit=pred_fit_result,
            size=size, energy_num=energy_num, time_steps=time_steps,
            dataset_type=dataset_type, save_dir=save_dir, num_samples=num_samples
        )
        
        avg_r2 = mean_energy_r2(np.array([true_curve]), np.array([pred_mean]))
        print(f"✅ Processed size={size}×{size} → Mean R²={avg_r2:.4f}")
        print(f"   - Fitted lifetime: True τ={true_fit_result['tau_D']:.2f}fs | Predicted τ={pred_fit_result['tau_D']:.2f}fs")
        print(f"   - Plot: {plot_name}")
        print(f"   - CSV: {csv_name}")
        
        total_plots += 1
        total_csvs += 1
    
    return total_plots, total_csvs


def collect_mean_r2_results(
    dataset: Dataset, dataset_type: str, model: nn.Module, 
    device: torch.device, save_dir: str
) -> Tuple[pd.DataFrame, str]:
    """Collect mean R² results"""
    model.eval()
    r2_records = []
    all_sizes = sorted(list(set([dataset[i][4] for i in range(len(dataset))])))
    
    print(f"\n=== Starting to collect {dataset_type.upper()} set mean R² results ===")
    for size in all_sizes:
        size_indices = [i for i in range(len(dataset)) if dataset[i][4] == size]
        if not size_indices:
            print(f"⚠️ Skipping size={size}×{size} (no data)")
            continue
        
        sample_idx = size_indices[0]
        _, _, _, _, _, energy_num, time_steps, _ = dataset[sample_idx]
        
        size_r2_list = []
        with torch.no_grad():
            for idx in size_indices:
                node_feat, adj, target_energy, initial_energy, _, _, _, _ = dataset[idx]
                
                edges = adj.nonzero().t()
                edge_weights = adj[edges[0], edges[1]].view(-1, 1)
                data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
                batch_data = Batch.from_data_list([data])
                initial_energy = initial_energy.to(device)
                
                pred = model(batch_data, initial_energy).cpu().numpy().flatten()
                true = target_energy.numpy().flatten()
                
                mean_true = np.mean(true)
                ss_total = np.sum((true - mean_true) ** 2)
                ss_residual = np.sum((true - pred) ** 2)
                single_r2 = 1 - (ss_residual / (ss_total + 1e-8))
                size_r2_list.append(single_r2)
        
        size_avg_r2 = np.mean(size_r2_list)
        size_std_r2 = np.std(size_r2_list)
        
        r2_records.append({
            "dataset_type": dataset_type.upper(),
            "size": size,
            "size_index": f"{size}×{size}",
            "energy_num": energy_num,
            "sample_count": len(size_indices),
            "mean_r2": round(size_avg_r2, 6),
            "std_r2": round(size_std_r2, 6),
            "ranking_key": size_avg_r2
        })
    
    r2_df = pd.DataFrame(r2_records)
    r2_df_sorted = r2_df.sort_values(
        by=["ranking_key", "size"], 
        ascending=[False, True]
    ).reset_index(drop=True)
    r2_df_sorted["ranking"] = range(1, len(r2_df_sorted) + 1)
    
    final_columns = [
        "ranking", "dataset_type", "size", "size_index", "energy_num",
        "sample_count", "mean_r2", "std_r2"
    ]
    r2_df_sorted = r2_df_sorted[final_columns]
    
    csv_save_dir = os.path.join(save_dir, "mean_r2_summary")
    os.makedirs(csv_save_dir, exist_ok=True)
    csv_filename = f"{dataset_type}_mean_r2_summary_sorted.csv"
    csv_save_path = os.path.join(csv_save_dir, csv_filename)
    r2_df_sorted.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    
    overall_mean_r2 = r2_df_sorted["mean_r2"].mean()
    print(f"✅ {dataset_type.upper()} set mean R² summary completed:")
    print(f"   - Number of plots: {len(r2_df_sorted)}")
    print(f"   - Overall mean R²: {round(overall_mean_r2, 6)}")
    print(f"   - Save path: {csv_save_path}")
    
    return r2_df_sorted, csv_save_path


def extract_mean_encoded_features(
    model: nn.Module, dataset: Dataset, device: torch.device,
    dataset_type: str, save_dir: str
) -> Tuple[pd.DataFrame, str]:
    """Extract mean encoded features"""
    model.eval()
    energy_groups = {}
    mean_enc_dir = os.path.join(save_dir, "mean_encoded_features")
    os.makedirs(mean_enc_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            (node_feat, adj, _, initial_energy, 
             size, energy_num, _, ham_idx) = dataset[idx]
            
            edges = adj.nonzero().t()
            edge_weights = adj[edges[0], edges[1]].view(-1, 1)
            data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
            batch_data = Batch.from_data_list([data])
            init_energy_tensor = initial_energy.to(device)
            
            _, enc_feat = model(batch_data, init_energy_tensor, return_features=True)
            enc_feat_np = enc_feat.cpu().numpy().flatten()
            
            if energy_num not in energy_groups:
                energy_groups[energy_num] = {
                    "encoded_features": [],
                    "size": size,
                    "count": 0,
                    "initial_energies": []
                }
            energy_groups[energy_num]["encoded_features"].append(enc_feat_np)
            energy_groups[energy_num]["initial_energies"].append(initial_energy.item())
            energy_groups[energy_num]["count"] += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples...")
    
    mean_encoded_features = []
    for energy_num, group in energy_groups.items():
        size = group["size"]
        count = group["count"]
        enc_feats_array = np.array(group["encoded_features"])
        
        enc_mean = np.mean(enc_feats_array, axis=0)
        initial_energy_mean = np.mean(group["initial_energies"])
        
        enc_mean_dict = {
            "energy_num": energy_num,
            "size": size,
            "size_index": f"{size}×{size}",
            "sample_count": count,
            "dataset_type": dataset_type,
            "initial_energy_mean_eV": initial_energy_mean
        }
        
        for feat_idx, feat_val in enumerate(enc_mean):
            enc_mean_dict[f"mean_encoded_feat_{feat_idx}"] = feat_val
        
        mean_encoded_features.append(enc_mean_dict)
    
    mean_enc_df = pd.DataFrame(mean_encoded_features)
    mean_enc_path = os.path.join(mean_enc_dir, f"{dataset_type}_mean_encoded_features.csv")
    mean_enc_df.to_csv(mean_enc_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ {dataset_type} mean encoded features saved successfully:")
    print(f"   - Number of energy indices: {len(mean_enc_df)}")
    print(f"   - Feature dimension: {len([col for col in mean_enc_df.columns if col.startswith('mean_encoded_feat_')])}")
    print(f"   - Save path: {mean_enc_path}")
    
    return mean_enc_df, mean_enc_path


# -------------------------- 6. Main workflow functions --------------------------
def regenerate_graph_dataset(config: Config, save_initial: bool = True):
    """Generate dataset"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.res, exist_ok=True)
    os.makedirs(config.res_target_energy, exist_ok=True)
    
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    
    full_dataset = GraphEnergyDataset(
        config=config, 
        ham_count_per_group=config.ham_count_per_group,
        load_initial_from_csv=not save_initial,
        initial_csv_path=os.path.join(config.save_dir, "energy_initial_values.csv") if not save_initial else None
    )
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty, please check file paths")
    
    # Split training and test sets
    all_energy_nums = []
    for size in full_dataset.valid_sizes:
        en_num = full_dataset.size_to_energy[size]["energy_num"]
        all_energy_nums.append(en_num)
    all_energy_nums = sorted(list(set(all_energy_nums)))
    
    if len(all_energy_nums) < config.train_energy_count:
        raise ValueError(f"Insufficient energy indices available! Have {len(all_energy_nums)}, need {config.train_energy_count} for training set")
    
    min_en = all_energy_nums[0]
    max_en = all_energy_nums[-1]
    train_energy_indices = [min_en, max_en]
    
    remaining_ens = [en for en in all_energy_nums if en not in train_energy_indices]
    need = config.train_energy_count - 2
    
    if need > 0:
        if len(remaining_ens) < need:
            raise ValueError(f"Insufficient remaining energy indices! Need {need} more, but only found {len(remaining_ens)}")
        
        step = max(1, len(remaining_ens) // need)
        additional_ens = remaining_ens[::step][:need]
        
        if len(additional_ens) < need:
            remaining_after = [en for en in remaining_ens if en not in additional_ens]
            additional_ens += random.sample(remaining_after, need - len(additional_ens))
        
        train_energy_indices.extend(additional_ens)
        train_energy_indices = sorted(list(set(train_energy_indices)))
    
    test_energy_indices = [en for en in all_energy_nums if en not in train_energy_indices]
    
    print(f"\n=== Dataset Splitting ===")
    print(f"All energy indices: {all_energy_nums}")
    print(f"Training set energy indices ({len(train_energy_indices)}): {sorted(train_energy_indices)}")
    print(f"Test set energy indices ({len(test_energy_indices)}): {sorted(test_energy_indices)}")
    
    # Build indices
    train_indices = []
    test_indices = []
    current_idx = 0
    
    for size in full_dataset.valid_sizes:
        en_num = full_dataset.size_to_energy[size]["energy_num"]
        count = len(full_dataset.energy_data[size])
        
        if en_num in train_energy_indices:
            train_indices.extend(range(current_idx, current_idx + count))
        else:
            test_indices.extend(range(current_idx, current_idx + count))
            
        current_idx += count
    
    # Create datasets
    train_data = [full_dataset[i] for i in train_indices]
    test_data = [full_dataset[i] for i in test_indices]
    
    # Save
    with open(os.path.join(config.save_dir, "train_graph_energy_dataset.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(config.save_dir, "test_graph_energy_dataset.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    
    data_params = {
        "energy_curve_len": full_dataset.energy_curve_len,
        "output_dim": full_dataset.output_dim,
        "train_energy_indices": sorted(train_energy_indices),
        "test_energy_indices": sorted(test_energy_indices),
        "node_feat_dim": config.node_feat_dim,
        "ham_count_per_group": config.ham_count_per_group,
        "initial_csv_path": full_dataset.initial_save_csv
    }
    
    with open(os.path.join(config.save_dir, "graph_energy_data_params.pkl"), "wb") as f:
        pickle.dump(data_params, f)
    
    print(f"\n✅ Dataset saved successfully:")
    print(f"  - Training set: {len(train_data)} samples | Test set: {len(test_data)} samples")
    print(f"  - Save path: {config.save_dir}")


def main_training(config: Config):
    """Main training workflow"""
    # Set random seeds
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.prediction_save_dir, exist_ok=True)
    os.makedirs(config.features_save_dir, exist_ok=True)
    train_pred_dir = os.path.join(config.prediction_save_dir, "train")
    test_pred_dir = os.path.join(config.prediction_save_dir, "test")
    os.makedirs(train_pred_dir, exist_ok=True)
    os.makedirs(test_pred_dir, exist_ok=True)
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    required_files = [
        "train_graph_energy_dataset.pkl", 
        "test_graph_energy_dataset.pkl", 
        "graph_energy_data_params.pkl"
    ]
    
    for file in required_files:
        file_path = os.path.join(config.save_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}\nPlease run regenerate_graph_dataset() first")
    
    with open(os.path.join(config.save_dir, "train_graph_energy_dataset.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(config.save_dir, "test_graph_energy_dataset.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(config.save_dir, "graph_energy_data_params.pkl"), "rb") as f:
        data_params = pickle.load(f)
    
    # Wrap dataset
    class WrappedEnergyDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = WrappedEnergyDataset(train_data)
    test_dataset = WrappedEnergyDataset(test_data)
    
    # Data loaders (set num_workers to 0 on Windows)
    num_workers = 0 if sys.platform == "win32" else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = EnergyPredTransformerGNN(
        config=config,
        input_dim=data_params["node_feat_dim"],
        output_dim=data_params["output_dim"]
    ).to(device)
    print(model)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(  # Use AdamW optimizer
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(  # Better learning rate scheduler
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training process
    print("\n=== Starting Training ===")
    history = {
        "train_loss": [], "train_r2": [],
        "test_loss": [], "test_r2": []
    }
    best_test_loss = float("inf")
    best_model_path = os.path.join(config.model_save_dir, "best_model.pth")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_total_r2 = 0.0
        
        for batch in train_loader:
            batch_data, targets, initial_energies, _, _, _, _ = batch
            batch_data = batch_data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            initial_energies = initial_energies.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(batch_data, initial_energies)
            loss = F.mse_loss(preds, targets)
            batch_r2 = r2_score(targets, preds)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
                batch_data, targets, initial_energies, _, _, _, _ = batch
                batch_data = batch_data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                initial_energies = initial_energies.to(device, non_blocking=True)
                
                preds = model(batch_data, initial_energies)
                loss = F.mse_loss(preds, targets)
                batch_r2 = r2_score(targets, preds)
                
                test_total_loss += loss.item() * preds.size(0)
                test_total_r2 += batch_r2 * preds.size(0)
        
        avg_test_loss = test_total_loss / len(test_dataset)
        avg_test_r2 = test_total_r2 / len(test_dataset)
        history["test_loss"].append(avg_test_loss)
        history["test_r2"].append(avg_test_r2)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_test_loss,
                'config': config
            }, best_model_path)
            print(f"📌 Epoch {epoch+1}: Saving best model (Test loss: {best_test_loss:.6f})")
        
        # Print progress
        if (epoch + 1) % config.print_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch+1}/{config.epochs}]")
            print(f"  Learning rate: {lr:.2e}")
            print(f"  Training set: Loss={avg_train_loss:.6f}, R²={avg_train_r2:.4f}")
            print(f"  Test set: Loss={avg_test_loss:.6f}, R²={avg_test_r2:.4f}")
    
    # Plot training history
    print("\n=== Plotting Training History ===")
    history_path = os.path.join(config.model_save_dir, "training_history.png")
    plot_training_history(history, history_path)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Batch prediction
    print("\n=== Batch Prediction ===")
    train_plots, train_csvs = predict_all_energies_with_mean(
        model=model, dataset=train_dataset, dataset_type="train",
        device=device, save_dir=train_pred_dir
    )
    
    test_plots, test_csvs = predict_all_energies_with_mean(
        model=model, dataset=test_dataset, dataset_type="test",
        device=device, save_dir=test_pred_dir
    )
    
    # Collect R² results
    print("\n=== Collecting Mean R² Results ===")
    train_r2_df, train_r2_csv = collect_mean_r2_results(
        dataset=train_dataset, dataset_type="train",
        model=model, device=device, save_dir=config.prediction_save_dir
    )
    
    test_r2_df, test_r2_csv = collect_mean_r2_results(
        dataset=test_dataset, dataset_type="test",
        model=model, device=device, save_dir=config.prediction_save_dir
    )
    
    # Extract encoded features
    print("\n=== Extracting and Saving Mean Encoded Features ===")
    train_mean_enc_df, train_mean_enc_path = extract_mean_encoded_features(
        model=model, dataset=train_dataset, device=device,
        dataset_type="train", save_dir=config.features_save_dir
    )
    
    test_mean_enc_df, test_mean_enc_path = extract_mean_encoded_features(
        model=model, dataset=test_dataset, device=device,
        dataset_type="test", save_dir=config.features_save_dir
    )
    
    print(f"\n🎉 All tasks completed!")
    print(f"1. Best model: {best_model_path}")
    print(f"2. Training history: {history_path}")
    print(f"3. Training set plots: {train_plots}, Mean R²: {round(train_r2_df['mean_r2'].mean(), 6)}")
    print(f"4. Test set plots: {test_plots}, Mean R²: {round(test_r2_df['mean_r2'].mean(), 6)}")
    print(f"5. Mean encoded features save directory: {config.features_save_dir}")


# -------------------------- Main program entry --------------------------
if __name__ == "__main__":
    # Create config instance
    config = Config(
        train_energy_count=7,
        epochs=100,
        batch_size=32
    )
    
    # Step 1: Generate dataset
    regenerate_graph_dataset(config, save_initial=True)
    
    # Step 2: Train model
    main_training(config)