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


# -------------------------- 1. Improved Gaussian Curve Fitting Tool (Ensure Initial Value Correspondence) --------------------------
def gaussian_decay(t, A, tau_D, t0=0):
    """Improved Gaussian decay function model: Ensure accurate initial value at t=0"""
    return A * np.exp(-0.5 * ((t - t0) / tau_D) ** 2)

def constrained_gaussian_decay(t, tau_D):
    """Constrained Gaussian decay function: Force value at t=0 to equal curve initial value"""
    global initial_value  # Use global variable to pass initial value constraint
    A = initial_value  # Ensure value at t=0 equals initial value
    return A * np.exp(-0.5 * (t / tau_D) ** 2)

def fit_gaussian_decay(x_data, y_data, enforce_initial=True):
    """Fit Gaussian decay function and ensure initial value matches
    enforce_initial: Whether to force fitted curve to pass through initial value point (y_data[0])
    """
    global initial_value  # Access initial value in constraint function
    initial_value = y_data[0]  # Record curve initial value
    
    try:
        if enforce_initial:
            # Force fitted curve to pass through initial value point, only optimize tau_D parameter
            initial_guess = [len(x_data) / 4]  # Only need to guess tau_D
            params, params_covariance = curve_fit(
                constrained_gaussian_decay, 
                x_data, 
                y_data, 
                p0=initial_guess,
                maxfev=10000
            )
            tau_D = params[0]
            A = initial_value  # A value determined by initial value
        else:
            # No forced constraint, optimize both A and tau_D
            initial_guess = [initial_value, len(x_data) / 4]
            params, params_covariance = curve_fit(
                gaussian_decay, 
                x_data, 
                y_data, 
                p0=initial_guess,
                maxfev=10000
            )
            A, tau_D = params
        
        # Generate fitted curve
        if enforce_initial:
            y_fit = constrained_gaussian_decay(x_data, tau_D)
        else:
            y_fit = gaussian_decay(x_data, A, tau_D)
            
        # Verify if initial value of fitted curve matches original curve
        fit_initial = y_fit[0]
        original_initial = y_data[0]
        initial_diff = abs(fit_initial - original_initial)
        
        return {
            'A': A,
            'tau_D': tau_D,
            'y_fit': y_fit,
            'success': True,
            'initial_value': original_initial,
            'fit_initial_value': fit_initial,
            'initial_diff': initial_diff  # Initial value difference (should be close to 0)
        }
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        return {
            'A': np.nan,
            'tau_D': np.nan,
            'y_fit': np.zeros_like(x_data),
            'success': False,
            'initial_value': y_data[0] if len(y_data) > 0 else np.nan,
            'fit_initial_value': np.nan,
            'initial_diff': np.nan
        }


# -------------------------- 2. Graph Structure Dataset Class (Only Process Initial Energy) --------------------------
class GraphEnergyDataset(Dataset):
    """Only extract the first value of the second column in jj-en.dat as initial energy, do not save complete curve"""
    def __init__(self, config, ham_count_per_group=400, load_initial_from_csv=False, initial_csv_path=None):
        self.config = config
        self.ham_count_per_group = ham_count_per_group
        self.load_initial_from_csv = load_initial_from_csv  # Whether to load initial values from CSV
        self.initial_csv_path = initial_csv_path  # Initial energy CSV path
        self.initial_save_csv = os.path.join(config["save_dir"], "energy_initial_values.csv")  # Initial energy save path
        
        # 1. Load Energy data (only extract initial values)
        if self.load_initial_from_csv:
            if not initial_csv_path or not os.path.exists(initial_csv_path):
                raise ValueError(f"Failed to load initial values: File does not exist → {initial_csv_path}")
            self.energy_data, self.size_to_energy, self.sample_to_initial = self._load_energy_from_initial_csv()
        else:
            self.energy_data, self.size_to_energy, self.sample_to_initial = self._load_energy_and_save_initial_csv()
        
        self.valid_sizes = sorted(self.size_to_energy.keys())
        self.max_size = max(self.valid_sizes) if self.valid_sizes else 0
        
        # 2. Preload Hamiltonians and build graph structures
        self.shared_full_hams = self._load_shared_full_hamiltonians()
        self.graph_groups = self._build_graph_groups_by_size()
        
        # 3. Calculate output dimension (Energy curve length)
        if self.valid_sizes:
            sample_size = self.valid_sizes[0]
            self.energy_curve_len = len(self.energy_data[sample_size][0])
            # Verify all curve lengths are consistent
            for size in self.valid_sizes:
                for curve in self.energy_data[size]:
                    if len(curve) != self.energy_curve_len:
                        raise ValueError(f"Inconsistent Energy curve lengths! Size={size}某曲线长度为{len(curve)}, standard is {self.energy_curve_len}")
        else:
            self.energy_curve_len = 0
        self.output_dim = self.energy_curve_len
        
        # 4. Validate data consistency (add verification of initial value and curve start point)
        self._validate_data()
        print(f"✅ Dataset initialization completed:")
        print(f"  - Maximum Size: {self.max_size}×{self.max_size} (Graph node count: {self.max_size})")
        print(f"  - Output dimension (Energy curve length): {self.output_dim}")
        print(f"  - Valid Size list: {self.valid_sizes}")
        print(f"  - Initial energy file: {self.initial_save_csv}" if not load_initial_from_csv else f"  - Loaded initial values from: {initial_csv_path}")

    def _save_initial_to_csv(self, initial_data):
        """Save initial energy of all samples to single CSV file (only save initial values, not complete curves)"""
        df = pd.DataFrame(initial_data)
        df.to_csv(self.initial_save_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Initial energy saved successfully: {self.initial_save_csv} ({len(df)} samples total)")
        return self.initial_save_csv

    def _load_energy_and_save_initial_csv(self):
        """Load Energy files, only extract first value of second column as initial energy and save"""
        energy_data = {}  # Complete curves (only for training)
        size_to_energy = {}  # Metadata
        sample_to_initial = {}  # Initial energy mapping
        initial_data_list = []  # Initial energy data for CSV saving
        start_en_num = 30
        end_en_num = 100
        global_sample_idx = 0
        
        for en_num in range(start_en_num, end_en_num + 1):
            size = en_num
            en_filename = f"jj-en{en_num}.dat"
            en_filepath = os.path.join(self.config["res_target_pop"], en_filename)
            
            try:
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                base_energy_curve = en_raw[:, 1]
                # Extract initial energy: first value of second column (i.e., first point of curve)
                base_initial_energy = base_energy_curve[0]
                
                # Generate corresponding curves for each Hamiltonian (simulate multiple samples)
                size_curves = []
                for ham_idx in range(self.ham_count_per_group):
                    # Base curve + small noise to simulate multiple samples
                    noise = np.random.normal(0, 0.001, size=len(base_energy_curve))
                    single_curve = (base_energy_curve + noise).astype(np.float32)
                    size_curves.append(single_curve)
                    
                    # Record initial energy of this sample (first point with noise)
                    sample_initial = single_curve[0]
                    sample_key = (size, ham_idx)
                    sample_to_initial[sample_key] = sample_initial
                    
                    # Collect initial energy data (for CSV saving)
                    initial_data_list.append({
                        "global_sample_idx": global_sample_idx,
                        "size": size,
                        "size_str": f"{size}×{size}",
                        "energy_num": en_num,
                        "ham_idx": ham_idx,
                        "initial_energy_eV": sample_initial,
                        "curve_first_point_eV": single_curve[0]  # Explicitly record first curve point for verification
                    })
                    global_sample_idx += 1
                
                energy_data[size] = size_curves
                size_to_energy[size] = {
                    "energy_num": en_num,
                    "filename": en_filename,
                    "time_steps": time_steps,
                    "curve_count": len(size_curves)
                }
                print(f"✅ Processed: Energy_num={en_num} (size={size}×{size}, {len(size_curves)} samples, initial energy={base_initial_energy:.6f}eV)")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath}")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load {en_filename}: {str(e)} (skipped)")
                continue
        
        # Save initial energy to CSV (only initial values, not complete curves)
        if initial_data_list:
            self._save_initial_to_csv(initial_data_list)
        
        return energy_data, size_to_energy, sample_to_initial

    def _load_energy_from_initial_csv(self):
        """Load initial energy from CSV, and load original curves for training"""
        energy_data = {}
        size_to_energy = {}
        sample_to_initial = {}
        
        # Read initial energy CSV
        df = pd.read_csv(self.initial_csv_path)
        required_cols = ["size", "ham_idx", "energy_num", "initial_energy_eV", "curve_first_point_eV"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Initial energy CSV missing required columns! Need to include: {required_cols}")
        
        # Process by size groups
        size_groups = df.groupby("size")
        for size, size_df in size_groups:
            size = int(size)
            energy_num = size_df["energy_num"].iloc[0]
            en_filename = f"jj-en{energy_num}.dat"
            en_filepath = os.path.join(self.config["res_target_pop"], en_filename)
            
            try:
                # Load original Energy file to get complete curves (for training)
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                base_energy_curve = en_raw[:, 1]
                
                # Generate curves matching sample count in CSV
                size_curves = []
                for _, row in size_df.iterrows():
                    ham_idx = int(row["ham_idx"])
                    target_initial = float(row["initial_energy_eV"])
                    curve_first_point = float(row["curve_first_point_eV"])
                    
                    # Verify if initial value in CSV matches first curve point
                    if not np.isclose(target_initial, curve_first_point, atol=1e-6):
                        print(f"⚠️ Warning: Sample (size={size}, ham_idx={ham_idx}) initial value does not match curve first point")
                    
                    # Generate curve with specified initial value
                    noise = np.random.normal(0, 0.001, size=len(base_energy_curve))
                    noise[0] = target_initial - base_energy_curve[0]  # Ensure initial value matches
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
                print(f"✅ Loaded: size={size}×{size} ({len(size_curves)} samples, energy_num={energy_num})")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath} (skipped size={size})")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load size={size}: {str(e)} (skipped)")
                continue
        
        return energy_data, size_to_energy, sample_to_initial

    def _load_shared_full_hamiltonians(self):
        """Load Hamiltonian matrices"""
        shared_hams = []
        filename_template = "0_Ham_{idx}_{part}"
        hamiltonian_size = 100  # Hamiltonian matrix size
        
        for idx in range(self.ham_count_per_group):
            re_path = os.path.join(self.config["res"], filename_template.format(idx=idx, part="re"))
            im_path = os.path.join(self.config["res"], filename_template.format(idx=idx, part="im"))
            
            try:
                re_data = np.loadtxt(re_path).astype(np.float32)
                im_data = np.loadtxt(im_path).astype(np.float32)
                re_mat = re_data.reshape(hamiltonian_size, hamiltonian_size)
                im_mat = im_data.reshape(hamiltonian_size, hamiltonian_size)
                
                if re_mat.shape != im_mat.shape:
                    raise ValueError(f"Hamiltonian {idx}: Real part dimension {re_mat.shape} does not match imaginary part dimension {im_mat.shape}")
                if hamiltonian_size < self.max_size:
                    raise ValueError(f"Hamiltonian size {hamiltonian_size} < maximum required size {self.max_size}")
                
                shared_hams.append((re_mat, im_mat))
            
            except FileNotFoundError:
                raise FileNotFoundError(f"Hamiltonian {idx} missing: {re_path} or {im_path}")
            except ValueError as e:
                raise ValueError(f"Hamiltonian {idx} processing failed: {str(e)}")
        
        return shared_hams

    def _build_graph_from_hamiltonian(self, re_mat, im_mat):
        """Build graph structure from Hamiltonian"""
        size = re_mat.shape[0]
        
        # Node features (2D: real part mean + imaginary part mean)
        node_feat = np.hstack([
            np.mean(re_mat, axis=1).reshape(-1, 1),
            np.mean(im_mat, axis=1).reshape(-1, 1)
        ]).astype(np.float32)
        
        # Edge weight matrix (interaction strength)
        adj_matrix = np.sqrt(re_mat**2 + im_mat**2).astype(np.float32)
        
        return (
            torch.tensor(node_feat, dtype=torch.float32),
            torch.tensor(adj_matrix, dtype=torch.float32)
        )

    def _build_graph_groups_by_size(self):
        """Build graph structure groups by size"""
        graph_groups = {}
        for size in self.valid_sizes:
            graph_list = []
            ham_count = min(len(self.shared_full_hams), len(self.energy_data[size]))
            for ham_idx in range(ham_count):
                full_re, full_im = self.shared_full_hams[ham_idx]
                sub_re = full_re[:size, :size]
                sub_im = full_im[:size, :size]
                node_feat, adj = self._build_graph_from_hamiltonian(sub_re, sub_im)
                graph_list.append((node_feat, adj))
            graph_groups[size] = graph_list
        return graph_groups

    def _validate_data(self):
        """Validate data consistency (enhanced verification of initial value and curve correspondence)"""
        for size in self.valid_sizes:
            ham_count = len(self.graph_groups[size])
            curve_count = len(self.energy_data[size])
            if ham_count != curve_count:
                raise ValueError(f"Size={size}×{size}: Hamiltonian count {ham_count} does not match curve count {curve_count}")
            
            for ham_idx in range(ham_count):
                # Verify graph structure dimensions
                node_feat, adj = self.graph_groups[size][ham_idx]
                if node_feat.shape[0] != size or adj.shape != (size, size):
                    raise ValueError(f"{size}×{size} graph structure dimension error (index {ham_idx})")
                
                # Verify curve length
                curve = self.energy_data[size][ham_idx]
                if len(curve) != self.energy_curve_len:
                    raise ValueError(f"{size}×{size} curve length error (index {ham_idx})")
                
                # Critical verification: whether initial value matches first curve point
                recorded_initial = self.sample_to_initial[(size, ham_idx)]
                curve_initial = curve[0]
                if not np.isclose(recorded_initial, curve_initial, atol=1e-6):
                    raise ValueError(
                        f"{size}×{size} sample {ham_idx} initial value mismatch!"
                        f"Recorded value={recorded_initial:.6f}, curve first point={curve_initial:.6f}"
                    )

    def __len__(self):
        total = 0
        for size in self.valid_sizes:
            total += len(self.energy_data[size])
        return total

    def __getitem__(self, idx):
        """Return graph structure + Energy curve + initial value + metadata"""
        # Calculate size and internal index corresponding to global index
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
        
        # Get graph structure and energy curve
        node_feat, adj = self.graph_groups[target_size][inner_idx]
        energy_curve = self.energy_data[target_size][inner_idx]
        meta = self.size_to_energy[target_size]
        energy_num = meta["energy_num"]
        time_steps = meta["time_steps"]
        
        # Get initial energy value (first curve point)
        initial_energy = energy_curve[0]  # Directly get from curve to ensure consistency
        sample_to_initial = self.sample_to_initial[(target_size, inner_idx)]
        
        # Verify consistency again
        if not np.isclose(initial_energy, sample_to_initial, atol=1e-6):
            print(f"⚠️ Sample ({target_size}, {inner_idx}) initial value inconsistency: curve={initial_energy:.6f}, recorded={sample_to_initial:.6f}")
        
        # Convert to tensors
        target_energy = torch.tensor(energy_curve, dtype=torch.float32)
        initial_energy_tensor = torch.tensor([[initial_energy]], dtype=torch.float32)
        
        return (node_feat, adj, target_energy, initial_energy_tensor, 
                target_size, energy_num, time_steps, inner_idx)  # Add inner_idx(ham_idx)


# -------------------------- 3. Graph Data Batching Tool (with Initial Values) --------------------------
def collate_graphs(batch):
    (node_feats, adjs, targets, initial_energies, 
     sizes, energy_nums, time_steps, ham_indices) = zip(*batch)
    
    data_list = []
    for nf, adj, t in zip(node_feats, adjs, targets):
        edges = adj.nonzero().t()
        edge_weights = adj[edges[0], edges[1]].view(-1, 1)  # Adjust edge weights to [num_edges, 1]
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


# -------------------------- 4. TransformerGNN Model (Using Initial Values, Supporting Feature Extraction) --------------------------
class EnergyPredTransformerGNN(nn.Module):
    """TransformerGNN model using initial energy to assist prediction, supporting intermediate feature return"""
    def __init__(self, input_dim, hidden_dim, output_dim, config, heads=4):
        super().__init__()
        self.config = config
        self.heads = heads  # Number of multi-head attention heads
        self.single_head_dim = hidden_dim  # Single attention head dimension
        
        # TransformerConv layers (explicitly set edge feature dimension to 1)
        self.transformer1 = TransformerConv(
            in_channels=input_dim,
            out_channels=self.single_head_dim,
            heads=self.heads,
            edge_dim=1
        )
        self.transformer2 = TransformerConv(
            in_channels=self.heads * self.single_head_dim,
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
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        self.bn2 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        self.bn3 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        
        # Initial value fusion layer - explicitly set input dimension to 1
        self.fc_initial = nn.Linear(1, self.heads * self.single_head_dim)
        
        # Final prediction layer
        self.fc = nn.Sequential(
            nn.Linear(2 * self.heads * self.single_head_dim, self.heads * self.single_head_dim),  # Fuse graph features and initial values
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(self.heads * self.single_head_dim, output_dim)
        )

    def forward(self, batch_data, initial_energies, return_features=False):
        """Forward propagation: graph data + initial values → predicted energy curve, supporting intermediate feature return"""
        x, edge_index, edge_weight, batch = (
            batch_data.x, 
            batch_data.edge_index, 
            batch_data.edge_weight,
            batch_data.batch
        )
        
        # Transformer layers
        x = self.transformer1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.transformer2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.transformer3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling - output shape: [batch_size, heads*single_head_dim]
        graph_features = global_mean_pool(x, batch)
        
        # Process initial energy - ensure input is 2D tensor [batch_size, 1]
        initial_energies = initial_energies.view(-1, 1)  # Ensure shape is [batch_size, 1]
        initial_features = self.fc_initial(initial_energies)  # Output shape: [batch_size, heads*single_head_dim]
        
        # Fuse features
        combined_features = torch.cat([graph_features, initial_features], dim=1)
        
        # Predict output
        output = self.fc(combined_features)
        
        # Support returning fused features for encoding
        if return_features:
            return output, combined_features
        return output


# -------------------------- 5. Evaluation and Visualization Tool Functions --------------------------
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R² score calculation"""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    r2_list = []
    for t, p in zip(y_true, y_pred):
        mean_t = np.mean(t)
        ss_total = np.sum((t - mean_t) ** 2)
        ss_residual = np.sum((t - p) ** 2)
        r2 = 1 - (ss_residual / (ss_total + 1e-8))  # Avoid division by zero
        r2_list.append(r2)
    
    return np.mean(r2_list) if r2_list else 0.0


def mean_energy_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean sequence R² calculation"""
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_pred, axis=0)
    
    mean_t = np.mean(y_true_mean)
    ss_total = np.sum((y_true_mean - mean_t) ** 2)
    ss_residual = np.sum((y_true_mean - y_pred_mean) ** 2)
    return 1 - (ss_residual / (ss_total + 1e-8)) if ss_total > 0 else 0.0


def save_energy_pred_to_csv(y_true_mean, y_pred_mean, true_fit, pred_fit, 
                           size, energy_num, time_steps, dataset_type, 
                           save_dir, num_samples):
    """Save prediction results to CSV, including initial value consistency information"""
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


def plot_mean_energy_prediction(y_true_mean, y_pred_mean, true_fit, pred_fit,
                               size, energy_num, time_steps, dataset_type, 
                               save_dir, num_samples):
    """Visualize prediction results, add initial value consistency information"""
    avg_r2 = mean_energy_r2(np.array([y_true_mean]), np.array([y_pred_mean]))
    plt.figure(figsize=(12, 7))
    
    # Plot curves
    plt.plot(time_steps, y_true_mean, label="True Energy Curve", color="#2E86AB", linewidth=3.0)
    plt.plot(time_steps, y_pred_mean, label="Averaged Predicted Curve", color="#A23B72", linewidth=3.0)
    
    # Plot Gaussian fits
    #plt.plot(time_steps, true_fit["y_fit"], label="True Gaussian Fit", color="#2E86AB", linewidth=2.5, linestyle=':')
    #plt.plot(time_steps, pred_fit["y_fit"], label="Predicted Gaussian Fit", color="#A23B72", linewidth=2.5, linestyle=':')
    
    # Title and labels
    plt.title(
        f"Energy Prediction - Energy_num={energy_num} (Size={size}×{size})\n"
        f"{dataset_type.upper()} | Mean Samples={num_samples} | R²={avg_r2:.4f}\n"
        f"True τ={true_fit['tau_D']:.2f} fs | Pred τ={pred_fit['tau_D']:.2f} fs\n"
        f"Initial Value Consistency: True={true_fit['initial_diff']:.2e}, Pred={pred_fit['initial_diff']:.2e}",
        fontsize=12, pad=15
    )
    plt.xlabel("Time (fs)", fontsize=10)
    plt.ylabel("Energy (eV)", fontsize=10)
    plt.legend(loc="best", fontsize=10)
    
    save_filename = f"{dataset_type}_energy{energy_num}_{size}x{size}_pred_with_fit.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return save_filename, save_path


def plot_training_history(history, save_path):
    """Training history visualization"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="#2E86AB", linewidth=2)
    plt.plot(history["test_loss"], label="Test Loss", color="#A23B72", linewidth=2)
    plt.title("MSE Loss", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_r2"], label="Train R²", color="#2E86AB", linewidth=2)
    plt.plot(history["test_r2"], label="Test R²", color="#A23B72", linewidth=2)
    plt.title("R² Score", fontsize=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("R²", fontsize=10)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history saved to: {save_path}")
    plt.close()


def predict_all_energies_with_mean(model, dataset, dataset_type, device, save_dir, num_samples_for_mean="all"):
    """Batch prediction and visualization, using improved fitting function to ensure initial value correspondence"""
    model.eval()
    random.seed(42)
    all_sizes = sorted(list(set([dataset[i][4] for i in range(len(dataset))])))
    total_plots = 0
    total_csvs = 0
    
    print(f"\n=== {dataset_type.upper()} Set Prediction ===")
    for size in all_sizes:
        size_indices = [i for i in range(len(dataset)) if dataset[i][4] == size]
        if not size_indices:
            print(f"⚠️ Skipped size={size}×{size} (no data)")
            continue
        
        selected_indices = size_indices if num_samples_for_mean == "all" else random.sample(size_indices, min(num_samples_for_mean, len(size_indices)))
        num_samples = len(selected_indices)
        if num_samples < 1:
            print(f"⚠️ Skipped size={size}×{size} (insufficient samples)")
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
                
                # Model prediction (pass initial value)
                edges = adj.nonzero().t()
                edge_weights = adj[edges[0], edges[1]].view(-1, 1)
                data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
                batch_data = Batch.from_data_list([data])
                initial_energy = initial_energy.to(device)
                
                pred = model(batch_data, initial_energy).cpu().numpy().flatten()
                all_preds.append(pred)
        
        # Calculate average prediction
        pred_mean = np.mean(np.array(all_preds), axis=0)
        
        # Gaussian fitting - force fitted curve to pass through initial value point
        true_fit_result = fit_gaussian_decay(time_steps, true_curve, enforce_initial=True)
        pred_fit_result = fit_gaussian_decay(time_steps, pred_mean, enforce_initial=True)
        
        # Check and report initial value consistency
        if not np.isclose(true_fit_result['initial_diff'], 0, atol=1e-6):
            print(f"⚠️ Warning: True curve fit initial value mismatch, difference={true_fit_result['initial_diff']:.2e}")
        if not np.isclose(pred_fit_result['initial_diff'], 0, atol=1e-6):
            print(f"⚠️ Warning: Predicted curve fit initial value mismatch, difference={pred_fit_result['initial_diff']:.2e}")
        
        # Save visualization
        plot_name, _ = plot_mean_energy_prediction(
            y_true_mean=true_curve,
            y_pred_mean=pred_mean,
            true_fit=true_fit_result,
            pred_fit=pred_fit_result,
            size=size,
            energy_num=energy_num,
            time_steps=time_steps,
            dataset_type=dataset_type,
            save_dir=save_dir,
            num_samples=num_samples
        )
        
        # Save CSV
        csv_name, _ = save_energy_pred_to_csv(
            y_true_mean=true_curve,
            y_pred_mean=pred_mean,
            true_fit=true_fit_result,
            pred_fit=pred_fit_result,
            size=size,
            energy_num=energy_num,
            time_steps=time_steps,
            dataset_type=dataset_type,
            save_dir=save_dir,
            num_samples=num_samples
        )
        
        print(f"✅ Processed size={size}×{size} → mean R²={mean_energy_r2(np.array([true_curve]), np.array([pred_mean])):.4f}")
        print(f"   - Fitted lifetime: True τ={true_fit_result['tau_D']:.2f}fs | Pred τ={pred_fit_result['tau_D']:.2f}fs")
        print(f"   - Initial value consistency: True difference={true_fit_result['initial_diff']:.2e} | Pred difference={pred_fit_result['initial_diff']:.2e}")
        print(f"   - Image: {plot_name}")
        print(f"   - CSV: {csv_name}")
        total_plots += 1
        total_csvs += 1
    
    print(f"\n=== {dataset_type.upper()} Set Prediction Completed ===")
    print(f"Generated images: {total_plots} | Generated CSVs: {total_csvs}")
    return total_plots, total_csvs


# -------------------------- 6. Average R² Result Summary Tool Function (One Result per Image) --------------------------
def collect_mean_r2_results(dataset, dataset_type, model, device, save_dir):
    """
    Collect average R² values for each size (corresponding to one image) in specified dataset (train/test)
    Only keep one result per size, corresponding to generated image
    """
    model.eval()
    r2_records = []  # Store detailed information of each R² record
    all_sizes = sorted(list(set([dataset[i][4] for i in range(len(dataset))])))  # Deduplicate and sort all sizes
    
    print(f"\n=== Start collecting {dataset_type.upper()} set average R² results ===")
    for size in all_sizes:
        # Filter all sample indices of current size
        size_indices = [i for i in range(len(dataset)) if dataset[i][4] == size]
        if not size_indices:
            print(f"⚠️ Skipped size={size}×{size} (no data)")
            continue
        
        # Get energy index and time step information of current size
        sample_idx = size_indices[0]
        _, _, _, _, _, energy_num, time_steps, _ = dataset[sample_idx]
        
        # Calculate R² of all samples in current size and take average
        size_r2_list = []
        with torch.no_grad():
            for idx in size_indices:
                # Extract sample data
                node_feat, adj, target_energy, initial_energy, _, _, _, _ = dataset[idx]
                
                # Model prediction
                edges = adj.nonzero().t()
                edge_weights = adj[edges[0], edges[1]].view(-1, 1)
                data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
                batch_data = Batch.from_data_list([data])
                initial_energy = initial_energy.to(device)
                
                pred = model(batch_data, initial_energy).cpu().numpy().flatten()
                true = target_energy.numpy().flatten()
                
                # Calculate R² of single sample
                mean_true = np.mean(true)
                ss_total = np.sum((true - mean_true) ** 2)
                ss_residual = np.sum((true - pred) ** 2)
                single_r2 = 1 - (ss_residual / (ss_total + 1e-8))  # Avoid division by zero
                size_r2_list.append(single_r2)
        
        # Calculate average R² of current size (average performance corresponding to one image)
        size_avg_r2 = np.mean(size_r2_list)
        size_std_r2 = np.std(size_r2_list)  # Calculate standard deviation to reflect performance fluctuation
        
        # Record information of current size (one image per size)
        r2_records.append({
            "dataset_type": dataset_type.upper(),  # Dataset type (TRAIN/TEST)
            "size": size,                          # Size (e.g., 30 → 30×30)
            "size_index": f"{size}×{size}",        # Size index (formatted display)
            "energy_num": energy_num,              # Energy file index
            "sample_count": len(size_indices),     # Number of samples under this size
            "mean_r2": round(size_avg_r2, 6),      # Average R² (keep 6 decimal places)
            "std_r2": round(size_std_r2, 6),       # R² standard deviation
            "ranking_key": size_avg_r2             # Key for sorting
        })
    
    # Convert to DataFrame and sort (descending by average R², ascending by size for same R²)
    r2_df = pd.DataFrame(r2_records)
    r2_df_sorted = r2_df.sort_values(
        by=["ranking_key", "size"],  # First by average R² descending, then by size ascending
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # Add ranking column after sorting
    r2_df_sorted["ranking"] = range(1, len(r2_df_sorted) + 1)
    
    # Adjust column order
    final_columns = [
        "ranking", "dataset_type", "size", "size_index", "energy_num",
        "sample_count", "mean_r2", "std_r2"
    ]
    r2_df_sorted = r2_df_sorted[final_columns]
    
    # Save as CSV file
    csv_save_dir = os.path.join(save_dir, "mean_r2_summary")
    os.makedirs(csv_save_dir, exist_ok=True)
    csv_filename = f"{dataset_type}_mean_r2_summary_sorted.csv"
    csv_save_path = os.path.join(csv_save_dir, csv_filename)
    r2_df_sorted.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    
    # Calculate and print overall average R²
    overall_mean_r2 = r2_df_sorted["mean_r2"].mean()
    print(f"✅ {dataset_type.upper()} set average R² summary completed:")
    print(f"   - Number of images: {len(r2_df_sorted)} (one image per size)")
    print(f"   - Overall average R²: {round(overall_mean_r2, 6)}")
    print(f"   - Save path: {csv_save_path}")
    print(f"   - Sorting rule: Descending by average R² → Ascending by size")
    
    return r2_df_sorted, csv_save_path


# -------------------------- 7. Average Encoded Feature Extraction and Saving Module (Only Keep This Function) --------------------------
def extract_mean_encoded_features(model, dataset, device, dataset_type, save_dir):
    """
    Only extract and save **average encoded features**:
    1. Extract model intermediate layer encoded features of each sample
    2. Group by energy index, calculate average encoded features of each group and save
    """
    model.eval()
    energy_groups = {}  # Store encoded features grouped by energy index
    mean_enc_dir = os.path.join(save_dir, "mean_encoded_features")
    os.makedirs(mean_enc_dir, exist_ok=True)  # Only create average encoded feature directory
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            # Get sample data
            (node_feat, adj, _, initial_energy, 
             size, energy_num, _, ham_idx) = dataset[idx]
            
            # Extract model encoded features (intermediate layer fused features)
            edges = adj.nonzero().t()
            edge_weights = adj[edges[0], edges[1]].view(-1, 1)
            data = Data(x=node_feat, edge_index=edges, edge_weight=edge_weights).to(device)
            batch_data = Batch.from_data_list([data])
            init_energy_tensor = initial_energy.to(device)
            
            _, enc_feat = model(batch_data, init_energy_tensor, return_features=True)
            enc_feat_np = enc_feat.cpu().numpy().flatten()
            
            # Group by energy index for subsequent mean calculation
            if energy_num not in energy_groups:
                energy_groups[energy_num] = {
                    "encoded_features": [],  # Store encoded features of all samples under this energy index
                    "size": size,
                    "count": 0,
                    "initial_energies": []  # Store initial energies for mean calculation
                }
            energy_groups[energy_num]["encoded_features"].append(enc_feat_np)
            energy_groups[energy_num]["initial_energies"].append(initial_energy.item())
            energy_groups[energy_num]["count"] += 1
            
            # Progress prompt
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples...")
    
    # Calculate and save average encoded features of each energy index
    mean_encoded_features = []
    for energy_num, group in energy_groups.items():
        size = group["size"]
        count = group["count"]
        enc_feats_array = np.array(group["encoded_features"])  # Shape: [number of samples, feature dimension]
        
        # Calculate mean of encoded features (average by feature dimension)
        enc_mean = np.mean(enc_feats_array, axis=0)
        # Calculate mean of initial energies
        initial_energy_mean = np.mean(group["initial_energies"])
        
        # Build average encoded feature dictionary
        enc_mean_dict = {
            "energy_num": energy_num,
            "size": size,
            "size_index": f"{size}×{size}",
            "sample_count": count,  # Number of samples under this energy index
            "dataset_type": dataset_type,
            "initial_energy_mean_eV": initial_energy_mean  # Mean initial energy (auxiliary information)
        }
        # Add average encoded features of each dimension
        for feat_idx, feat_val in enumerate(enc_mean):
            enc_mean_dict[f"mean_encoded_feat_{feat_idx}"] = feat_val
        
        mean_encoded_features.append(enc_mean_dict)
    
    # Convert to DataFrame and save
    mean_enc_df = pd.DataFrame(mean_encoded_features)
    mean_enc_path = os.path.join(mean_enc_dir, f"{dataset_type}_mean_encoded_features.csv")
    mean_enc_df.to_csv(mean_enc_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ {dataset_type} average encoded features saved successfully:")
    print(f"   - Number of energy indices: {len(mean_enc_df)}")
    print(f"   - Feature dimension: {len([col for col in mean_enc_df.columns if col.startswith('mean_encoded_feat_')])}")
    print(f"   - Save path: {mean_enc_path}")
    
    return mean_enc_df, mean_enc_path


# -------------------------- 8. Main Process Function --------------------------
def regenerate_graph_dataset(save_initial=True, train_energy_count=10):
    """Generate dataset (ensure initial value corresponds to curve)"""
    config = {
        "res": "./res",  # Hamiltonian directory
        "res_target_pop": "./res_target_pop",  # Energy data directory
        "ham_count_per_group": 1000,  # Number of Hamiltonians per size group
        "save_dir": "./pop_shi_mo_data",  # Data save directory
        "train_energy_count": train_energy_count,  # Number of training set energy indices (customizable)
        "random_state": 42  # Random seed
    }
    
    # Create directories
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["res"], exist_ok=True)
    os.makedirs(config["res_target_pop"], exist_ok=True)
    
    # Set random seed
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    
    # Initialize dataset
    full_dataset = GraphEnergyDataset(
        config=config, 
        ham_count_per_group=config["ham_count_per_group"],
        load_initial_from_csv=not save_initial,
        initial_csv_path=os.path.join(config["save_dir"], "energy_initial_values.csv") if not save_initial else None
    )
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty, please check file paths")
    
    # Split training and test sets by energy index (energy_num)
    # 1. Collect all valid energy indices
    all_energy_nums = []
    for size in full_dataset.valid_sizes:
        en_num = full_dataset.size_to_energy[size]["energy_num"]
        all_energy_nums.append(en_num)
    all_energy_nums = sorted(list(set(all_energy_nums)))  # Deduplicate and sort
    
    if len(all_energy_nums) < config["train_energy_count"]:
        raise ValueError(f"Insufficient available energy indices! Found {len(all_energy_nums)}, need {config['train_energy_count']} for training set")
    
    # 2. Ensure minimum and maximum indices are included, then select remaining from middle evenly
    min_en = all_energy_nums[0]  # Minimum energy index
    max_en = all_energy_nums[-1]  # Maximum energy index
    
    # Must include minimum and maximum indices
    train_energy_indices = [min_en, max_en]
    
    # Select from remaining indices
    remaining_ens = [en for en in all_energy_nums if en not in train_energy_indices]
    
    # Number needed to supplement
    need = config["train_energy_count"] - 2
    if need > 0:
        if len(remaining_ens) < need:
            raise ValueError(f"Insufficient remaining energy indices! Need to supplement {need}, but only found {len(remaining_ens)}")
        
        # Sample remaining indices evenly
        step = max(1, len(remaining_ens) // need)
        additional_ens = remaining_ens[::step][:need]
        
        # If insufficient samples, randomly supplement from remaining
        if len(additional_ens) < need:
            remaining_after = [en for en in remaining_ens if en not in additional_ens]
            additional_ens += random.sample(remaining_after, need - len(additional_ens))
        
        train_energy_indices.extend(additional_ens)
        train_energy_indices = sorted(list(set(train_energy_indices)))  # Deduplicate and sort
    
    # 3. Test set is remaining energy indices
    test_energy_indices = [en for en in all_energy_nums if en not in train_energy_indices]
    
    print(f"\n=== Dataset Split ===")
    print(f"All energy indices: {all_energy_nums}")
    print(f"Minimum energy index: {min_en}, Maximum energy index: {max_en}")
    print(f"Training set energy indices ({len(train_energy_indices)}): {sorted(train_energy_indices)}")
    print(f"Test set energy indices ({len(test_energy_indices)}): {sorted(test_energy_indices)}")
    
    # 4. Build global indices for training and test sets
    train_indices = []
    test_indices = []
    current_idx = 0
    
    for size in full_dataset.valid_sizes:
        en_num = full_dataset.size_to_energy[size]["energy_num"]
        count = len(full_dataset.energy_data[size])
        
        # Determine if current energy index belongs to training or test set
        if en_num in train_energy_indices:
            train_indices.extend(range(current_idx, current_idx + count))
        else:
            test_indices.extend(range(current_idx, current_idx + count))
            
        current_idx += count
    
    # Create training and test sets
    train_data = [full_dataset[i] for i in train_indices]
    test_data = [full_dataset[i] for i in test_indices]
    
    # Save dataset
    try:
        with open(os.path.join(config["save_dir"], "train_graph_energy_dataset.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        with open(os.path.join(config["save_dir"], "test_graph_energy_dataset.pkl"), "wb") as f:
            pickle.dump(test_data, f)
        
        # Save parameters
        data_params = {
            "energy_curve_len": full_dataset.energy_curve_len,
            "output_dim": full_dataset.output_dim,
            "train_energy_indices": sorted(train_energy_indices),
            "test_energy_indices": sorted(test_energy_indices),
            "node_feat_dim": 2,
            "ham_count_per_group": config["ham_count_per_group"],
            "initial_csv_path": full_dataset.initial_save_csv
        }
        with open(os.path.join(config["save_dir"], "graph_energy_data_params.pkl"), "wb") as f:
            pickle.dump(data_params, f)
        
        print(f"\n✅ Dataset saved successfully:")
        print(f"  - Training set: {len(train_data)} samples | Test set: {len(test_data)} samples")
        print(f"  - Initial energy file: {data_params['initial_csv_path']}")
        print(f"  - Save path: {config['save_dir']}")
    
    except Exception as e:
        print(f"❌ Failed to save dataset: {str(e)}")
        raise


def main_training(train_energy_count=10):
    """Train model and predict (ensure initial value corresponds to curve, only keep average encoded feature extraction)"""
    config = {
        "processed_data_dir": "./pop_shi_mo_data",  # Dataset directory
        "model_save_dir": "./pop_shi_mo_models",    # Model directory
        "prediction_save_dir": "./pop_Si_shi_mo_predictions",  # Prediction result directory
        "features_save_dir": "./pop_Si_shi_mo_features",  # Feature save directory (only save average encoded features)
        "batch_size": 32,          # Batch size
        "hidden_dim": 32,          # Hidden layer dimension
        "dropout": 0.2,            # Dropout
        "learning_rate": 3e-4,     # Learning rate
        "weight_decay": 1e-5,      # Weight decay
        "epochs": 100,              # Training epochs
        "print_interval": 1,       # Print interval
        "random_state": 42,        # Random seed
        "transformer_heads": 6,    # Number of multi-head attention heads
        "ham_count_per_group": 1000  # Consistent with dataset
    }
    
    # Set random seed
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    torch.manual_seed(config["random_state"])
    torch.cuda.manual_seed_all(config["random_state"])
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config["processed_data_dir"], exist_ok=True)
    os.makedirs(config["model_save_dir"], exist_ok=True)
    os.makedirs(config["prediction_save_dir"], exist_ok=True)
    os.makedirs(config["features_save_dir"], exist_ok=True)  # Only create feature root directory
    train_pred_dir = os.path.join(config["prediction_save_dir"], "train")
    test_pred_dir = os.path.join(config["prediction_save_dir"], "test")
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
        file_path = os.path.join(config["processed_data_dir"], file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}\nPlease run regenerate_graph_dataset() first")
    
    with open(os.path.join(config["processed_data_dir"], "train_graph_energy_dataset.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(config["processed_data_dir"], "test_graph_energy_dataset.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(config["processed_data_dir"], "graph_energy_data_params.pkl"), "rb") as f:
        data_params = pickle.load(f)
    
    print(f"Training set energy indices: {data_params['train_energy_indices']}")
    print(f"Test set energy indices: {data_params['test_energy_indices']}")
    
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
    
    # Create data loaders
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
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = EnergyPredTransformerGNN(
        input_dim=data_params["node_feat_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=data_params["output_dim"],
        config=config,
        heads=config["transformer_heads"]
    ).to(device)
    model.config = config  # Save configuration for feature extraction
    print(model)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )
    
    # Training process
    print("\n=== Starting Training ===")
    history = {
        "train_loss": [], "train_r2": [],
        "test_loss": [], "test_r2": []
    }
    best_test_loss = float("inf")
    best_model_path = os.path.join(config["model_save_dir"], "best_model.pth")
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_total_r2 = 0.0
        
        for batch in train_loader:
            batch_data, targets, initial_energies, _, _, _, _ = batch
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            initial_energies = initial_energies.to(device)  # Pass initial values
            
            optimizer.zero_grad()
            preds = model(batch_data, initial_energies)  # Model uses initial values
            loss = F.mse_loss(preds, targets)
            batch_r2 = r2_score(targets, preds)
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item() * preds.size(0)
            train_total_r2 += batch_r2 * preds.size(0)
        
        avg_train_loss = train_total_loss / len(train_dataset)
        avg_train_r2 = train_total_r2 / len(train_dataset)
        history["train_loss"].append(avg_train_loss)
        history["train_r2"].append(avg_train_r2)
        
        # Test phase
        model.eval()
        test_total_loss = 0.0
        test_total_r2 = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_data, targets, initial_energies, _, _, _, _ = batch
                batch_data = batch_data.to(device)
                targets = targets.to(device)
                initial_energies = initial_energies.to(device)
                
                preds = model(batch_data, initial_energies)
                loss = F.mse_loss(preds, targets)
                batch_r2 = r2_score(targets, preds)
                
                test_total_loss += loss.item() * preds.size(0)
                test_total_r2 += batch_r2 * preds.size(0)
        
        avg_test_loss = test_total_loss / len(test_dataset)
        avg_test_r2 = test_total_r2 / len(test_dataset)
        history["test_loss"].append(avg_test_loss)
        history["test_r2"].append(avg_test_r2)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"📌 Epoch {epoch+1}: Saved best model (test loss: {best_test_loss:.6f})")
        
        # Print progress
        if (epoch + 1) % config["print_interval"] == 0:
            print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
            print(f"  Training set: Loss={avg_train_loss:.6f}, R²={avg_train_r2:.4f}")
            print(f"  Test set: Loss={avg_test_loss:.6f}, R²={avg_test_r2:.4f}")
    
    # Plot training history
    print("\n=== Plotting Training History ===")
    history_path = os.path.join(config["model_save_dir"], "training_history.png")
    plot_training_history(history, history_path)
    
    # Batch prediction
    print("\n=== Batch Prediction ===")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Training set prediction
    print("\n--- Training Set Prediction ---")
    train_plots, train_csvs = predict_all_energies_with_mean(
        model=model,
        dataset=train_dataset,
        dataset_type="train",
        device=device,
        save_dir=train_pred_dir
    )
    
    # Test set prediction
    print("\n--- Test Set Prediction ---")
    test_plots, test_csvs = predict_all_energies_with_mean(
        model=model,
        dataset=test_dataset,
        dataset_type="test",
        device=device,
        save_dir=test_pred_dir
    )
    
    # Collect average R² results for each image
    print("\n=== Average R² Results Summary ===")
    # 1. Training set average R² summary (one result per image)
    train_r2_df, train_r2_csv = collect_mean_r2_results(
        dataset=train_dataset,
        dataset_type="train",
        model=model,
        device=device,
        save_dir=config["prediction_save_dir"]
    )
    
    # 2. Test set average R² summary (one result per image)
    test_r2_df, test_r2_csv = collect_mean_r2_results(
        dataset=test_dataset,
        dataset_type="test",
        model=model,
        device=device,
        save_dir=config["prediction_save_dir"]
    )
    
    # -------------------------- Only extract and save average encoded features --------------------------
    print("\n=== Extracting and Saving Average Encoded Features ===")
    # 1. Training set average encoded feature extraction
    print("\n--- Training Set Average Encoded Feature Extraction ---")
    train_mean_enc_df, train_mean_enc_path = extract_mean_encoded_features(
        model=model,
        dataset=train_dataset,
        device=device,
        dataset_type="train",
        save_dir=config["features_save_dir"]
    )
    
    # 2. Test set average encoded feature extraction
    print("\n--- Test Set Average Encoded Feature Extraction ---")
    test_mean_enc_df, test_mean_enc_path = extract_mean_encoded_features(
        model=model,
        dataset=test_dataset,
        device=device,
        dataset_type="test",
        save_dir=config["features_save_dir"]
    )
    
    print(f"\n🎉 All tasks completed!")
    print(f"1. Best model: {best_model_path}")
    print(f"2. Training history: {history_path}")
    print(f"3. Training set images: {train_plots}, average R²: {round(train_r2_df['mean_r2'].mean(), 6)}")
    print(f"4. Test set images: {test_plots}, average R²: {round(test_r2_df['mean_r2'].mean(), 6)}")
    print(f"5. Average encoded features save directory: {config['features_save_dir']}")
    print(f"   - Training set average encoded features: {train_mean_enc_path} ({len(train_mean_enc_df)} energy indices)")
    print(f"   - Test set average encoded features: {test_mean_enc_path} ({len(test_mean_enc_df)} energy indices)")


# -------------------------- Main Program Entry --------------------------
if __name__ == "__main__":
    # Can customize number of training set energy indices (ensure ≥2)
    custom_train_count = 7  # Example: 10 training set indices (including min and max)
    
    # Step 1: Generate dataset (save_initial=True to save initial energy for first run, can set to False to load from CSV later)
    regenerate_graph_dataset(save_initial=True, train_energy_count=custom_train_count)
    
    # Step 2: Train model and predict (only keep average encoded feature extraction)
    main_training(train_energy_count=custom_train_count)