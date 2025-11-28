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


# -------------------------- 1. Gaussian Curve Fitting Tools --------------------------
def gaussian_decay(t, A, tau_D, t0=0):
    return A * np.exp(-0.5 * ((t - t0) / tau_D) ** 2)

def constrained_gaussian_decay(t, tau_D):
    global initial_value
    A = initial_value
    return A * np.exp(-0.5 * (t / tau_D) ** 2)

def fit_gaussian_decay(x_data, y_data, enforce_initial=True):
    global initial_value
    initial_value = y_data[0] if len(y_data) > 0 else np.nan
    
    try:
        if enforce_initial:
            initial_guess = [len(x_data) / 4]
            params, _ = curve_fit(
                constrained_gaussian_decay, 
                x_data, 
                y_data, 
                p0=initial_guess,
                maxfev=10000
            )
            tau_D = params[0]
            A = initial_value
        else:
            initial_guess = [initial_value, len(x_data) / 4]
            params, _ = curve_fit(
                gaussian_decay, 
                x_data, 
                y_data, 
                p0=initial_guess,
                maxfev=10000
            )
            A, tau_D = params
        
        y_fit = constrained_gaussian_decay(x_data, tau_D) if enforce_initial else gaussian_decay(x_data, A, tau_D)
        fit_initial = y_fit[0]
        original_initial = y_data[0] if len(y_data) > 0 else np.nan
        initial_diff = abs(fit_initial - original_initial)
        
        return {
            'A': A, 'tau_D': tau_D, 'y_fit': y_fit, 'success': True,
            'initial_value': original_initial, 'fit_initial_value': fit_initial, 'initial_diff': initial_diff
        }
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        return {
            'A': np.nan, 'tau_D': np.nan, 'y_fit': np.zeros_like(x_data), 'success': False,
            'initial_value': y_data[0] if len(y_data) > 0 else np.nan,
            'fit_initial_value': np.nan, 'initial_diff': np.nan
        }


# -------------------------- 2. Graph Structure Dataset Class (Core Modification: Bottom-Right Submatrix Cropping) --------------------------
class GraphEnergyDataset(Dataset):
    def __init__(self, config, ham_count_per_group=400, load_initial_from_csv=False, initial_csv_path=None):
        self.config = config
        self.ham_count_per_group = ham_count_per_group
        self.load_initial_from_csv = load_initial_from_csv
        self.initial_csv_path = initial_csv_path
        self.initial_save_csv = os.path.join(config["save_dir"], "energy_initial_values.csv")
        
        # Key modification: Predefine large matrix size to ensure energy data access during loading
        self.full_matrix_size = 152  # Fixed large matrix size as 87×87
        
        # Load energy data (now full_matrix_size is defined)
        if self.load_initial_from_csv:
            if not initial_csv_path or not os.path.exists(initial_csv_path):
                raise ValueError(f"Failed to load initial values: File does not exist → {initial_csv_path}")
            self.energy_data, self.size_to_energy, self.sample_to_initial = self._load_energy_from_initial_csv()
        else:
            self.energy_data, self.size_to_energy, self.sample_to_initial = self._load_energy_and_save_initial_csv()
        
        self.valid_sizes = sorted(self.size_to_energy.keys())
        self.max_size = max(self.valid_sizes) if self.valid_sizes else 0
        
        # Load Hamiltonians and build graphs
        self.shared_full_hams = self._load_shared_full_hamiltonians()
        self.graph_groups = self._build_graph_groups_by_energy_num()
        
        # Output dimension calculation
        if self.valid_sizes:
            sample_size = self.valid_sizes[0]
            self.energy_curve_len = len(self.energy_data[sample_size][0])
            for size in self.valid_sizes:
                for curve in self.energy_data[size]:
                    if len(curve) != self.energy_curve_len:
                        raise ValueError(f"Energy curve length mismatch! Size={size} has curve length {len(curve)}, standard is {self.energy_curve_len}")
        else:
            self.energy_curve_len = 0
        self.output_dim = self.energy_curve_len
        
        # Data validation
        self._validate_data()
        print(f"✅ Dataset initialization completed:")
        print(f"  - Large matrix size: {self.full_matrix_size}×{self.full_matrix_size}")
        print(f"  - Maximum submatrix size: {self.max_size}×{self.max_size}")
        print(f"  - Output dimension (Energy curve length): {self.output_dim}")
        print(f"  - Valid submatrix size list: {self.valid_sizes}")
        print(f"  - Initial energy file: {self.initial_save_csv}" if not load_initial_from_csv else f"  - Loading initial values from: {initial_csv_path}")

    def _save_initial_to_csv(self, initial_data):
        df = pd.DataFrame(initial_data)
        df.to_csv(self.initial_save_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Initial energy saved successfully: {self.initial_save_csv} ({len(df)} samples total)")
        return self.initial_save_csv

    def _load_energy_and_save_initial_csv(self):
        """Core modification: size = 87 - en_num + 1 (submatrix size), en_num corresponds to X (jj-enX.dat)"""
        energy_data = {}  # Key: submatrix size (87-X+1), Value: list of energy curves
        size_to_energy = {}  # Key: submatrix size, Value: metadata (including en_num)
        sample_to_initial = {}  # Key: (size, ham_idx), Value: initial energy
        initial_data_list = []
        start_en_num = 1  # Start from jj-en2.dat (adjust as needed)
        end_en_num = 120   # End at jj-en87.dat
        global_sample_idx = 0
        
        for en_num in range(start_en_num, end_en_num + 1):
            # Calculate submatrix size: 87 - X + 1 (X = en_num)
            size = self.full_matrix_size - en_num + 1
            if size <= 0:
                print(f"⚠️ Skipping en_num={en_num}: Submatrix size {size} (invalid)")
                continue
            
            en_filename = f"jj-en{en_num}.dat"
            en_filepath = os.path.join(self.config["res_target_energy"], en_filename)
            
            try:
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                base_energy_curve = en_raw[:, 1]
                base_initial_energy = base_energy_curve[0]
                
                # Generate multiple samples (add noise)
                size_curves = []
                for ham_idx in range(self.ham_count_per_group):
                    noise = np.random.normal(0, 0.001, size=len(base_energy_curve))
                    single_curve = (base_energy_curve + noise).astype(np.float32)
                    size_curves.append(single_curve)
                    
                    sample_initial = single_curve[0]
                    sample_key = (size, ham_idx)
                    sample_to_initial[sample_key] = sample_initial
                    
                    initial_data_list.append({
                        "global_sample_idx": global_sample_idx,
                        "size": size,
                        "size_str": f"{size}×{size}",
                        "energy_num": en_num,  # Record corresponding X (jj-enX.dat)
                        "ham_idx": ham_idx,
                        "initial_energy_eV": sample_initial,
                        "curve_first_point_eV": single_curve[0]
                    })
                    global_sample_idx += 1
                
                energy_data[size] = size_curves
                size_to_energy[size] = {
                    "energy_num": en_num,  # Save X value for subsequent submatrix cropping
                    "filename": en_filename,
                    "time_steps": time_steps,
                    "curve_count": len(size_curves)
                }
                print(f"✅ Processed: en_num={en_num} → Submatrix size {size}×{size} ({len(size_curves)} samples, initial energy={base_initial_energy:.6f}eV)")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath}")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load {en_filename}: {str(e)} (skipped)")
                continue
        
        if initial_data_list:
            self._save_initial_to_csv(initial_data_list)
        
        return energy_data, size_to_energy, sample_to_initial

    def _load_energy_from_initial_csv(self):
        energy_data = {}
        size_to_energy = {}
        sample_to_initial = {}
        
        df = pd.read_csv(self.initial_csv_path)
        required_cols = ["size", "ham_idx", "energy_num", "initial_energy_eV", "curve_first_point_eV"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Initial energy CSV missing required columns! Must contain: {required_cols}")
        
        size_groups = df.groupby("size")
        for size, size_df in size_groups:
            size = int(size)
            energy_num = size_df["energy_num"].iloc[0]  # Get X value
            en_filename = f"jj-en{energy_num}.dat"
            en_filepath = os.path.join(self.config["res_target_energy"], en_filename)
            
            try:
                en_raw = np.loadtxt(en_filepath).astype(np.float32)
                time_steps = en_raw[:, 0]
                base_energy_curve = en_raw[:, 1]
                
                size_curves = []
                for _, row in size_df.iterrows():
                    ham_idx = int(row["ham_idx"])
                    target_initial = float(row["initial_energy_eV"])
                    curve_first_point = float(row["curve_first_point_eV"])
                    
                    if not np.isclose(target_initial, curve_first_point, atol=1e-6):
                        print(f"⚠️ Warning: Sample (size={size}, ham_idx={ham_idx}) initial value does not match curve first point")
                    
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
                print(f"✅ Loaded: size={size}×{size} (en_num={energy_num}, {len(size_curves)} samples)")
            
            except FileNotFoundError:
                print(f"⚠️ Warning: Energy file missing: {en_filepath} (skipped size={size})")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load size={size}: {str(e)} (skipped)")
                continue
        
        return energy_data, size_to_energy, sample_to_initial

    def _load_shared_full_hamiltonians(self):
        """Load 87×87 large matrix Hamiltonians"""
        shared_hams = []
        filename_template = "0_Ham_{idx}_{part}"
        hamiltonian_size = self.full_matrix_size  # Fixed as 87
        
        for idx in range(self.ham_count_per_group):
            re_path = os.path.join(self.config["res"], filename_template.format(idx=idx, part="re"))
            im_path = os.path.join(self.config["res"], filename_template.format(idx=idx, part="im"))
            
            try:
                re_data = np.loadtxt(re_path).astype(np.float32)
                im_data = np.loadtxt(im_path).astype(np.float32)
                re_mat = re_data.reshape(hamiltonian_size, hamiltonian_size)
                im_mat = im_data.reshape(hamiltonian_size, hamiltonian_size)
                
                if re_mat.shape != (hamiltonian_size, hamiltonian_size) or im_mat.shape != (hamiltonian_size, hamiltonian_size):
                    raise ValueError(f"Hamiltonian {idx} dimension error, should be {hamiltonian_size}×{hamiltonian_size}")
                
                shared_hams.append((re_mat, im_mat))
            
            except FileNotFoundError:
                raise FileNotFoundError(f"Hamiltonian {idx} missing: {re_path} or {im_path}")
            except ValueError as e:
                raise ValueError(f"Hamiltonian {idx} processing failed: {str(e)}")
        
        return shared_hams

    def _build_graph_from_hamiltonian(self, re_mat, im_mat):
        """Build graph structure from submatrix"""
        size = re_mat.shape[0]
        
        # Node features: real part mean + imaginary part mean
        node_feat = np.hstack([
            np.mean(re_mat, axis=1).reshape(-1, 1),
            np.mean(im_mat, axis=1).reshape(-1, 1)
        ]).astype(np.float32)
        
        # Edge weights: modulus (interaction strength)
        adj_matrix = np.sqrt(re_mat**2 + im_mat**2).astype(np.float32)
        
        return (
            torch.tensor(node_feat, dtype=torch.float32),
            torch.tensor(adj_matrix, dtype=torch.float32)
        )

    def _build_graph_groups_by_energy_num(self):
        """Core modification: Crop bottom-right submatrix (X~87 range) by en_num (X)"""
        graph_groups = {}
        for size in self.valid_sizes:
            # Get en_num (X value) corresponding to current size
            en_num = self.size_to_energy[size]["energy_num"]
            # Calculate 0-based start index corresponding to 1-based X (X-1)
            start_idx = en_num - 1  # Key: X=2 → start_idx=1 (0-based), corresponding to 1~86 (including 86)
            
            # Validate start index validity
            if start_idx < 0 or start_idx + size > self.full_matrix_size:
                raise ValueError(
                    f"Submatrix cropping error: en_num={en_num}, size={size}, "
                    f"start index={start_idx}, large matrix size={self.full_matrix_size}×{self.full_matrix_size}"
                )
            
            graph_list = []
            ham_count = min(len(self.shared_full_hams), len(self.energy_data[size]))
            for ham_idx in range(ham_count):
                full_re, full_im = self.shared_full_hams[ham_idx]
                # Crop bottom-right submatrix: [start_idx:, start_idx:] (0-based)
                sub_re = full_re[start_idx:, start_idx:]
                sub_im = full_im[start_idx:, start_idx:]
                
                # Validate cropped submatrix size correctness
                if sub_re.shape != (size, size) or sub_im.shape != (size, size):
                    raise ValueError(
                        f"Submatrix size error: Expected {size}×{size}, actual {sub_re.shape} (en_num={en_num}, ham_idx={ham_idx})"
                    )
                
                node_feat, adj = self._build_graph_from_hamiltonian(sub_re, sub_im)
                graph_list.append((node_feat, adj))
            graph_groups[size] = graph_list
        return graph_groups

    def _validate_data(self):
        """Validate consistency between submatrix size and cropping logic"""
        for size in self.valid_sizes:
            en_num = self.size_to_energy[size]["energy_num"]
            # Validate size calculation correctness (87 - X + 1)
            expected_size = self.full_matrix_size - en_num + 1
            if size != expected_size:
                raise ValueError(f"Size calculation error: en_num={en_num} expected size={expected_size}, actual {size}")
            
            ham_count = len(self.graph_groups[size])
            curve_count = len(self.energy_data[size])
            if ham_count != curve_count:
                raise ValueError(f"Size={size}×{size}: Hamiltonian count {ham_count} does not match curve count {curve_count}")
            
            for ham_idx in range(ham_count):
                node_feat, adj = self.graph_groups[size][ham_idx]
                if node_feat.shape[0] != size or adj.shape != (size, size):
                    raise ValueError(f"{size}×{size} graph structure dimension error (index {ham_idx})")
                
                curve = self.energy_data[size][ham_idx]
                if len(curve) != self.energy_curve_len:
                    raise ValueError(f"{size}×{size} curve length error (index {ham_idx})")
                
                recorded_initial = self.sample_to_initial[(size, ham_idx)]
                curve_initial = curve[0]
                if not np.isclose(recorded_initial, curve_initial, atol=1e-6):
                    raise ValueError(
                        f"{size}×{size} sample {ham_idx} initial value mismatch! "
                        f"Recorded={recorded_initial:.6f}, curve first point={curve_initial:.6f}"
                    )

    def __len__(self):
        return sum(len(self.energy_data[size]) for size in self.valid_sizes)

    def __getitem__(self, idx):
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
        energy_num = meta["energy_num"]  # Save X value (jj-enX.dat)
        time_steps = meta["time_steps"]
        
        initial_energy = energy_curve[0]
        sample_to_initial = self.sample_to_initial[(target_size, inner_idx)]
        
        if not np.isclose(initial_energy, sample_to_initial, atol=1e-6):
            print(f"⚠️ Sample ({target_size}, {inner_idx}) initial value inconsistency: Curve={initial_energy:.6f}, Recorded={sample_to_initial:.6f}")
        
        target_energy = torch.tensor(energy_curve, dtype=torch.float32)
        initial_energy_tensor = torch.tensor([[initial_energy]], dtype=torch.float32)
        
        return (node_feat, adj, target_energy, initial_energy_tensor, 
                target_size, energy_num, time_steps, inner_idx)


# -------------------------- 3. Graph Data Batch Processing Tools --------------------------
def collate_graphs(batch):
    (node_feats, adjs, targets, initial_energies, 
     sizes, energy_nums, time_steps, ham_indices) = zip(*batch)
    
    data_list = []
    for nf, adj, t in zip(node_feats, adjs, targets):
        edges = adj.nonzero().t()
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


# -------------------------- 4. TransformerGNN Model --------------------------
class EnergyPredTransformerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, config, heads=4):
        super().__init__()
        self.config = config
        self.heads = heads
        self.single_head_dim = hidden_dim
        
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
        
        self.bn1 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        self.bn2 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        self.bn3 = nn.BatchNorm1d(self.heads * self.single_head_dim)
        
        self.fc_initial = nn.Linear(1, self.heads * self.single_head_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * self.heads * self.single_head_dim, self.heads * self.single_head_dim),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(self.heads * self.single_head_dim, output_dim)
        )

    def forward(self, batch_data, initial_energies, return_features=False):
        x, edge_index, edge_weight, batch = (
            batch_data.x, 
            batch_data.edge_index, 
            batch_data.edge_weight,
            batch_data.batch
        )
        
        x = self.transformer1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.transformer2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.transformer3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        graph_features = global_mean_pool(x, batch)
        
        initial_energies = initial_energies.view(-1, 1)
        initial_features = self.fc_initial(initial_energies)
        
        combined_features = torch.cat([graph_features, initial_features], dim=1)
        output = self.fc(combined_features)
        
        if return_features:
            return output, combined_features
        return output


# -------------------------- 5. Evaluation and Visualization Tool Functions --------------------------
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
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
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_pred, axis=0)
    
    mean_t = np.mean(y_true_mean)
    ss_total = np.sum((y_true_mean - mean_t) ** 2)
    ss_residual = np.sum((y_true_mean - y_pred_mean) ** 2)
    return 1 - (ss_residual / (ss_total + 1e-8)) if ss_total > 0 else 0.0


def save_energy_pred_to_csv(y_true_mean, y_pred_mean, true_fit, pred_fit, 
                           size, energy_num, time_steps, dataset_type, 
                           save_dir, num_samples):
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
    avg_r2 = mean_energy_r2(np.array([y_true_mean]), np.array([y_pred_mean]))
    plt.figure(figsize=(12, 7))
    
    plt.plot(time_steps, y_true_mean, label="True Energy Curve", color="#2E86AB", linewidth=3.0)
    plt.plot(time_steps, y_pred_mean, label="Averaged Predicted Curve", color="#A23B72", linewidth=3.0)
    
    plt.title(
        f"Energy Prediction - jj-en{energy_num}.dat (Size={size}×{size})\n"
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
        
        true_fit_result = fit_gaussian_decay(time_steps, true_curve, enforce_initial=True)
        pred_fit_result = fit_gaussian_decay(time_steps, pred_mean, enforce_initial=True)
        
        if not np.isclose(true_fit_result['initial_diff'], 0, atol=1e-6):
            print(f"⚠️ Warning: True curve fitting initial value mismatch, difference={true_fit_result['initial_diff']:.2e}")
        if not np.isclose(pred_fit_result['initial_diff'], 0, atol=1e-6):
            print(f"⚠️ Warning: Predicted curve fitting initial value mismatch, difference={pred_fit_result['initial_diff']:.2e}")
        
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
        
        print(f"✅ Processed jj-en{energy_num}.dat (size={size}×{size}) → Mean R²={mean_energy_r2(np.array([true_curve]), np.array([pred_mean])):.4f}")
        print(f"   - Fitted lifetime: True τ={true_fit_result['tau_D']:.2f}fs | Predicted τ={pred_fit_result['tau_D']:.2f}fs")
        print(f"   - Initial value consistency: True difference={true_fit_result['initial_diff']:.2e} | Predicted difference={pred_fit_result['initial_diff']:.2e}")
        print(f"   - Image: {plot_name}")
        print(f"   - CSV: {csv_name}")
        total_plots += 1
        total_csvs += 1
    
    print(f"\n=== {dataset_type.upper()} Set Prediction Completed ===")
    print(f"Generated images: {total_plots} | Generated CSVs: {total_csvs}")
    return total_plots, total_csvs


# -------------------------- 6. Average R² Results Summary --------------------------
def collect_mean_r2_results(dataset, dataset_type, model, device, save_dir):
    model.eval()
    r2_records = []
    all_sizes = sorted(list(set([dataset[i][4] for i in range(len(dataset))])))
    
    print(f"\n=== Collecting {dataset_type.upper()} Set Average R² Results ===")
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
    print(f"✅ {dataset_type.upper()} Set Average R² Summary Completed:")
    print(f"   - Number of images: {len(r2_df_sorted)} (one image per size)")
    print(f"   - Overall average R²: {round(overall_mean_r2, 6)}")
    print(f"   - Save path: {csv_save_path}")
    
    return r2_df_sorted, csv_save_path


# -------------------------- 7. Average Encoded Features Extraction --------------------------
def extract_mean_encoded_features(model, dataset, device, dataset_type, save_dir):
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
    
    print(f"✅ {dataset_type} Average Encoded Features Saved Successfully:")
    print(f"   - Number of energy indices: {len(mean_enc_df)}")
    print(f"   - Feature dimension: {len([col for col in mean_enc_df.columns if col.startswith('mean_encoded_feat_')])}")
    print(f"   - Save path: {mean_enc_path}")
    
    return mean_enc_df, mean_enc_path


# -------------------------- 8. Main Process Functions --------------------------
def regenerate_graph_dataset(save_initial=True, train_energy_count=10):
    config = {
        "res": "./res",  # Hamiltonian directory (87×87 matrices)
        "res_target_energy": "./res_target_energy",  # Energy data directory (jj-enX.dat)
        "ham_count_per_group": 1001,  # Number of Hamiltonian samples per group
        "save_dir": "./shi_mo_data",  # Data save directory
        "train_energy_count": train_energy_count,  # Number of en_nums included in training set
        "random_state": 42
    }
    
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["res"], exist_ok=True)
    os.makedirs(config["res_target_energy"], exist_ok=True)
    
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    
    full_dataset = GraphEnergyDataset(
        config=config, 
        ham_count_per_group=config["ham_count_per_group"],
        load_initial_from_csv=not save_initial,
        initial_csv_path=os.path.join(config["save_dir"], "energy_initial_values.csv") if not save_initial else None
    )
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty, please check file paths")
    
    # Split training and test sets by en_num
    all_energy_nums = [full_dataset.size_to_energy[size]["energy_num"] for size in full_dataset.valid_sizes]
    all_energy_nums = sorted(list(set(all_energy_nums)))
    
    if len(all_energy_nums) < config["train_energy_count"]:
        raise ValueError(f"Insufficient available energy indices! Found {len(all_energy_nums)}, need {config['train_energy_count']} for training set")
    
    min_en = all_energy_nums[0]
    max_en = all_energy_nums[-1]
    train_energy_indices = [min_en, max_en]
    
    remaining_ens = [en for en in all_energy_nums if en not in train_energy_indices]
    need = config["train_energy_count"] - 2
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
    
    print(f"\n=== Dataset Split ===")
    print(f"All energy indices (en_num): {all_energy_nums}")
    print(f"Minimum en_num: {min_en}, Maximum en_num: {max_en}")
    print(f"Training set en_nums ({len(train_energy_indices)}): {sorted(train_energy_indices)}")
    print(f"Test set en_nums ({len(test_energy_indices)}): {sorted(test_energy_indices)}")
    
    # Build training and test set indices
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
    
    train_data = [full_dataset[i] for i in train_indices]
    test_data = [full_dataset[i] for i in test_indices]
    
    try:
        with open(os.path.join(config["save_dir"], "train_graph_energy_dataset.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        with open(os.path.join(config["save_dir"], "test_graph_energy_dataset.pkl"), "wb") as f:
            pickle.dump(test_data, f)
        
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
        
        print(f"\n✅ Dataset Saved Successfully:")
        print(f"  - Training set: {len(train_data)} samples | Test set: {len(test_data)} samples")
        print(f"  - Initial energy file: {data_params['initial_csv_path']}")
        print(f"  - Save path: {config['save_dir']}")
    
    except Exception as e:
        print(f"❌ Failed to save dataset: {str(e)}")
        raise


def main_training(train_energy_count=10):
    config = {
        "processed_data_dir": "./shi_mo_data",
        "model_save_dir": "./shi_mo_models",
        "prediction_save_dir": "./Si_QD_hole_predictions",
        "features_save_dir": "./Si_QD_hole_features",
        "batch_size": 32,
        "hidden_dim": 32,
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "epochs": 20,
        "print_interval": 1,
        "random_state": 42,
        "transformer_heads": 6,
        "ham_count_per_group": 1001
    }
    
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    torch.manual_seed(config["random_state"])
    torch.cuda.manual_seed_all(config["random_state"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config["processed_data_dir"], exist_ok=True)
    os.makedirs(config["model_save_dir"], exist_ok=True)
    os.makedirs(config["prediction_save_dir"], exist_ok=True)
    os.makedirs(config["features_save_dir"], exist_ok=True)
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
    
    print(f"Training set en_nums: {data_params['train_energy_indices']}")
    print(f"Test set en_nums: {data_params['test_energy_indices']}")
    
    class WrappedEnergyDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = WrappedEnergyDataset(train_data)
    test_dataset = WrappedEnergyDataset(test_data)
    
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
    model.config = config
    print(model)
    
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
        model.train()
        train_total_loss = 0.0
        train_total_r2 = 0.0
        
        for batch in train_loader:
            batch_data, targets, initial_energies, _, _, _, _ = batch
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            initial_energies = initial_energies.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_data, initial_energies)
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
        
        scheduler.step(avg_test_loss)
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"📌 Epoch {epoch+1}: Saving best model (test loss: {best_test_loss:.6f})")
        
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
    
    # Average R² summary
    print("\n=== Average R² Results Summary ===")
    train_r2_df, train_r2_csv = collect_mean_r2_results(
        dataset=train_dataset,
        dataset_type="train",
        model=model,
        device=device,
        save_dir=config["prediction_save_dir"]
    )
    
    test_r2_df, test_r2_csv = collect_mean_r2_results(
        dataset=test_dataset,
        dataset_type="test",
        model=model,
        device=device,
        save_dir=config["prediction_save_dir"]
    )
    
    # Average encoded features extraction
    print("\n=== Extracting and Saving Average Encoded Features ===")
    print("\n--- Training Set Average Encoded Features Extraction ---")
    train_mean_enc_df, train_mean_enc_path = extract_mean_encoded_features(
        model=model,
        dataset=train_dataset,
        device=device,
        dataset_type="train",
        save_dir=config["features_save_dir"]
    )
    
    print("\n--- Test Set Average Encoded Features Extraction ---")
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


# -------------------------- Main Program Entry --------------------------
if __name__ == "__main__":
    # Customize number of en_nums in training set (≥2)
    custom_train_count = 12  # Example: Use 5 en_nums as training set
    
    # Step 1: Generate dataset (set save_initial=True for first run, False for subsequent runs)
    regenerate_graph_dataset(save_initial=True, train_energy_count=custom_train_count)
    
    # Step 2: Train model and predict
    main_training(train_energy_count=custom_train_count)