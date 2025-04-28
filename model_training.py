"""
Stock Price Prediction System - Model Training Module
This module implements a recurrent neural network (LSTM or GRU) based deep learning model
with an attention mechanism for stock price prediction using PyTorch with CUDA acceleration.
Includes hyperparameter optimization using Optuna with walk-forward validation.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Removed ParameterGrid, added optuna
import optuna
import matplotlib.pyplot as plt
import datetime
import joblib
import math
import json
import traceback
import time

# Set up Multi-GPU support with DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Fix for Kaggle multi-GPU utilization
def initialize_multi_gpu():
    """Initialize multi-GPU environment properly for Kaggle"""
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return None
    
    # Get number of available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Only {n_gpus} GPU found. Multi-GPU mode disabled.")
        return [0] if n_gpus == 1 else None
    
    print(f"Initializing {n_gpus} GPUs for parallel training")
    
    # Print device information
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - CUDA Capability: {props.major}.{props.minor}")
    
    # CRITICAL FIX FOR KAGGLE: Force memory allocation on each GPU
    # This makes sure Kaggle recognizes that each GPU is actually being used
    dummy_tensors = []
    for i in range(n_gpus):
        # Explicitly select device
        with torch.cuda.device(i):
            # Create a substantial tensor on each GPU to force device activation
            # Using 500MB per GPU to ensure Kaggle recognizes GPU utilization
            tensor_size = 4000  # Creates roughly a 500MB tensor (4000×4000×4bytes)
            dummy = torch.ones(tensor_size, tensor_size, device=f'cuda:{i}')
            # Force computation to ensure GPU is activated
            dummy = dummy + dummy
            # Keep tensor in memory until all GPUs are initialized
            dummy_tensors.append(dummy)
            # Verify memory allocation
            torch.cuda.synchronize(i)  # Ensure operation completes
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"    - Allocated {allocated:.2f} GB on GPU {i}")
            
            # ENHANCED: Perform additional CUDA operations to ensure GPU is recognized
            # This helps with T4 GPU detection in some environments
            dummy_matmul = torch.matmul(dummy[:1000, :1000], dummy[:1000, :1000].t())
            dummy_norm = torch.norm(dummy_matmul)
            torch.cuda.synchronize(i)
            print(f"    - Additional CUDA operations completed on GPU {i}")
    
    # Now clean up
    for i, dummy in enumerate(dummy_tensors):
        del dummy
    dummy_tensors = []
    torch.cuda.empty_cache()
    
    # Ensure CUDA_VISIBLE_DEVICES is set correctly for Kaggle
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(n_gpus)])
    # Force Kaggle to use both GPUs by setting this environment variable
    os.environ["NVIDIA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(n_gpus)])
    
    # ENHANCED: Add specific environment variables to ensure both T4 GPUs are utilized
    # This is crucial for some cloud environments including Kaggle
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["NCCL_DEBUG"] = "INFO"  # Helps diagnose multi-GPU communication issues
    os.environ["NCCL_P2P_DISABLE"] = "0"  # Ensure peer-to-peer communication is enabled
    
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"NVIDIA_VISIBLE_DEVICES set to: {os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    
    # Return list of device ids
    return list(range(n_gpus))

# Replace the setup_gpu_devices function with this improved version
def setup_gpu_devices():
    """Setup GPU devices for multi-GPU training with Kaggle compatibility"""
    devices = initialize_multi_gpu()
    
    if devices and len(devices) >= 2:
        print(f"Multi-GPU training enabled on {len(devices)} devices: {devices}")
        
        # Check GPU memory usage after initialization
        for i in devices:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i} Memory after initialization: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
            
        # Set optimizations for T4 GPUs
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes (common in training)
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 in cuDNN ops
        
        # Force peer-to-peer communication between GPUs
        if hasattr(torch.cuda, 'set_device'):
            # Initialize peer access between all GPUs
            for i in devices:
                torch.cuda.set_device(i)
                for j in devices:
                     if i != j:
                        try:
                            # Check if peer access is possible
                            can_access_peer = torch.cuda.can_device_access_peer(i, j)
                            if can_access_peer:
                                torch.cuda.device_enable_peer_access(i, j)
                                print(f"Enabled peer access from GPU {i} → GPU {j}")
                            else:
                                print(f"Peer access not supported from GPU {i} → GPU {j}")
                        except (RuntimeError, AttributeError) as e:
                            print(f"Could not enable peer access {i} → {j}: {e}")
        
        # Synchronize all devices to ensure they're ready
        for i in devices:
            torch.cuda.synchronize(i)
        
        return devices
    elif devices and len(devices) == 1:
        print(f"Single GPU training on: {torch.cuda.get_device_name(0)}")
        # Even for single GPU, set optimization flags
        torch.backends.cudnn.benchmark = True
        return [0]  # Single GPU
    else:
        print("CUDA not available. Training will use CPU only.")
        return None

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Directory setup
ROOT_DIR = '/content/drive/MyDrive'
DATA_DIR = '/content/drive/MyDrive/data' # Changed for Colab/general use
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
# Changed MODEL_DIR to be relative or absolute depending on environment
# Check if running in Kaggle environment
if os.path.exists('/kaggle/working/'):
    print("Kaggle environment detected. Setting MODEL_DIR to /kaggle/working/")
    MODEL_DIR = '/kaggle/working/' # Use Kaggle's writable directory
else:
    # Default for local/other environments
    MODEL_DIR = 'models' # Output models to the local 'models' directory

# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    print(f"Creating model directory: {MODEL_DIR}")
    os.makedirs(MODEL_DIR)
else:
    print(f"Model directory already exists: {MODEL_DIR}")

class Attention(nn.Module):
    """Simple Attention Layer"""
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.feature_dim = feature_dim; self.step_dim = step_dim; self.bias = bias
        self.attention_weights = nn.Parameter(torch.Tensor(feature_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)
        if bias: self.attention_bias = nn.Parameter(torch.Tensor(1)); nn.init.zeros_(self.attention_bias)

    def forward(self, x):
        eij = torch.matmul(x, self.attention_weights).squeeze(-1)
        if self.bias: eij = eij + self.attention_bias
        eij = torch.tanh(eij); a = torch.softmax(eij, dim=1)
        context = torch.sum(a.unsqueeze(-1) * x, dim=1)
        return context, a

class RecurrentAttentionModel(nn.Module):
    """Enhanced model architecture combining Recurrent layers, Transformers, and CNN for superior feature extraction"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, lookback_window, rnn_type='lstm', 
                 dropout_prob=0.2, bidirectional=True, transformer_layers=2, transformer_heads=8):
        super(RecurrentAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # CNN feature extraction layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Recurrent layers (LSTM or GRU)
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                              dropout=dropout_prob if num_layers > 1 else 0, bidirectional=bidirectional)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                              dropout=dropout_prob if num_layers > 1 else 0, bidirectional=bidirectional)
        else: 
            raise ValueError("Invalid rnn_type. Choose 'lstm' or 'gru'.")

        # Self-attention transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.rnn_output_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Attention mechanism for temporal feature importance
        self.attention = Attention(self.rnn_output_dim, lookback_window)
        
        # Dropout layers
        self.cnn_dropout = nn.Dropout(dropout_prob/2)  # Less aggressive dropout for CNN layers
        self.rnn_dropout = nn.Dropout(dropout_prob)
        self.fc_dropout1 = nn.Dropout(dropout_prob)
        self.fc_dropout2 = nn.Dropout(dropout_prob)
        
        # Dense layers with residual connections
        self.fc1 = nn.Linear(self.rnn_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Activations
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # GELU typically works better with transformers
        
        # Residual/skip connection layer (adapts dimensions if needed)
        self.residual_adapter = nn.Linear(self.rnn_output_dim, hidden_dim) if self.rnn_output_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        
        # Reshape for CNN: [batch_size, features, seq_len]
        batch_size, seq_len, features = x.size()
        x_cnn = x.permute(0, 2, 1)
        
        # CNN feature extraction with residual connections
        cnn_out = self.relu(self.bn1(self.conv1(x_cnn)))
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = self.relu(self.bn2(self.conv2(cnn_out)))
        cnn_out = self.cnn_dropout(cnn_out)
        
        # Reshape back for RNN: [batch_size, seq_len, hidden_dim]
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # RNN processing
        rnn_out, _ = self.rnn(cnn_out)
        
        # Transformer encoder for global dependencies
        transformer_out = self.transformer_encoder(rnn_out)
        
        # Attention mechanism
        context, _ = self.attention(transformer_out)
        context = self.rnn_dropout(context)
        
        # MLP with residual connections
        residual = self.residual_adapter(context)
        fc1_out = self.fc1(context)
        fc1_out = self.gelu(fc1_out)
        fc1_out = self.fc_dropout1(fc1_out)
        fc1_out = fc1_out + residual  # Skip connection
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.gelu(fc2_out)
        fc2_out = self.fc_dropout2(fc2_out)
        
        output = self.fc3(fc2_out)
        
        return output

class ModelTrainer:
    """Model trainer using RNN with Attention, supporting Optuna optimization"""
    def __init__(self, lookback_window=30, batch_size=64, epochs=50, num_splits=5):
        self.lookback_window = lookback_window
        self.batch_size = batch_size # Default, overridden by Optuna/Grid
        self.epochs = epochs
        self.num_splits = num_splits
        self.model = None
        self.scalers = {}
        self.feature_columns = None
        self.target_column = 'Price_Increase'
        self.model_hyperparams = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Add multi_gpu attribute and default to False
        self.multi_gpu = False
        self.devices = None
        print(f"ModelTrainer using device: {self.device}")

    def _prepare_sequences(self, data, target_col):
        """Prepare sequences for RNN model"""
        X, y = [], []
        if self.feature_columns is None: raise ValueError("feature_columns must be set")
        missing = [f for f in self.feature_columns if f not in data.columns]
        if missing: raise ValueError(f"Missing features: {missing}")
        features = data[self.feature_columns].values; targets = data[target_col].values
        if len(features) <= self.lookback_window: return np.array([]), np.array([])
        for i in range(len(features) - self.lookback_window):
            X.append(features[i:i+self.lookback_window]); y.append(targets[i+self.lookback_window])
        return np.array(X), np.array(y)

    def get_walk_forward_splits(self, df):
        """Generates train/validation splits for walk-forward validation."""
        if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Requires DatetimeIndex")
        df = df.sort_index(); total_len = len(df)
        num_segments = self.num_splits + 1; split_size = total_len // num_segments
        min_train_size = self.lookback_window + 1
        if split_size < 1: raise ValueError(f"Data size {total_len} too small for {self.num_splits} splits.")
        if split_size < min_train_size: print(f"Warning: Train split size {split_size} small vs lookback {self.lookback_window}.")
        print(f"Generating {self.num_splits} splits. Total: {total_len}. Segment: {split_size}")
        for i in range(self.num_splits):
            train_end = split_size * (i + 1); val_start = train_end; val_end = val_start + split_size
            if i == self.num_splits - 1: val_end = total_len
            train_df = df.iloc[0:train_end]; validation_df = df.iloc[val_start:val_end]
            if validation_df.empty: print(f"Warn: Fold {i+1} val empty. Stopping."); break
            if len(train_df) < min_train_size: print(f"Warn: Fold {i+1} train too small. Skipping."); continue
            print(f"  Split {i+1}: Train {train_df.index.min().date()}-{train_df.index.max().date()} ({len(train_df)}), Val {validation_df.index.min().date()}-{validation_df.index.max().date()} ({len(validation_df)})")
            yield train_df, validation_df

    def _prepare_fold_data(self, train_df, validation_df, ticker, batch_size, noise_level=0.01):
        """Prepares sequences and DataLoaders for a single walk-forward fold."""
        try:
            if train_df is None or train_df.empty or validation_df is None or validation_df.empty: return None, None, None, None, None, None
            target_col = self.target_column
            if target_col not in train_df.columns or target_col not in validation_df.columns: return None, None, None, None, None, None

            missing_train = [f for f in self.feature_columns if f not in train_df.columns]
            missing_val = [f for f in self.feature_columns if f not in validation_df.columns]
            if missing_train or missing_val:
                all_missing = sorted(list(set(missing_train + missing_val)))
                print(f"Warn: Features missing {ticker}: {all_missing[:5]}...")
                current_fold_features = [f for f in self.feature_columns if f in train_df.columns and f in validation_df.columns]
                if not current_fold_features: print(f"Error: No common features {ticker}."); return None, None, None, None, None, None
            else: current_fold_features = self.feature_columns

            cols_to_check = current_fold_features + [target_col]
            if train_df[cols_to_check].isna().any().any() or np.isinf(train_df[cols_to_check].values).any(): train_df = train_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            if validation_df[cols_to_check].isna().any().any() or np.isinf(validation_df[cols_to_check].values).any(): validation_df = validation_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            train_df = train_df.dropna(subset=cols_to_check); validation_df = validation_df.dropna(subset=cols_to_check)
            if len(train_df) <= self.lookback_window or len(validation_df) == 0: return None, None, None, None, None, None

            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_df[current_fold_features])
            train_scaled_df = pd.DataFrame(train_features_scaled, columns=current_fold_features, index=train_df.index)
            validation_features_scaled = scaler.transform(validation_df[current_fold_features])
            validation_scaled_df = pd.DataFrame(validation_features_scaled, columns=current_fold_features, index=validation_df.index)
            train_scaled_df[target_col] = train_df[target_col]; validation_scaled_df[target_col] = validation_df[target_col]

            original_features = self.feature_columns; self.feature_columns = current_fold_features
            X_train, y_train = self._prepare_sequences(train_scaled_df, target_col)
            X_validation, y_validation = self._prepare_sequences(validation_scaled_df, target_col)
            self.feature_columns = original_features

            if noise_level > 0 and X_train.size > 0: X_train = X_train + np.random.normal(0, noise_level, X_train.shape)
            if X_train.size == 0 or X_validation.size == 0: return None, None, None, None, None, None

            X_train_tensor = torch.FloatTensor(X_train).to(self.device); y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            X_validation_tensor = torch.FloatTensor(X_validation).to(self.device); y_validation_tensor = torch.FloatTensor(y_validation).unsqueeze(1).to(self.device)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor); validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
            # Conditionally set num_workers based on device type to avoid CUDA init errors in workers
            if self.device.type == 'cuda':
                num_workers = 0 # Set to 0 for CUDA to prevent initialization errors
                pin_memory = False # Pin memory typically used with workers > 0
            else:
                num_workers = 4 # Use multiple workers for CPU
                pin_memory = False # No pinning needed for CPU
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            return train_loader, validation_loader, X_validation, y_validation, y_train, scaler
        except Exception as e: print(f"Error prep fold {ticker}: {e}"); traceback.print_exc(); return None, None, None, None, None, None

    def train_single_run(self, train_loader, val_loader, params, pos_weight, lookback_window, patience, epoch_callback=None):
        """Train the model using multiple GPUs if available"""
        input_dim = len(self.feature_columns)
        output_dim = 1
        
        # Initialize base model
        base_model = RecurrentAttentionModel(
            input_dim=input_dim, 
            hidden_dim=params['hidden_dim'], 
            num_layers=params['num_layers'],
            output_dim=output_dim, 
            lookback_window=lookback_window, 
            rnn_type=params.get('rnn_type', 'lstm'),
            dropout_prob=params['dropout_prob'], 
            bidirectional=params.get('bidirectional', True),
            transformer_layers=params.get('transformer_layers', 2),
            transformer_heads=params.get('transformer_heads', 8)
        )
        
        # Set up for multi-GPU training if enabled
        if self.multi_gpu:
            print(f"Initializing model for multi-GPU training on {len(self.devices)} devices")
            
            # Force CUDA to recognize all devices before creating DataParallel model
            for i in self.devices:
                with torch.cuda.device(i):
                    # Create a small tensor on each device to ensure it's initialized
                    torch.ones(1, device=f'cuda:{i}')
                    # Force some computation to activate GPU
                    test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                    _ = torch.matmul(test_tensor, test_tensor.t())
                    torch.cuda.synchronize(i)
                    print(f"  GPU {i} initialized for training")
            
            # Use DataParallel for multi-GPU training with explicit device mapping
            model = nn.DataParallel(base_model, device_ids=self.devices)
            
            # Move model to primary device
            model = model.to(f'cuda:{self.devices[0]}')
            
            # Set environment variable to force multi-GPU usage in Kaggle
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in self.devices])
            print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            
            # Monitor GPU memory after model creation
            for i in self.devices:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                print(f"GPU {i} Memory after model creation: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
            
            # Adjust batch size for multi-GPU (each GPU gets a portion of the batch)
            effective_batch_size = params['batch_size'] * len(self.devices)
            print(f"Effective batch size with {len(self.devices)} GPUs: {effective_batch_size}")
        else:
            # Single GPU or CPU mode
            model = base_model.to(self.device)
        
        # Set up loss function with stronger class balancing
        if pos_weight is None: 
            pos_weight = torch.tensor(1.0).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Advanced optimizer configuration for better convergence
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'], 
            weight_decay=params.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Adjust learning rate scheduler for multi-GPU if needed
        if self.multi_gpu:
            # Scale learning rate based on number of GPUs
            scaled_lr = params['learning_rate'] * len(self.devices)**0.5
            print(f"Scaling learning rate: {params['learning_rate']} → {scaled_lr}")
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = scaled_lr
        
        # Multi-stage learning rate scheduler for better convergence
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimizer.param_groups[0]['lr'],  # Use the current LR from optimizer
            steps_per_epoch=len(train_loader),
            epochs=self.epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # Backup plateau scheduler for final fine-tuning
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=params['learning_rate']/100
        )
        
        early_stopping_patience = patience
        early_stopping_counter = 0
        best_val_loss = float('inf')
        best_model_state = None
        train_losses, val_losses = [], []
        
        # T4 GPU Optimization with mixed precision
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        # Optimize CUDA configurations for best performance with multiple GPUs
        if self.device.type == 'cuda':
            # Set benchmark mode for improved performance with fixed input sizes
            torch.backends.cudnn.benchmark = True
            
            # When using multiple T4 GPUs, we might need to adjust TF32 settings
            if self.multi_gpu:
                # Enable TF32 for faster computation across multiple GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Monitor memory usage
                for i, device_id in enumerate(self.devices):
                    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                    print(f"GPU {device_id} Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
        
        # Track best metrics
        best_metrics = {'accuracy': 0, 'f1_positive': 0}
        last_plateau_epoch = 0
        
        try:
            for epoch in range(self.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                if len(train_loader.dataset) == 0:
                    continue
                
                # For distributed training, set the epoch for the sampler
                if self.multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                optimizer.zero_grad(set_to_none=True)
                
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    # Use autocast for mixed precision training
                    with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                        # Forward pass - DataParallel will split batch across GPUs automatically
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                    
                    # Scale loss and backpropagate with mixed precision
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation for larger effective batch size
                    accumulation_steps = params.get('accumulation_steps', 4)
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        # Unscale before gradient clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get('max_grad_norm', 1.0))
                        
                        # Step optimizer and update scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        
                        # Step OneCycleLR scheduler which is called per-iteration
                        lr_scheduler.step()
                    
                    # Track loss
                    train_loss += loss.item() * X_batch.size(0)
                    
                    # Status update for multi-GPU training
                    if self.multi_gpu and (batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1):
                        print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                
                # Calculate average training loss
                avg_train_loss = train_loss / len(train_loader.dataset)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                # Skip validation if no validation data
                if len(val_loader.dataset) == 0:
                    avg_val_loss = float('inf')
                else:
                    val_preds = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            # Use autocast for consistent precision during validation
                            with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                                y_pred = model(X_batch)
                                loss = criterion(y_pred, y_batch)
                            
                            # Store predictions and targets for metrics calculation
                            val_preds.append(torch.sigmoid(y_pred).cpu())
                            val_targets.append(y_batch.cpu())
                            
                            val_loss += loss.item() * X_batch.size(0)
                    
                    # Calculate validation metrics
                    avg_val_loss = val_loss / len(val_loader.dataset)
                    val_losses.append(avg_val_loss)
                    
                    # Calculate validation metrics every 5 epochs or on last epoch
                    if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                        val_preds_cat = torch.cat(val_preds, dim=0).numpy().flatten()
                        val_targets_cat = torch.cat(val_targets, dim=0).numpy().flatten()
                        val_preds_binary = (val_preds_cat > 0.5).astype(int)
                        
                        try:
                            accuracy = accuracy_score(val_targets_cat, val_preds_binary)
                            precision = precision_score(val_targets_cat, val_preds_binary, zero_division=0)
                            recall = recall_score(val_targets_cat, val_preds_binary, zero_division=0)
                            f1 = f1_score(val_targets_cat, val_preds_binary, zero_division=0)
                            
                            print(f'  Epoch {epoch+1}/{self.epochs} Metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, '
                                  f'Recall={recall:.4f}, F1={f1:.4f}, Val Loss={avg_val_loss:.6f}')
                            
                            # Track best metrics
                            if f1 > best_metrics['f1_positive']:
                                best_metrics['f1_positive'] = f1
                                best_metrics['accuracy'] = accuracy
                        except Exception as e:
                            print(f"  Error calculating metrics: {e}")
                    
                    # Step ReduceLROnPlateau scheduler once per epoch
                    plateau_scheduler.step(avg_val_loss)
                    
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Check for learning rate plateau
                    if epoch > 0 and epoch - last_plateau_epoch >= 10:
                        # Apply warmup restart with 10% of initial learning rate
                        restart_lr = params['learning_rate'] * 0.1
                        if self.multi_gpu:
                            restart_lr *= len(self.devices)**0.5
                        
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = restart_lr
                            
                        last_plateau_epoch = epoch
                        print(f"  Restarting learning rate at epoch {epoch+1} to {restart_lr:.7f}")
                
                # Save best model state
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    
                    # For DataParallel, we need to save the module state
                    if self.multi_gpu:
                        best_model_state = model.module.state_dict().copy()
                    else:
                        best_model_state = model.state_dict().copy()
                        
                    print(f'  Epoch {epoch+1}: New best model saved (Val Loss: {avg_val_loss:.6f})')
                else:
                    early_stopping_counter += 1
                
                # Progress reporting
                if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1 or early_stopping_counter >= early_stopping_patience:
                    print(f'  Epoch {epoch+1}/{self.epochs}: Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.7f}')
                
                # Callback for external reporting
                if epoch_callback is not None:
                    epoch_callback(epoch, avg_train_loss, avg_val_loss)
                
                # Early stopping check
                if early_stopping_counter >= early_stopping_patience:
                    print(f'  Early stopping triggered after epoch {epoch+1} '
                          f'(no improvement for {early_stopping_patience} epochs)')
                    break
            
            # Load best model before returning
            if best_model_state:
                # For returning, we need to use the base model without DataParallel
                if self.multi_gpu:
                    # Create a new base model to hold the best state
                    final_model = RecurrentAttentionModel(
                        input_dim=input_dim, 
                        hidden_dim=params['hidden_dim'], 
                        num_layers=params['num_layers'],
                        output_dim=output_dim, 
                        lookback_window=lookback_window, 
                        rnn_type=params.get('rnn_type', 'lstm'),
                        dropout_prob=params['dropout_prob'], 
                        bidirectional=params.get('bidirectional', True),
                        transformer_layers=params.get('transformer_layers', 2),
                        transformer_heads=params.get('transformer_heads', 8)
                    ).to(self.device)
                    
                    # Load the best state
                    final_model.load_state_dict(best_model_state)
                else:
                    # Just use the original model with the best state
                    model.load_state_dict(best_model_state)
                    final_model = model
                
                print(f"  Loaded best model (Val Loss: {best_val_loss:.6f})")
                print(f"  Best metrics - F1: {best_metrics['f1_positive']:.4f}, "
                      f"Accuracy: {best_metrics['accuracy']:.4f}")
                
                return final_model, train_losses, val_losses
            else:
                # If no best model was saved, return the current model
                if self.multi_gpu:
                    # For multi-GPU, return the module without DataParallel wrapper
                    return model.module, train_losses, val_losses
                else:
                    return model, train_losses, val_losses
                
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            return None, [], []

    def evaluate(self, model, X_validation, y_validation, batch_size, ticker='default'): # Optimize: Pass batch_size
        """Evaluate the model on validation data"""
        if model is None or len(X_validation) == 0: return None, {}
        y_val_class = y_validation.astype(int) # Keep original y for metrics
        model.eval()
        all_pred_logits = []
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_validation))
        # Optimize: Use specified batch_size and add num_workers/pin_memory
        # Conditionally set num_workers based on device type
        if self.device.type == 'cuda':
            num_workers = 0
            pin_memory = False # Pin memory typically used with workers > 0
        else:
            num_workers = 4
            pin_memory = False
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        try:
            with torch.no_grad():
                for (batch_X,) in val_loader: # Unpack tuple from TensorDataset
                    batch_X = batch_X.to(self.device)
                    # Optimize: Use autocast during evaluation inference
                    with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                        batch_logits = model(batch_X).cpu() # Move to CPU after model forward pass
                    all_pred_logits.append(batch_logits)

            y_pred_logits = torch.cat(all_pred_logits, dim=0)
            y_pred_probs = torch.sigmoid(y_pred_logits).numpy().flatten(); y_pred_class = (y_pred_probs > 0.5).astype(int)
            try:
                accuracy = accuracy_score(y_val_class, y_pred_class)
                precision = precision_score(y_val_class, y_pred_class, pos_label=1, zero_division=0)
                recall = recall_score(y_val_class, y_pred_class, pos_label=1, zero_division=0)
                f1 = f1_score(y_val_class, y_pred_class, pos_label=1, zero_division=0)
                try: roc_auc = roc_auc_score(y_val_class, y_pred_probs)
                except ValueError: roc_auc = np.nan
                cm = confusion_matrix(y_val_class, y_pred_class)
                metrics = {'accuracy': accuracy, 'roc_auc': roc_auc, 'precision_positive': precision,
                           'recall_positive': recall, 'f1_positive': f1, 'confusion_matrix': cm.tolist()}
                return y_pred_probs, metrics
            except Exception as metrics_err: print(f"Metrics error: {metrics_err}"); return None, {}
        except Exception as e: print(f"Eval error: {e}"); traceback.print_exc(); return None, {}

    def predict(self, df, ticker):
        """Make classification prediction"""
        if self.model is None: raise ValueError("Model not loaded.")
        scaler = None; scalers_path = os.path.join(MODEL_DIR, 'scalers.pkl')
        if os.path.exists(scalers_path):
             try:
                 scalers_dict = joblib.load(scalers_path)
                 if isinstance(scalers_dict, dict):
                      keys_to_try = [ticker, 'combined', 'last_fold']; scaler_key = None
                      for key in keys_to_try:
                          if key in scalers_dict: scaler = scalers_dict[key]; scaler_key = key; break
                      if scaler is None and scalers_dict: scaler = list(scalers_dict.values())[0]; scaler_key = list(scalers_dict.keys())[0]
                      if scaler: print(f"Predict: Loaded scaler '{scaler_key}'.")
                      else: raise ValueError("No suitable scaler found.")
                 else: raise ValueError("Invalid scaler file format.")
             except Exception as e: raise ValueError(f"Error loading scaler: {e}")
        else: raise ValueError(f"Scaler file not found: {scalers_path}")

        self.model.eval()
        if self.feature_columns is None: raise ValueError("Feature columns not set.")
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing: print(f"Warn: Missing features: {missing}. Filling 0."); df[missing] = 0.0
        if len(df) < self.lookback_window: raise ValueError(f"Not enough data. Need {self.lookback_window}, got {len(df)}")
        X = df[self.feature_columns].iloc[-self.lookback_window:].values
        if np.isnan(X).any() or np.isinf(X).any(): X = np.nan_to_num(X)

        try: X_scaled = scaler.transform(X)
        except ValueError as ve: raise ValueError(f"Scaler/feature mismatch: {ve}")
        except Exception as e: raise ValueError(f"Scaling error: {e}")

        X_sequence = X_scaled.reshape(1, self.lookback_window, len(self.feature_columns))
        X_tensor = torch.tensor(X_sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad(): prediction_logits = self.model(X_tensor).cpu()
        return torch.sigmoid(prediction_logits).item()

class MultiGPUModelTrainer(ModelTrainer):
    """Enhanced Model Trainer that utilizes multiple GPUs for parallel training"""
    def __init__(self, lookback_window=30, batch_size=64, epochs=50, num_splits=5, devices=None):
        super().__init__(lookback_window, batch_size, epochs, num_splits)
        self.devices = devices if devices is not None else setup_gpu_devices()
        self.multi_gpu = self.devices is not None and len(self.devices) > 1
        if self.multi_gpu:
            print(f"Multi-GPU training enabled on {len(self.devices)} devices: {self.devices}")
            
            # Set environment variables to ensure both GPUs are utilized
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in self.devices])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_P2P_DISABLE"] = "0"
        else:
            print("Single-GPU or CPU training mode")

    def _prepare_fold_data(self, train_df, validation_df, ticker, batch_size, noise_level=0.01):
        """Enhanced data preparation for multi-GPU training"""
        try:
            # Call the parent method to get the basic data preparation
            result = super()._prepare_fold_data(train_df, validation_df, ticker, batch_size, noise_level)
            
            if result is None:
                return None
                
            train_loader, val_loader, X_validation, y_validation, y_train, scaler = result
            
            # If multi-GPU is enabled, modify the data loaders to use DistributedSampler
            if self.multi_gpu and len(self.devices) > 1:
                # Create new DataLoaders with optimized settings for multi-GPU training
                train_dataset = train_loader.dataset
                val_dataset = val_loader.dataset
                
                # Use larger batch size proportional to number of GPUs
                actual_batch_size = batch_size * len(self.devices)
                print(f"Multi-GPU enabled: scaling batch size from {batch_size} to {actual_batch_size}")
                
                # Create samplers for each dataset
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=len(self.devices),
                    rank=0,  # Set rank to 0 for the main process
                    shuffle=True,
                    drop_last=False
                )
                
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=len(self.devices),
                    rank=0,  # Set rank to 0 for the main process
                    shuffle=False,
                    drop_last=False
                )
                
                # Recreate DataLoaders with the samplers
                # For CUDA, use 0 workers to avoid initialization errors
                num_workers = 0
                pin_memory = True  # Enable pin_memory for faster data transfer to GPU
                
                # Create optimized data loaders for multi-GPU training
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=actual_batch_size,
                    sampler=train_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=actual_batch_size,
                    sampler=val_sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
                
                print(f"Created multi-GPU data loaders with effective batch size {actual_batch_size}")
            
            return train_loader, val_loader, X_validation, y_validation, y_train, scaler
            
        except Exception as e:
            print(f"Error preparing fold data for multi-GPU: {e}")
            traceback.print_exc()
            return None

    def train_single_run(self, train_loader, val_loader, params, pos_weight, lookback_window, patience, epoch_callback=None):
        """Train the model using multiple GPUs if available"""
        input_dim = len(self.feature_columns)
        output_dim = 1
        
        # Initialize base model
        base_model = RecurrentAttentionModel(
            input_dim=input_dim, 
            hidden_dim=params['hidden_dim'], 
            num_layers=params['num_layers'],
            output_dim=output_dim, 
            lookback_window=lookback_window, 
            rnn_type=params.get('rnn_type', 'lstm'),
            dropout_prob=params['dropout_prob'], 
            bidirectional=params.get('bidirectional', True),
            transformer_layers=params.get('transformer_layers', 2),
            transformer_heads=params.get('transformer_heads', 8)
        )
        
        # Set up for multi-GPU training if enabled
        if self.multi_gpu and len(self.devices) > 1:
            print(f"Initializing model for multi-GPU training on {len(self.devices)} devices")
            
            # Pre-initialize each GPU before creating DataParallel model
            for i in self.devices:
                with torch.cuda.device(i):
                    # Force GPU initialization with computation
                    test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                    _ = torch.matmul(test_tensor, test_tensor.t())
                    torch.cuda.synchronize(i)
                    print(f"  GPU {i} initialized with test computation")
                    
                    # Check memory usage
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    print(f"  GPU {i} Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
            
            # Use DataParallel for multi-GPU training with explicit device mapping
            model = nn.DataParallel(base_model, device_ids=self.devices)
            
            # Move model to primary device
            model = model.to(f'cuda:{self.devices[0]}')
            
            # Adjust batch size for multi-GPU (each GPU gets a portion of the batch)
            effective_batch_size = params['batch_size'] * len(self.devices)
            print(f"Effective batch size with {len(self.devices)} GPUs: {effective_batch_size}")
            
            # Monitor GPU memory after model creation
            for i in self.devices:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                print(f"GPU {i} Memory after model creation: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
        else:
            # Single GPU or CPU mode
            model = base_model.to(self.device)
        
        # Set up loss function with stronger class balancing
        if pos_weight is None: 
            pos_weight = torch.tensor(1.0).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Advanced optimizer configuration for better convergence
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'], 
            weight_decay=params.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scale learning rate based on number of GPUs
        if self.multi_gpu and len(self.devices) > 1:
            # Scale learning rate based on number of GPUs (using square root scaling)
            scaled_lr = params['learning_rate'] * len(self.devices)**0.5
            print(f"Scaling learning rate: {params['learning_rate']} → {scaled_lr}")
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = scaled_lr
        
        # Multi-stage learning rate scheduler for better convergence
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimizer.param_groups[0]['lr'],
            steps_per_epoch=len(train_loader),
            epochs=self.epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # Backup plateau scheduler for final fine-tuning
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=params['learning_rate']/100
        )
        
        early_stopping_patience = patience
        early_stopping_counter = 0
        best_val_loss = float('inf')
        best_model_state = None
        train_losses, val_losses = [], []
        
        # T4 GPU Optimization with mixed precision
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        # Optimize CUDA configurations for best performance with multiple GPUs
        if self.device.type == 'cuda':
            # Set benchmark mode for improved performance with fixed input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable TF32 for faster computation when using T4 GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Track best metrics
        best_metrics = {'accuracy': 0, 'f1_positive': 0}
        last_plateau_epoch = 0
        
        try:
            for epoch in range(self.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                if len(train_loader.dataset) == 0:
                    continue
                
                # For distributed training, set the epoch for the sampler
                if self.multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                optimizer.zero_grad(set_to_none=True)
                
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    # Force data to use the right device in multi-GPU setting
                    if self.multi_gpu:
                        X_batch = X_batch.to(f'cuda:{self.devices[0]}', non_blocking=True)
                        y_batch = y_batch.to(f'cuda:{self.devices[0]}', non_blocking=True)
                    
                    # Use autocast for mixed precision training
                    with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                        # Forward pass - DataParallel will split batch across GPUs automatically
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                    
                    # Scale loss and backpropagate with mixed precision
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation for larger effective batch size
                    accumulation_steps = params.get('accumulation_steps', 4)
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        # Unscale before gradient clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get('max_grad_norm', 1.0))
                        
                        # Step optimizer and update scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        
                        # Step OneCycleLR scheduler which is called per-iteration
                        lr_scheduler.step()
                    
                    # Track loss
                    train_loss += loss.item() * X_batch.size(0)
                    
                    # Status update for multi-GPU training
                    if self.multi_gpu and (batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1):
                        print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                        
                        # Monitor GPU memory usage periodically
                        if batch_idx % 50 == 0:
                            for i in self.devices:
                                free_mem, total_mem = torch.cuda.mem_get_info(i)
                                print(f"  GPU {i} Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
                
                # Calculate average training loss
                avg_train_loss = train_loss / len(train_loader.dataset)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                # Skip validation if no validation data
                if len(val_loader.dataset) == 0:
                    avg_val_loss = float('inf')
                else:
                    val_preds = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            # Force data to use the right device in multi-GPU setting
                            if self.multi_gpu:
                                X_batch = X_batch.to(f'cuda:{self.devices[0]}', non_blocking=True)
                                y_batch = y_batch.to(f'cuda:{self.devices[0]}', non_blocking=True)
                                
                            # Use autocast for consistent precision during validation
                            with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                                y_pred = model(X_batch)
                                loss = criterion(y_pred, y_batch)
                            
                            # Store predictions and targets for metrics calculation
                            val_preds.append(torch.sigmoid(y_pred).cpu())
                            val_targets.append(y_batch.cpu())
                            
                            val_loss += loss.item() * X_batch.size(0)
                    
                    # Calculate validation metrics
                    avg_val_loss = val_loss / len(val_loader.dataset)
                    val_losses.append(avg_val_loss)
                    
                    # Calculate validation metrics every 5 epochs or on last epoch
                    if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                        val_preds_cat = torch.cat(val_preds, dim=0).numpy().flatten()
                        val_targets_cat = torch.cat(val_targets, dim=0).numpy().flatten()
                        val_preds_binary = (val_preds_cat > 0.5).astype(int)
                        
                        try:
                            accuracy = accuracy_score(val_targets_cat, val_preds_binary)
                            precision = precision_score(val_targets_cat, val_preds_binary, zero_division=0)
                            recall = recall_score(val_targets_cat, val_preds_binary, zero_division=0)
                            f1 = f1_score(val_targets_cat, val_preds_binary, zero_division=0)
                            
                            print(f'  Epoch {epoch+1}/{self.epochs} Metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, '
                                  f'Recall={recall:.4f}, F1={f1:.4f}, Val Loss={avg_val_loss:.6f}')
                            
                            # Track best metrics
                            if f1 > best_metrics['f1_positive']:
                                best_metrics['f1_positive'] = f1
                                best_metrics['accuracy'] = accuracy
                        except Exception as e:
                            print(f"  Error calculating metrics: {e}")
                    
                    # Step ReduceLROnPlateau scheduler once per epoch
                    plateau_scheduler.step(avg_val_loss)
                    
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Check for learning rate plateau
                    if epoch > 0 and epoch - last_plateau_epoch >= 10:
                        # Apply warmup restart with 10% of initial learning rate
                        restart_lr = params['learning_rate'] * 0.1
                        if self.multi_gpu:
                            restart_lr *= len(self.devices)**0.5
                        
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = restart_lr
                            
                        last_plateau_epoch = epoch
                        print(f"  Restarting learning rate at epoch {epoch+1} to {restart_lr:.7f}")
                
                # Save best model state
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    
                    # For DataParallel, we need to save the module state
                    if self.multi_gpu:
                        best_model_state = model.module.state_dict().copy()
                    else:
                        best_model_state = model.state_dict().copy()
                        
                    print(f'  Epoch {epoch+1}: New best model saved (Val Loss: {avg_val_loss:.6f})')
                else:
                    early_stopping_counter += 1
                
                # Progress reporting
                if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1 or early_stopping_counter >= early_stopping_patience:
                    print(f'  Epoch {epoch+1}/{self.epochs}: Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.7f}')
                
                # Callback for external reporting
                if epoch_callback is not None:
                    epoch_callback(epoch, avg_train_loss, avg_val_loss)
                
                # Early stopping check
                if early_stopping_counter >= early_stopping_patience:
                    print(f'  Early stopping triggered after epoch {epoch+1} '
                          f'(no improvement for {early_stopping_patience} epochs)')
                    break
            
            # Load best model before returning
            if best_model_state:
                # For returning, we need to use the base model without DataParallel
                if self.multi_gpu:
                    # Create a new base model to hold the best state
                    final_model = RecurrentAttentionModel(
                        input_dim=input_dim, 
                        hidden_dim=params['hidden_dim'], 
                        num_layers=params['num_layers'],
                        output_dim=output_dim, 
                        lookback_window=lookback_window, 
                        rnn_type=params.get('rnn_type', 'lstm'),
                        dropout_prob=params['dropout_prob'], 
                        bidirectional=params.get('bidirectional', True),
                        transformer_layers=params.get('transformer_layers', 2),
                        transformer_heads=params.get('transformer_heads', 8)
                    ).to(self.device)
                    
                    # Load the best state
                    final_model.load_state_dict(best_model_state)
                else:
                    # Just use the original model with the best state
                    model.load_state_dict(best_model_state)
                    final_model = model
                
                print(f"  Loaded best model (Val Loss: {best_val_loss:.6f})")
                print(f"  Best metrics - F1: {best_metrics['f1_positive']:.4f}, "
                      f"Accuracy: {best_metrics['accuracy']:.4f}")
                
                return final_model, train_losses, val_losses
            else:
                # If no best model was saved, return the current model
                if self.multi_gpu:
                    # For multi-GPU, return the module without DataParallel wrapper
                    return model.module, train_losses, val_losses
                else:
                    return model, train_losses, val_losses
                
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            return None, [], []

# --- Optuna Objective Function ---
def objective(trial, trainer_instance, df_full, feature_columns, num_splits, lookback_window, epochs):
    """Enhanced Optuna objective function for hyperparameter optimization targeting >90% accuracy"""
    # Define hyperparameter search space using trial object with expanded search space
    params = {
        # Model architecture parameters
        'rnn_type': trial.suggest_categorical('rnn_type', ['lstm', 'gru']),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [512, 1024, 2048]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'transformer_layers': trial.suggest_int('transformer_layers', 1, 3),
        'transformer_heads': trial.suggest_categorical('transformer_heads', [4, 8, 16]),
        'bidirectional': True,  # Fixed value
        
        # Regularization parameters
        'dropout_prob': trial.suggest_float('dropout_prob', 0.3, 0.7, step=0.1),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        
        # Training parameters
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
        'accumulation_steps': trial.suggest_categorical('accumulation_steps', [1, 2, 4, 8]),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0),
        
        # Early stopping and other parameters
        'patience': trial.suggest_int('patience', 10, 30),
        'threshold_adjustment': trial.suggest_float('threshold_adjustment', 0.4, 0.6),
        'noise_level': trial.suggest_float('noise_level', 0.001, 0.05, log=True)
    }
    
    print(f"\n[Trial {trial.number}] Testing Params: {params}")

    # Use the passed trainer_instance but update its relevant attributes if needed
    trainer_instance.feature_columns = feature_columns
    trainer_instance.epochs = epochs

    # Feature Engineering: Data Augmentation and Feature Selection
    X_columns = feature_columns.copy()
    
    # Optional: Create polynomial features for top 10 most important features
    # This will be implemented conditionally based on trial parameters
    if trial.suggest_categorical('use_poly_features', [True, False]):
        top_features = trial.suggest_categorical('top_features_count', [5, 10, 15])
        poly_degree = trial.suggest_int('poly_degree', 2, 3)
        
        # In a real implementation, we would select the top N features based on feature importance
        # Here we'll randomly select a subset as a placeholder
        if len(X_columns) > top_features:
            import random
            random.seed(trial.number)  # Use trial number for reproducibility
            selected_features = random.sample(X_columns, top_features)
            
            # Get indices of the selected features
            selected_indices = [X_columns.index(feature) for feature in selected_features]
            
            # We'll create these polynomial features during data preprocessing in fold evaluation
            params['poly_features'] = {
                'selected_indices': selected_indices,
                'degree': poly_degree
            }

    # Use walk-forward validation for each trial
    fold_metrics_list = []
    fold_num = 0
    split_generator = trainer_instance.get_walk_forward_splits(df_full)

    for train_fold_df, validation_fold_df in split_generator:
        fold_num += 1
        print(f"  [Trial {trial.number}, Fold {fold_num}] Processing...")
        
        # Data preparation with optional polynomial features and new noise level
        prep_result = trainer_instance._prepare_fold_data(
            train_fold_df, validation_fold_df, 
            ticker=f'trial_{trial.number}_fold_{fold_num}',
            batch_size=params['batch_size'], 
            noise_level=params['noise_level']
        )
        
        if prep_result is None: 
            print(f"  Fold {fold_num} prep failed. Skipping.")
            continue
            
        train_loader, val_loader, X_val, y_val, y_train_fold, _ = prep_result
        
        # Class balancing with stronger weighting for minority class
        pos_weight = None
        if y_train_fold.size > 0:
            n_pos = np.sum(y_train_fold == 1)
            n_neg = y_train_fold.size - n_pos
            
            # Enhanced class weighting strategy
            if n_pos > 0 and n_neg > 0:
                # Apply a stronger weight if imbalance is severe
                imbalance_ratio = n_neg / n_pos
                if imbalance_ratio > 3.0:
                    # Apply a stronger boosting factor for severe imbalance
                    boost_factor = 1.5
                    pos_weight = torch.tensor(imbalance_ratio * boost_factor).to(trainer_instance.device)
                else:
                    pos_weight = torch.tensor(imbalance_ratio).to(trainer_instance.device)
            else:
                pos_weight = torch.tensor(1.0).to(trainer_instance.device)

        # Train the model for this fold with expanded parameters
        fold_model, _, _ = trainer_instance.train_single_run(
            train_loader, val_loader, params, pos_weight,
            lookback_window=trainer_instance.lookback_window, 
            patience=params['patience']
        )
        
        if fold_model is None: 
            print(f"  Fold {fold_num} training failed. Skipping.")
            continue

        # Evaluate with custom threshold adjustment
        y_preds, fold_metrics = trainer_instance.evaluate(
            fold_model, X_val, y_val, params['batch_size'], 
            f'trial_{trial.number}_fold_{fold_num}'
        )
        
        # Adjust prediction threshold if needed to improve metrics
        if y_preds is not None and fold_metrics:
            # Try different classification thresholds for optimal F1/accuracy
            threshold = params['threshold_adjustment']
            if threshold != 0.5:
                y_pred_class_adjusted = (y_preds > threshold).astype(int)
                y_val_class = y_val.astype(int)
                
                try:
                    # Recalculate metrics with adjusted threshold
                    accuracy_adj = accuracy_score(y_val_class, y_pred_class_adjusted)
                    precision_adj = precision_score(y_val_class, y_pred_class_adjusted, zero_division=0)
                    recall_adj = recall_score(y_val_class, y_pred_class_adjusted, zero_division=0)
                    f1_adj = f1_score(y_val_class, y_pred_class_adjusted, zero_division=0)
                    
                    # Create adjusted metrics
                    adjusted_metrics = fold_metrics.copy()
                    adjusted_metrics['accuracy'] = accuracy_adj
                    adjusted_metrics['precision_positive'] = precision_adj
                    adjusted_metrics['recall_positive'] = recall_adj
                    adjusted_metrics['f1_positive'] = f1_adj
                    
                    print(f"  Threshold adjusted from 0.5 to {threshold}: F1 {fold_metrics['f1_positive']:.4f} → {f1_adj:.4f}")
                    
                    # Use adjusted metrics if they're better
                    if f1_adj > fold_metrics['f1_positive']:
                        fold_metrics = adjusted_metrics
                except Exception as e:
                    print(f"  Error calculating adjusted metrics: {e}")
        
        if fold_metrics:
            fold_metrics_list.append(fold_metrics)
        else:
            print(f"  Fold {fold_num} evaluation failed.")

        # Add pruning support
        if fold_num >= 1 and fold_metrics:
            intermediate_value = -fold_metrics.get('f1_positive', 0)
            trial.report(intermediate_value, fold_num)
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned after fold {fold_num}")
                raise optuna.exceptions.TrialPruned()

    if not fold_metrics_list:
        print(f"  [Trial {trial.number}] No valid metrics obtained. Returning high error.")
        return float('inf')

    # Calculate average metrics across folds
    avg_metrics = {}
    for key in fold_metrics_list[0]:
        if key != 'confusion_matrix':
            vals = [m[key] for m in fold_metrics_list if key in m and isinstance(m[key], (int, float, np.number)) and not np.isnan(m[key])]
            avg_metrics[key] = np.mean(vals) if vals else np.nan

    # Create a combined metric that weights accuracy more heavily to target >90%
    f1_positive = avg_metrics.get('f1_positive', 0.0)
    accuracy = avg_metrics.get('accuracy', 0.0)
    
    # Store all metrics in trial user attributes for later analysis
    trial.set_user_attr('avg_metrics', avg_metrics)
    
    # Weighted score prioritizing both F1 and accuracy, targeting >90% accuracy
    weighted_score = (0.4 * f1_positive) + (0.6 * accuracy)
    
    print(f"  [Trial {trial.number}] Avg F1(Pos): {f1_positive:.4f}, "
          f"Avg Acc: {accuracy:.4f}, Avg ROC AUC: {avg_metrics.get('roc_auc', np.nan):.4f}, "
          f"Weighted Score: {weighted_score:.4f}")
    
    # Return negative weighted score because Optuna minimizes
    return -weighted_score

# --- Optuna Optimization Runner ---
def run_optuna_optimization(processed_files, feature_columns, n_trials, epochs, num_splits, lookback_window):
    """Runs advanced Optuna optimization with enhanced T4 GPU utilization for >90% accuracy"""
    print("--- Starting Advanced Optuna Hyperparameter Optimization for T4 GPU ---")
    start_time = time.time()

    # Set CUDA optimizations for T4 GPU
    if torch.cuda.is_available():
        print("Configuring CUDA for optimal T4 GPU performance...")
        # Set CUDA device to device 0 (assumed to be T4)
        torch.cuda.set_device(0)
        # Enable TF32 on Ampere GPUs for performance (T4 is not Ampere, but future-proofing)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set cudnn benchmark mode for optimal performance
        torch.backends.cudnn.benchmark = True
        # Disable cudnn deterministic mode for better performance
        torch.backends.cudnn.deterministic = False
        # Cache-friendly setting for CUDA allocator
        torch.cuda.empty_cache()
        # Print available GPU memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")

    # Load data using a sequential approach instead of parallel processing
    # This avoids the pickling error on Kaggle
    print("Loading and combining data...")
    all_dfs = []
    
    for file_path in processed_files:
        try:
            print(f"Loading {os.path.basename(file_path)}")
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            all_dfs.append(df)
            print(f"Loaded {os.path.basename(file_path)}: {df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
    
    if not all_dfs:
        print("Error: No data loaded.")
        return None, None, None, None
    
    # Concatenate all dataframes
    df_full = pd.concat(all_dfs).sort_index()
    if df_full.empty:
        print("Error: Combined data empty.")
        return None, None, None, None
    
    print(f"Combined data shape: {df_full.shape}, Memory usage: {df_full.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Handle missing features
    missing = [f for f in feature_columns if f not in df_full.columns]
    if missing:
        print(f"Warning: Features missing: {missing}. Filling with 0.")
        df_full[missing] = 0.0
    
    # Advanced feature engineering
    print("Performing advanced feature engineering...")
    
    # 1. Create additional technical indicators not present
    try:
        # Check if we need to add more technical indicators
        existing_tech_indicators = {'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV'} 
        missing_indicators = existing_tech_indicators - set(df_full.columns)
        
        if missing_indicators and 'Close' in df_full.columns:
            print(f"Adding missing technical indicators: {missing_indicators}")
            
            # Import TA-Lib functionalities (if available)
            try:
                import talib
                has_talib = True
            except ImportError:
                has_talib = False
                print("TA-Lib not available, using pandas-ta instead")
                try:
                    import pandas_ta as ta
                except ImportError:
                    print("pandas-ta not available either, skipping technical indicators")
                    has_pandas_ta = False
                else:
                    has_pandas_ta = True
            
            # Create missing indicators if we have a suitable library
            if has_talib or has_pandas_ta:
                if 'RSI' in missing_indicators:
                    if has_talib:
                        df_full['RSI'] = talib.RSI(df_full['Close'], timeperiod=14)
                    elif has_pandas_ta:
                        df_full['RSI'] = df_full.ta.rsi(length=14)
                
                if 'MACD' in missing_indicators:
                    if has_talib:
                        macd, macdsignal, macdhist = talib.MACD(df_full['Close'])
                        df_full['MACD'] = macd
                        df_full['MACD_Signal'] = macdsignal
                        df_full['MACD_Hist'] = macdhist
                    elif has_pandas_ta:
                        macd = df_full.ta.macd()
                        if macd is not None:
                            df_full['MACD'] = macd['MACD_12_26_9']
                            df_full['MACD_Signal'] = macd['MACDs_12_26_9']
                            df_full['MACD_Hist'] = macd['MACDh_12_26_9']
                
                if 'BB_Upper' in missing_indicators or 'BB_Lower' in missing_indicators:
                    if has_talib:
                        bb_upper, bb_middle, bb_lower = talib.BBANDS(df_full['Close'])
                        df_full['BB_Upper'] = bb_upper
                        df_full['BB_Middle'] = bb_middle
                        df_full['BB_Lower'] = bb_lower
                    elif has_pandas_ta:
                        bbands = df_full.ta.bbands()
                        if bbands is not None:
                            df_full['BB_Upper'] = bbands['BBU_5_2.0']
                            df_full['BB_Middle'] = bbands['BBM_5_2.0']  
                            df_full['BB_Lower'] = bbands['BBL_5_2.0']
                
                if 'ATR' in missing_indicators and all(col in df_full.columns for col in ['High', 'Low', 'Close']):
                    if has_talib:
                        df_full['ATR'] = talib.ATR(df_full['High'], df_full['Low'], df_full['Close'], timeperiod=14)
                    elif has_pandas_ta:
                        df_full['ATR'] = df_full.ta.atr(length=14)
                
                if 'OBV' in missing_indicators and 'Volume' in df_full.columns:
                    if has_talib:
                        df_full['OBV'] = talib.OBV(df_full['Close'], df_full['Volume'])
                    elif has_pandas_ta:
                        df_full['OBV'] = df_full.ta.obv()
    except Exception as e:
        print(f"Error creating technical indicators: {e}")
    
    # 2. Create advanced features for better prediction
    try:
        # Add volatility features
        if 'Close' in df_full.columns:
            # Calculate daily returns
            df_full['Returns'] = df_full['Close'].pct_change()
            
            # Rolling volatility windows
            for window in [5, 10, 20]:
                col_name = f'Volatility_{window}D'
                if col_name not in df_full.columns:
                    df_full[col_name] = df_full['Returns'].rolling(window).std()
            
            # Momentum features
            for window in [5, 10, 20]:
                col_name = f'Momentum_{window}D'
                if col_name not in df_full.columns:
                    df_full[col_name] = df_full['Close'].pct_change(periods=window)
            
            # Add gap features for overnight price movement
            if 'Open' in df_full.columns:
                df_full['Gap'] = df_full['Open'] / df_full['Close'].shift(1) - 1
            
            # Add price acceleration (momentum of momentum)
            df_full['Acceleration'] = df_full['Returns'].diff()
            
            # Create market regime features using moving average crossovers
            if all(col in df_full.columns for col in ['SMA_20', 'SMA_50']):
                df_full['MA_Crossover'] = (df_full['SMA_20'] > df_full['SMA_50']).astype(int)
                
            # Create mean reversion features
            if all(col in df_full.columns for col in ['BB_Middle', 'BB_Upper', 'BB_Lower']):
                # Distance from price to Bollinger middle band normalized by band width
                df_full['BB_Position'] = (df_full['Close'] - df_full['BB_Middle']) / (df_full['BB_Upper'] - df_full['BB_Lower'])
            
            # Trend strength indicator
            if 'ADX' in df_full.columns:
                df_full['Strong_Trend'] = (df_full['ADX'] > 25).astype(int)
    except Exception as e:
        print(f"Error creating advanced features: {e}")
    
    # Replace NaN values that may have been created during feature engineering
    df_full = df_full.replace([np.inf, -np.inf], np.nan)
    df_full = df_full.ffill().bfill().fillna(0)
    
    # Update feature columns to include any new engineered features
    original_feature_count = len(feature_columns)
    feature_columns = [col for col in df_full.columns if col not in ['Date', 'Ticker', 'Company', 'Close_Next', 'Price_Change', 'Price_Increase']]
    print(f"Feature set expanded from {original_feature_count} to {len(feature_columns)} features")

    # Create an optimized trainer instance with additional configuration for T4 GPU
    trainer_instance = ModelTrainer(
        lookback_window=lookback_window,
        epochs=epochs,
        num_splits=num_splits
    )
    trainer_instance.feature_columns = feature_columns

    # Configure pruner and sampler for more efficient optimization
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Number of trials to run before pruning
        n_warmup_steps=1,    # Number of steps in a trial before pruning
        interval_steps=1     # Interval at which pruning is considered
    )
    
    # Use TPE sampler with multivariate option for better parameter exploration
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=42
    )
    
    # Create study with pruning and advanced sampling
    study = optuna.create_study(
        direction='minimize',
        study_name=f'advanced_rnn_opt_{datetime.datetime.now():%Y%m%d_%H%M}',
        pruner=pruner,
        sampler=sampler
    )
    
    # We'll create a callback to save intermediate results
    def save_best_intermediate_result(study, trial):
        """Save the best model parameters after each trial"""
        if study.best_trial.number == trial.number:
            # Only save if this is the best trial so far
            best_params = trial.params
            best_value = -trial.value  # Convert back from minimized negative score
            
            # Save the current best parameters to a file
            try:
                model_info = {
                    'trial_number': trial.number,
                    'best_params': best_params,
                    'best_score': best_value,
                    'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Check if MODEL_DIR exists, create if needed
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                
                with open(os.path.join(MODEL_DIR, 'best_intermediate_model.json'), 'w') as f:
                    json.dump(model_info, f, indent=4)
                    
                print(f"\n>>> New best trial #{trial.number} found! Score: {best_value:.4f}")
                print(f"    Saved intermediate best result to best_intermediate_model.json")
            except Exception as e:
                print(f"Error saving intermediate result: {e}")
    
    # Optimize with advanced callbacks and configuration
    try:
        study.optimize(
            lambda trial: objective(
                trial, trainer_instance, df_full, feature_columns, 
                num_splits, lookback_window, epochs
            ),
            n_trials=n_trials, 
            callbacks=[save_best_intermediate_result],
            n_jobs=1,  # Keep at 1 for GPU utilization
            gc_after_trial=True,  # Force garbage collection to free memory
            show_progress_bar=False  # We use our own progress bar
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Using best trial found so far.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        traceback.print_exc()
        if study.best_trial:
            print("Using best trial found before error occurred.")
        else:
            print("No valid trials found. Exiting.")
            return None, None, None, None

    # ... rest of the function remains the same
    
    # Optimization completed or interrupted
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n--- Optuna Optimization Complete ({int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}) ---")

    if not study.best_trial:
        print("Optuna optimization failed to find any valid trials.")
        return None, None, None, None

    best_params = study.best_trial.params
    best_value = -study.best_trial.value  # Convert back from negative
    
    print("\nTop 5 Best Trials:")
    best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]
    for i, trial in enumerate(best_trials, 1):
        if trial.value is not None:
            score = -trial.value  # Convert back from negative
            print(f"#{i}: Trial {trial.number}, Score: {score:.4f}")
            # Print key parameters only for brevity
            key_params = {k: v for k, v in trial.params.items() 
                         if k in ['rnn_type', 'hidden_dim', 'num_layers', 'transformer_layers',
                                'learning_rate', 'batch_size', 'dropout_prob']}
            print(f"    Key params: {key_params}")
    
    print("\nBest Parameters Found:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    print(f"\nBest Score: {best_value:.4f}")

    # Generate parameter importance plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        param_importances = optuna.importance.get_param_importances(study)
        
        # Plot top 10 parameters by importance
        plt.figure(figsize=(10, 6))
        params = list(param_importances.keys())[:10]  # Top 10 parameters
        importances = list(param_importances.values())[:10]
        
        plt.barh(params, importances)
        plt.xlabel('Importance')
        plt.ylabel('Parameter')
        plt.title('Parameter Importance')
        plt.tight_layout
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        plt.savefig(os.path.join(MODEL_DIR, 'parameter_importance.png'))
        print("Parameter importance plot saved to 'parameter_importance.png'")
    except Exception as e:
        print(f"Could not generate parameter importance plot: {e}")

    # Ensemble multiple top models for improved reliability
    print("\nTraining ensemble of top models for improved accuracy...")
    
    # Get top 3 trial parameters
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:3]
    top_params_list = [trial.params for trial in top_trials if trial.value is not None]
    
    if len(top_params_list) == 0:
        print("No valid top trials found. Falling back to single best model.")
        top_params_list = [best_params]
    
    # Get the last fold for final training
    split_generator = trainer_instance.get_walk_forward_splits(df_full)
    last_train_df, last_val_df = None, None
    for train_fold, val_fold in split_generator:
        last_train_df, last_val_df = train_fold, val_fold

    if last_train_df is None or last_val_df is None:
        print("Error: Could not get the last fold data for final training.")
        return None, None, best_params, None
    
    # Train models with top parameter sets
    ensemble_models = []
    for i, params in enumerate(top_params_list, 1):
        print(f"\nTraining ensemble model #{i} with trial {top_trials[i-1].number} parameters...")
        
        # Prepare data
        prep_result = trainer_instance._prepare_fold_data(
            last_train_df, last_val_df, ticker=f'ensemble_{i}',
            batch_size=params['batch_size'], noise_level=params['noise_level']
        )
        
        if prep_result is None:
            print(f"Error preparing data for ensemble model #{i}. Skipping.")
            continue
            
        train_loader, val_loader, _, _, y_train_fold, scaler = prep_result
        
        # Calculate class weights
        if y_train_fold.size > 0:
            n_pos = np.sum(y_train_fold == 1)
            n_neg = y_train_fold.size - n_pos
            
            # Enhanced class weighting strategy
            if n_pos > 0 and n_neg > 0:
                imbalance_ratio = n_neg / n_pos
                if imbalance_ratio > 3.0:
                    boost_factor = 1.5
                    pos_weight = torch.tensor(imbalance_ratio * boost_factor).to(trainer_instance.device)
                else:
                    pos_weight = torch.tensor(imbalance_ratio).to(trainer_instance.device)
            else:
                pos_weight = torch.tensor(1.0).to(trainer_instance.device)
    if ensemble_models:
        print(f"\nSaving ensemble of {len(ensemble_models)} models...")
        final_model = ensemble_models[0][0]  # Use first model as primary
        final_params = ensemble_models[0][1]  # Use first model's params as reference
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # Save each ensemble model
        for i, (model, params) in enumerate(ensemble_models, 1):
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'stock_prediction_model_ensemble_{i}.pth'))
        
        # Save the main model separately
        torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, 'stock_prediction_model.pth'))
        
        # Save the scaler
        joblib.dump({'ensemble_scaler': scaler}, os.path.join(MODEL_DIR, 'scalers.pkl'))
        
        # Format metrics for saving
        best_avg_metrics_dict = {
            'ensemble_size': len(ensemble_models),
            'best_score': best_value,
            'ensemble_model_trials': [t.number for t in top_trials[:len(ensemble_models)]]
        }
        
        # Add trial user attributes if available
        try:
            for trial in top_trials[:len(ensemble_models)]:
                if 'avg_metrics' in trial.user_attrs:
                    best_avg_metrics_dict[f'trial_{trial.number}_metrics'] = trial.user_attrs['avg_metrics']
        except Exception as e:
            print(f"Could not retrieve trial metrics: {e}")
        
        # Save comprehensive model info
        model_info = {
            'feature_columns': feature_columns,
            'lookback_window': lookback_window,
            'target_variable': 'Price_Increase',
            'is_classification': True,
            'is_ensemble': True,
            'ensemble_size': len(ensemble_models),
            'best_model_params': final_params,
            'all_ensemble_params': [p for _, p in ensemble_models],
            'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_type': 'advanced_optuna_ensemble',
            'n_trials': n_trials,
            'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'validation_type': 'walk_forward',
            'num_splits': num_splits,
            'epochs_per_run': epochs,
            'final_training_epochs': extended_epochs,
            'best_trial': study.best_trial.number,
            'best_avg_metrics': best_avg_metrics_dict,
            'data_files': [os.path.basename(f) for f in processed_files],
            'hardware': 'T4_GPU' if torch.cuda.is_available() else 'CPU',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'NA',
            'pytorch_version': torch.__version__
        }
        
        # Save model info
        with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4)
            
        print("Ensemble models and information saved successfully")
        return final_model, scaler, final_params, best_avg_metrics_dict
    else:
        print("Error: No ensemble models created. Saving single best model instead.")
        
        # Fall back to single best model approach
        # Prepare data for last fold
        prep_result = trainer_instance._prepare_fold_data(
            last_train_df, last_val_df, ticker='final_best',
            batch_size=best_params['batch_size'], noise_level=best_params.get('noise_level', 0.01)
        )
        
        if prep_result is None:
            print("Error preparing data for final model.")
            return None, None, best_params, None
            
        final_train_loader, final_val_loader, _, _, y_train_final, final_scaler = prep_result
        
        # Calculate pos_weight for class balance
        if y_train_final.size > 0:
            n_pos = np.sum(y_train_final == 1)
            n_neg = y_train_final.size - n_pos
            if n_pos > 0 and n_neg > 0:
                imbalance_ratio = n_neg / n_pos
                if n_pos > 0 and n_neg > 0:
                    imbalance_ratio = n_neg / n_pos
                    # Use a stronger weight for minority class if severe imbalance
                    if imbalance_ratio > 3.0:
                        boost_factor = 1.5
                        pos_weight_final = torch.tensor(imbalance_ratio * boost_factor).to(trainer_instance.device)
                    else:
                        pos_weight_final = torch.tensor(imbalance_ratio).to(trainer_instance.device)
                else:
                    pos_weight_final = torch.tensor(1.0).to(trainer_instance.device)
        
        # Train final model
        # Use 1.5x the epochs for final model
        extended_epochs = int(epochs * 1.5)
        trainer_instance.epochs = extended_epochs
        
        final_model, _, _ = trainer_instance.train_single_run(
            final_train_loader, final_val_loader,
            best_params, pos_weight_final, 
            lookback_window=trainer_instance.lookback_window,
            patience=best_params.get('patience', 20)  # Higher patience for final model
        )
        
        # Reset epochs to original value
        trainer_instance.epochs = epochs
        
        if final_model is None:
            print("Error: Final model training failed.")
            return None, None, best_params, None
        
        # Save model artifacts
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, 'stock_prediction_model.pth'))
        joblib.dump({'final_scaler': final_scaler}, os.path.join(MODEL_DIR, 'scalers.pkl'))
        
        best_avg_metrics_dict = {'best_score': best_value}
        
        model_info = {
            'feature_columns': feature_columns,
            'lookback_window': lookback_window,
            'target_variable': 'Price_Increase',
            'is_classification': True,
            'is_ensemble': False,
            'best_params': best_params,
            'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_type': 'advanced_optuna',
            'n_trials': n_trials,
            'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'validation_type': 'walk_forward',
            'num_splits': num_splits,
            'epochs_per_run': epochs,
            'final_training_epochs': extended_epochs,
            'best_trial': study.best_trial.number,
            'best_avg_metrics': best_avg_metrics_dict,
            'data_files': [os.path.basename(f) for f in processed_files],
            'hardware': 'T4_GPU' if torch.cuda.is_available() else 'CPU',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'NA',
            'pytorch_version': torch.__version__
        }
        
        with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4)
            
        print("Single best model and information saved successfully")
        return final_model, final_scaler, best_params, best_avg_metrics_dict

class StockPredictionEnsemble:
    """Ensemble model that combines predictions from multiple models for higher accuracy"""
    def __init__(self, model_paths, scaler, feature_columns, lookback_window, device=None):
        self.model_paths = model_paths
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.lookback_window = lookback_window
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.load_models()
        
    def load_models(self):
        """Load all models in the ensemble"""
        print(f"Loading ensemble of {len(self.model_paths)} models...")
        
        for i, path in enumerate(self.model_paths):
            try:
                # Get parameters from model info or use defaults
                if os.path.exists(os.path.join(MODEL_DIR, 'model_info.json')):
                    with open(os.path.join(MODEL_DIR, 'model_info.json'), 'r') as f:
                        model_info = json.load(f)
                    
                    if 'is_ensemble' in model_info and model_info['is_ensemble'] and 'all_ensemble_params' in model_info:
                        # Get parameters for this specific ensemble model
                        if i < len(model_info['all_ensemble_params']):
                            params = model_info['all_ensemble_params'][i]
                        else:
                            params = model_info['best_model_params']
                    else:
                        params = model_info.get('best_params', model_info.get('best_model_params', {}))
                else:
                    # Default parameters if model_info.json doesn't exist
                    params = {
                        'rnn_type': 'lstm',
                        'hidden_dim': 1024,
                        'num_layers': 2, 
                        'dropout_prob': 0.5,
                        'bidirectional': True,
                        'transformer_layers': 2,
                        'transformer_heads': 8
                    }
                
                # Initialize model with parameters
                model = RecurrentAttentionModel(
                    input_dim=len(self.feature_columns),
                    hidden_dim=params.get('hidden_dim', 1024),
                    num_layers=params.get('num_layers', 2),
                    output_dim=1,
                    lookback_window=self.lookback_window,
                    rnn_type=params.get('rnn_type', 'lstm'),
                    dropout_prob=params.get('dropout_prob', 0.5),
                    bidirectional=params.get('bidirectional', True),
                    transformer_layers=params.get('transformer_layers', 2),
                    transformer_heads=params.get('transformer_heads', 8)
                ).to(self.device)
                
                # Load trained weights
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()  # Set to evaluation mode
                self.models.append(model)
                print(f"  Loaded model from {os.path.basename(path)}")
                
            except Exception as e:
                print(f"Error loading model {path}: {e}")
                traceback.print_exc()
        
        print(f"Successfully loaded {len(self.models)} ensemble models")
    
    def predict(self, df, threshold=0.5):
        """Make prediction using ensemble of models"""
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        # Prepare input data
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            print(f"Warning: Missing features: {missing}. Filling with 0.")
            df = df.copy()
            for col in missing:
                df[col] = 0.0
                
        if len(df) < self.lookback_window:
            raise ValueError(f"Not enough data. Need {self.lookback_window}, got {len(df)}")
            
        X = df[self.feature_columns].iloc[-self.lookback_window:].values
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X)
            
        # Scale the data
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            raise ValueError(f"Error scaling input data: {e}")
            
        X_sequence = X_scaled.reshape(1, self.lookback_window, len(self.feature_columns))
        X_tensor = torch.tensor(X_sequence, dtype=torch.float32).to(self.device)
        
        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for model in self.models:
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    pred_logits = model(X_tensor).cpu()
                    pred_prob = torch.sigmoid(pred_logits).item()
                    predictions.append(pred_prob)
        
        # Ensemble the predictions (average)
        avg_prediction = np.mean(predictions)
        
        # Calculate additional ensemble statistics
        std_prediction = np.std(predictions)
        min_prediction = np.min(predictions)
        max_prediction = np.max(predictions)
        median_prediction = np.median(predictions)
        
        # Calculate ensemble confidence based on agreement among models
        agreement_level = 1.0 - (std_prediction / 0.5)  # Higher agreement = lower std
        agreement_level = max(0.0, min(1.0, agreement_level))  # Clip to [0, 1]
        
        # Return prediction and additional information
        result = {
            'prediction': int(avg_prediction > threshold),  # Binary prediction
            'probability': avg_prediction,
            'confidence': agreement_level,
            'model_agreement': {
                'std': std_prediction,
                'min': min_prediction,
                'max': max_prediction,
                'median': median_prediction,
                'raw_predictions': predictions
            }
        }
        
        return result
        
    def predict_batch(self, df, batch_size=32, threshold=0.5):
        """Make predictions for a batch of sequences"""
        if not self.models:
            raise ValueError("No models loaded in ensemble")
            
        if len(df) < self.lookback_window:
            raise ValueError(f"Not enough data. Need at least {self.lookback_window} rows.")
            
        # Prepare input data
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            print(f"Warning: Missing features: {missing}. Filling with 0.")
            df = df.copy()
            for col in missing:
                df[col] = 0.0
        
        # Prepare sequences
        X_list = []
        timestamps = []
        
        for i in range(len(df) - self.lookback_window + 1):
            seq_data = df.iloc[i:i+self.lookback_window][self.feature_columns].values
            if np.isnan(seq_data).any() or np.isinf(seq_data).any():
                seq_data = np.nan_to_num(seq_data)
            X_list.append(seq_data)
            timestamps.append(df.index[i+self.lookback_window-1])
        
        # Scale the data
        try:
            X_scaled_list = [self.scaler.transform(x) for x in X_list]
            X_array = np.array(X_scaled_list)
        except Exception as e:
            raise ValueError(f"Error scaling input data: {e}")
        
        # Create DataLoader for batch processing
        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        
        # Conditionally set num_workers based on device type
        if self.device.type == 'cuda':
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 4
            pin_memory = False
            
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        # Get predictions from all models
        all_model_preds = []
        
        for model in self.models:
            model.eval()
            model_preds = []
            
            with torch.no_grad():
                for (batch_X,) in data_loader:
                    batch_X = batch_X.to(self.device)
                    with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                        batch_logits = model(batch_X).cpu()
                        batch_probs = torch.sigmoid(batch_logits).numpy().flatten()
                    model_preds.extend(batch_probs)
            
            all_model_preds.append(model_preds)
        
        # Transpose to get predictions for each sequence from all models
        # From [num_models, num_sequences] to [num_sequences, num_models]
        all_model_preds = np.array(all_model_preds).T
        
        # Calculate ensemble predictions and statistics
        results = []
        
        for i, preds in enumerate(all_model_preds):
            avg_pred = np.mean(preds)
            std_pred = np.std(preds)
            agreement_level = 1.0 - (std_pred / 0.5)
            agreement_level = max(0.0, min(1.0, agreement_level))
            
            results.append({
                'timestamp': timestamps[i],
                'prediction': int(avg_pred > threshold),
                'probability': avg_pred,
                'confidence': agreement_level,
                'std': std_pred,
                'min': np.min(preds),
                'max': np.max(preds),
                'median': np.median(preds)
            })
        
        return pd.DataFrame(results).set_index('timestamp')

# Example Usage
if __name__ == "__main__":
    print("Running Advanced Stock Prediction with Dual T4 GPU Optimization")
    
    # Set up multi-GPU devices with explicit worker initialization
    devices = setup_gpu_devices()
    
    # Force CUDA initialization on all available GPUs 
    if torch.cuda.is_available():
        # Explicitly initialize each GPU to ensure Kaggle recognizes them
        n_gpus = torch.cuda.device_count()
        
        if n_gpus >= 2:
            print(f"Initializing {n_gpus} GPUs for Kaggle compatibility")
            
            # Create a small tensor on each device and perform an operation
            # This forces Kaggle to recognize all GPUs
            for i in range(n_gpus):
                with torch.cuda.device(i):
                    # Create a non-trivial tensor to ensure the GPU is utilized
                    x = torch.randn(1000, 1000, device=f'cuda:{i}')
                    # Perform a computation to activate the GPU
                    y = torch.matmul(x, x.t())
                    # Explicitly sync to ensure operation completes
                    torch.cuda.synchronize(i)
                    # Clean up to free memory
                    del x, y
                    torch.cuda.empty_cache()
                    print(f"  GPU {i} initialized and verified")
                    
            # Set CUDA_VISIBLE_DEVICES explicitly for Kaggle
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Verify CUDA status and optimize for T4
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # T4-specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        # Monitor GPU memory usage
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i} Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
    else:
        print("CUDA not available. Training will use CPU only.")
    
    # Check for processed data files
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Error: Directory not found: {PROCESSED_DATA_DIR}")
        exit()
        
    processed_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) 
                      if f.endswith('_processed_data.csv')]
    
    if not processed_files:
        print(f"Error: No processed files found in {PROCESSED_DATA_DIR}")
        exit()
    
    print(f"Found {len(processed_files)} processed data files.")
    
    # Load a sample file to determine feature columns
    try:
        temp_df = pd.read_csv(processed_files[0])
        feature_columns = [col for col in temp_df.columns 
                         if col not in ['Date', 'Ticker', 'Company', 'Close_Next', 
                                       'Price_Change', 'Price_Increase']]
        print(f"Using {len(feature_columns)} features from {os.path.basename(processed_files[0])}")
    except Exception as e:
        print(f"Error reading features: {e}")
        # Provide a fallback set of features if needed
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
            'Compound', 'Positive', 'Neutral', 'Negative', 'Count', 'Interest', 'FEDFUNDS',
            'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
            'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SlowK', 'SlowD', 'ADX',
            'Chaikin_AD', 'OBV', 'ATR', 'Williams_R', 'ROC', 'CCI', 'Close_Open_Ratio',
            'High_Low_Diff', 'Close_Prev_Ratio', 'Close_Lag_1', 'Volume_Lag_1',
            'Compound_Lag_1', 'Interest_Lag_1', 'FEDFUNDS_Lag_1', 'Close_Lag_3',
            'Volume_Lag_3', 'Compound_Lag_3', 'Interest_Lag_3', 'FEDFUNDS_Lag_3',
            'Close_Lag_5', 'Volume_Lag_5', 'Compound_Lag_5', 'Interest_Lag_5',
            'FEDFUNDS_Lag_5', 'Volatility_20D', 'Day_Of_Week'
        ]
        try:
            temp_df_cols = pd.read_csv(processed_files[0], nrows=1).columns
            feature_columns = [f for f in feature_columns if f in temp_df_cols]
            print(f"Warning: Using filtered default features ({len(feature_columns)})")
        except Exception:
            print(f"Warning: Using unfiltered default features ({len(feature_columns)})")

    # Configure optimization settings
    N_TRIALS = 40  # Increased from 20 for better optimization
    EPOCHS = 75    # Increased from 50 for deeper learning
    NUM_SPLITS = 3 # Keep 3 splits for walk-forward validation
    LOOKBACK_WINDOW = 30
    
    # Optionally use predefined best parameters if you want to skip optimization
    use_predefined_params = False
    
    if use_predefined_params:
        print("Using predefined best parameters instead of running optimization")
        
        # These parameters represent an optimized configuration for dual T4 GPUs
        best_params = {
            'rnn_type': 'lstm',
            'hidden_dim': 2048,
            'num_layers': 3,
            'transformer_layers': 2,
            'transformer_heads': 8,
            'bidirectional': True,
            'dropout_prob': 0.5,
            'weight_decay': 5e-4,
            'learning_rate': 0.0005,
            'batch_size': 1024,  # Doubled for dual GPUs
            'accumulation_steps': 2,  # Reduced since effective batch size is doubled
            'max_grad_norm': 1.0,
            'patience': 20,
            'threshold_adjustment': 0.45,
            'noise_level': 0.02
        }
        
        # Create a multi-GPU trainer instance with the selected parameters
        trainer = MultiGPUModelTrainer(
            lookback_window=LOOKBACK_WINDOW,
            batch_size=best_params['batch_size'],
            epochs=EPOCHS,
            num_splits=NUM_SPLITS,
            devices=devices
        )
        
        # Determine feature columns if not set above
        if not feature_columns:
            print("Error: No feature columns defined.")
            exit()
            
        trainer.feature_columns = feature_columns
        
        # Load and preprocess data
        print("Loading data...")
        all_dfs = []
        for file_path in processed_files:
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                all_dfs.append(df)
                print(f"Loaded {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        if not all_dfs:
            print("Error: No data loaded.")
            exit()
            
        df_combined = pd.concat(all_dfs).sort_index()
        print(f"Combined data shape: {df_combined.shape}")
        
        # Train on the entire dataset
        split_generator = trainer.get_walk_forward_splits(df_combined)
        last_train_df, last_val_df = None, None
        
        # Get the last fold for final training
        for train_fold, val_fold in split_generator:
            last_train_df, last_val_df = train_fold, val_fold
            
        if last_train_df is None or last_val_df is None:
            print("Error: Could not get training data.")
            exit()
            
        # Prepare data with multi-GPU support
        prep_result = trainer._prepare_fold_data(
            last_train_df, last_val_df, ticker='final_model',
            batch_size=best_params['batch_size'], noise_level=best_params.get('noise_level', 0.01)
        )
        
        if prep_result is None:
            print("Error: Data preparation failed.")
            exit()
            
        train_loader, val_loader, X_val, y_val, y_train, scaler = prep_result
        
        # Calculate class weights
        if y_train.size > 0:
            n_pos = np.sum(y_train == 1)
            n_neg = y_train.size - n_pos
            print(f"Class distribution - Positive: {n_pos} ({n_pos/y_train.size:.2%}), Negative: {n_neg} ({n_neg/y_train.size:.2%})")
            
            if n_pos > 0 and n_neg > 0:
                imbalance_ratio = n_neg / n_pos
                # Use a stronger weight for minority class if severe imbalance
                if imbalance_ratio > 3.0:
                    boost_factor = 1.5
                    pos_weight = torch.tensor(imbalance_ratio * boost_factor).to(trainer.device)
                else:
                    pos_weight = torch.tensor(imbalance_ratio).to(trainer.device)
                print(f"Using class weight: {pos_weight.item():.2f} for positive class")
            else:
                pos_weight = torch.tensor(1.0).to(trainer.device)
        else:
            pos_weight = torch.tensor(1.0).to(trainer.device)
        
        # Train the model using multi-GPU
        print(f"Training model with predefined parameters using {'multiple GPUs' if trainer.multi_gpu else 'single GPU'}...")
        model, train_losses, val_losses = trainer.train_single_run(
            train_loader, val_loader, best_params, pos_weight,
            lookback_window=LOOKBACK_WINDOW,
            patience=best_params.get('patience', 20)
        )
        
        if model is None:
            print("Error: Model training failed.")
            exit()
            
        # Save model and artifacts
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'stock_prediction_model.pth'))
        joblib.dump({'model_scaler': scaler}, os.path.join(MODEL_DIR, 'scalers.pkl'))
        
        # Save model info
        model_info = {
            'feature_columns': feature_columns,
            'lookback_window': LOOKBACK_WINDOW,
            'target_variable': 'Price_Increase',
            'is_classification': True,
            'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': best_params,
            'hardware': 'Multi_T4_GPU' if trainer.multi_gpu else ('T4_GPU' if torch.cuda.is_available() else 'CPU'),
            'num_gpus': len(devices) if devices else 0
        }
        
        with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4)
            
        print("Model training and saving complete.")
        
        # Evaluate on validation data
        print("Evaluating model...")
        y_preds, metrics = trainer.evaluate(model, X_val, y_val, best_params['batch_size'])
        
        if metrics:
            print("\nModel Evaluation Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score (Positive class): {metrics['f1_positive']:.4f}")
            print(f"Precision (Positive class): {metrics['precision_positive']:.4f}")
            print(f"Recall (Positive class): {metrics['recall_positive']:.4f}")
            print(f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")
            
            # Save metrics
            with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
                metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
                json.dump(metrics_to_save, f, indent=4)
                
            print("Metrics saved to model_metrics.json")
            
    else:
        # Run full Optuna optimization with Multi-GPU
        print("Starting full Optuna hyperparameter optimization with multi-GPU...")
        
        # Create a multi-GPU trainer for optimization
        multi_gpu_trainer = MultiGPUModelTrainer(
            lookback_window=LOOKBACK_WINDOW,
            batch_size=512,  # Initial batch size, will be overridden by Optuna
            epochs=EPOCHS,
            num_splits=NUM_SPLITS,
            devices=devices
        )
        
        # Run optimization
        best_model, final_scaler, best_params, best_metrics = run_optuna_optimization(
            processed_files=processed_files,
            feature_columns=feature_columns,
            n_trials=N_TRIALS,
            epochs=EPOCHS,
            num_splits=NUM_SPLITS,
            lookback_window=LOOKBACK_WINDOW
        )
        
        if best_model:
            print("\n--- Optimization Finished: Best model found & saved ---")
            
            # Check if ensemble models were created
            ensemble_model_paths = [
                os.path.join(MODEL_DIR, f'stock_prediction_model_ensemble_{i+1}.pth')
                for i in range(3)  # Look for up to 3 ensemble models
            ]
            
            ensemble_paths_exist = [os.path.exists(path) for path in ensemble_model_paths]
            
            if any(ensemble_paths_exist):
                # We have ensemble models
                valid_ensemble_paths = [path for i, path in enumerate(ensemble_model_paths) 
                                       if ensemble_paths_exist[i]]
                
                print(f"Found {len(valid_ensemble_paths)} ensemble models. Creating ensemble predictor.")
                
                # Create an ensemble for future predictions
                ensemble = StockPredictionEnsemble(
                    model_paths=valid_ensemble_paths,
                    scaler=final_scaler,
                    feature_columns=feature_columns,
                    lookback_window=LOOKBACK_WINDOW
                )
                
                # Save multi-GPU info in the ensemble
                with open(os.path.join(MODEL_DIR, 'ensemble_info.json'), 'w') as f:
                    json.dump({
                        'ensemble_size': len(valid_ensemble_paths),
                        'hardware': 'Multi_T4_GPU' if len(devices or []) > 1 else ('T4_GPU' if torch.cuda.is_available() else 'CPU'),
                        'num_gpus': len(devices) if devices else 0,
                        'creation_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }, f, indent=4)
                
                print("Ensemble predictor created successfully.")
            else:
                print("Single best model saved.")
        else:
            print("\n--- Optimization Finished: Failed ---")

    print("\nMulti-GPU training process complete.")
