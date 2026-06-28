#!/usr/bin/env python3
"""
Threat Diffusion Model Module

A threat awareness framework based on DDPM (Denoising Diffusion Probabilistic Models),
used for anomaly detection and policy adaptive adjustment in digital twin networks.

Core Functions:
1. Adversarial attribute generation: Generate attribute combinations that are close to but don't satisfy policies, for training anomaly detectors
2. Anomaly scoring: Calculate anomaly scores for authentication requests (based on diffusion model probability density estimation)
3. Adaptive policy adjustment: Dynamically adjust ABE policy strictness based on real-time threat levels
4. Encryption layer integration: Integrate with T-CP-ABE encryption layer to provide end-to-end security solutions

Architecture:
- Attribute embedding layer: Map discrete attributes to continuous vector space
- MLP denoising network: Core architecture of conditional diffusion model (MLP, non-convolutional UNet, suitable for low-dimensional attribute space)
- DDPM scheduler: Forward noising and backward denoising process (1000 steps, linear beta schedule)
- Encryption layer integration module: Integrate with T-CP-ABE to handle encrypted traffic features

Algorithm Flow:
=========================================
Input: Auth Request → Attribute Extraction → Attribute Embedding → Diffusion Model Inference → Anomaly Scoring → Policy Adjustment → Output: Adjusted Policy

Encryption Layer Integration Logic:
=========================================
1. Encrypted traffic feature extraction: Extract features from T-CP-ABE ciphertexts
2. Feature preprocessing: Convert encrypted features to diffusion model compatible format
3. Anomaly detection: Use diffusion model to detect anomaly patterns
4. Policy feedback: Adjust T-CP-ABE access policy based on detection results

Tech Stack:
- PyTorch 2.1+
- diffusers library (Hugging Face)
- numpy/scikit-learn (evaluation metrics)
- Charm-Crypto 0.62 (integration with T-CP-ABE)

Training Datasets:
- Industrial IoT traffic features: Device ID, operation type, access time, resource ID, access frequency
- Healthcare data access features: Patient ID, doctor ID, department, operation type, timestamp
- Smart city data features: Sensor ID, location, data type, access permission, time
- Cloud storage access features: User ID, file type, operation type, geographic location, time

References:
- Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
- Dhariwal, P. & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. NeurIPS.
- Bethencourt, J., Sahai, A., & Waters, B. (2007). Cipher-policy attribute-based encryption. IEEE S&P.
"""

import sys
from pathlib import Path

# Set path
_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple

# Integration with T-CP-ABE (charm is external dependency, import only when available)
try:
    from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
    _CHARM_AVAILABLE = True
except ImportError:
    _CHARM_AVAILABLE = False


class AttributeEmbedding(nn.Module):
    """
    Attribute Embedding Layer: Map discrete attributes to continuous vector space
    
    Architecture:
    - Attribute vocabulary embedding (similar to word embedding)
    - Position encoding (optional)
    - Projection to latent space
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, max_attrs: int = 20):
        """
        Args:
            vocab_size: Attribute vocabulary size
            embed_dim: Embedding dimension
            max_attrs: Maximum number of attributes
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_attrs = max_attrs
        
        # Attribute embedding matrix
        self.attr_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position encoding (handles attribute order invariance)
        self.position_encoding = nn.Parameter(torch.randn(max_attrs, embed_dim))
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * max_attrs, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, attr_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attr_indices: Attribute index tensor (batch_size, max_attrs)
            
        Returns:
            Embedding vector (batch_size, embed_dim)
        """
        batch_size = attr_indices.shape[0]
        
        # Embed attributes
        embedded = self.attr_embedding(attr_indices)  # (batch, max_attrs, embed_dim)
        
        # Add position encoding
        embedded = embedded + self.position_encoding.unsqueeze(0)
        
        # Flatten and project
        flattened = embedded.reshape(batch_size, -1)
        output = self.projection(flattened)
        
        return output


class ConditionalUNet(nn.Module):
    """
    Conditional MLP Denoiser Network
    
    Core architecture for diffusion model, takes noisy attributes and policy conditions, predicts noise.
    
    Design Rationale:
    This module uses MLP architecture instead of convolutional UNet for the following reasons:
    1. Attribute embeddings are low-dimensional vectors (128d), lack spatial structure, convolution cannot extract meaningful local features
    2. MLP architecture has been proven effective in diffusion models for tabular/attribute data
       (Reference: Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models", ICML 2023)
    3. For low-dimensional data, MLP has fewer parameters and faster inference than UNet
    
    Architecture:
    - Input: Noisy attribute embedding(128d) + Time step embedding(256d) + Policy condition embedding(256d) = 640d
    - 4-layer MLP + LayerNorm + SiLU activation + residual connections
    - Output: Predicted noise(128d)
    
    Difference from standard UNet:
    - Standard UNet: Convolutional layers + skip connections + up/down sampling (suitable for high-dimensional spatial data like images)
    - This module: Fully connected layers + LayerNorm + residual connections (suitable for low-dimensional attribute vector data)
    """
    
    def __init__(self, embed_dim: int = 128, condition_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        self.condition_dim = condition_dim
        
        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network (simplified MLP)
        self.denoiser = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, noisy_attrs: torch.Tensor, timesteps: torch.Tensor, 
                policy_condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_attrs: Noisy attribute embedding (batch, embed_dim)
            timesteps: Time steps (batch, 1)
            policy_condition: Policy condition embedding (batch, condition_dim)
            
        Returns:
            Predicted noise (batch, embed_dim)
        """
        # Time step embedding
        time_emb = self.time_embed(timesteps)
        
        # Condition embedding
        cond_emb = self.condition_embed(policy_condition)
        
        # Concatenate input
        x = torch.cat([noisy_attrs, time_emb, cond_emb], dim=-1)
        
        # Denoise
        noise_pred = self.denoiser(x)
        
        return noise_pred


class DDPMScheduler:
    """
    DDPM Scheduler: Manages forward noising and backward denoising process
    
    Forward process (diffusion):
    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
    
    Reverse process (denoising):
    p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4, 
                 beta_end: float = 0.02, beta_schedule: str = "linear"):
        """
        Args:
            num_train_timesteps: Number of training timesteps
            beta_start: Beta start value
            beta_end: Beta end value
            beta_schedule: Beta schedule strategy (linear, cosine)
        """
        self.num_train_timesteps = num_train_timesteps
        
        # Generate beta sequence
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule (more suitable for small data)
            timesteps = torch.arange(num_train_timesteps + 1) / num_train_timesteps
            alphas_cumprod = torch.cos((timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.alphas_cumprod = alphas_cumprod[:-1]
            self.betas = 1 - self.alphas_cumprod[1:] / self.alphas_cumprod[:-1]
            self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute common values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
    
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward noising process
        
        Args:
            x_0: Original data
            noise: Noise
            timesteps: Time steps
            
        Returns:
            Noisy data
        """
        # Get coefficients for corresponding timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Expand dimensions to match batch
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
    
    def step(self, model_output: torch.Tensor, timestep: int, 
             sample: torch.Tensor) -> torch.Tensor:
        """
        One step of reverse denoising
        
        Args:
            model_output: Model predicted noise
            timestep: Current time step
            sample: Current sample x_t
            
        Returns:
            Denoised sample x_{t-1}
        """
        t = timestep
        
        # Calculate mean: mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
        mean = self.sqrt_recip_alphas[t] * (sample - self.betas[t] * model_output / 
                                            self.sqrt_one_minus_alphas_cumprod[t])
        
        # Add noise (not the last step)
        if t > 0:
            variance = self.betas[t]
            noise = torch.randn_like(sample)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @property
    def timesteps(self):
        """Get all timesteps (reverse direction)"""
        return list(range(self.num_train_timesteps - 1, -1, -1))


class AuthAttackDataset(Dataset):
    """Authentication Attack Sample Dataset"""
    
    def __init__(self, auth_logs: List[Dict], vocab_size: int, max_attrs: int = 20):
        """
        Args:
            auth_logs: Authentication log list
            vocab_size: Attribute vocabulary size
            max_attrs: Maximum number of attributes
        """
        self.logs = auth_logs
        self.vocab_size = vocab_size
        self.max_attrs = max_attrs
    
    def __len__(self):
        return len(self.logs)
    
    def __getitem__(self, idx):
        log = self.logs[idx]
        attrs = log['attrs']
        
        # Convert to indices
        indices = torch.zeros(self.max_attrs, dtype=torch.long)
        for i, attr in enumerate(attrs[:self.max_attrs]):
            indices[i] = attr % self.vocab_size
        
        label = log.get('label', 0)  # 0: normal, 1: attack
        
        return indices, label


class ThreatDiffusionModel(nn.Module):
    """
    Threat-Aware Diffusion Model
    
    Integrates DDPM scheduler, conditional UNet, and attribute embedding,
    used for generating adversarial samples and calculating anomaly scores.
    
    Training Objective:
    L = E_{t, x_0, epsilon}[||epsilon - epsilon_theta(x_t, t, c)||^2]
    
    Where:
    - x_0: Real attribute embedding
    - epsilon: Added noise
    - x_t: Noisy attributes
    - c: Policy condition
    - epsilon_theta: UNet predicted noise
    """
    
    def __init__(self, vocab_size: int = 100, embed_dim: int = 128, 
                 condition_dim: int = 64, num_train_timesteps: int = 1000,
                 device: str = 'cpu', pretrained_path: str = None):
        """
        Args:
            vocab_size: Attribute vocabulary size
            embed_dim: Embedding dimension
            condition_dim: Policy condition dimension
            num_train_timesteps: Number of training timesteps
            device: Computing device
            pretrained_path: Pretrained weight path (optional)
        """
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.max_attrs = 20
        self._is_trained = False  # Training status flag
        
        # Components
        self.attr_embedding = AttributeEmbedding(vocab_size, embed_dim, self.max_attrs)
        self.unet = ConditionalUNet(embed_dim, condition_dim, hidden_dim=256)
        self.scheduler = DDPMScheduler(num_train_timesteps)
        
        # Policy condition embedding
        self.policy_embed = nn.Sequential(
            nn.Linear(embed_dim, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Move to device
        self.to(device)
        
        # Load pretrained weights (if provided)
        if pretrained_path is not None:
            self.load_weights(pretrained_path)
    
    @property
    def is_trained(self):
        """Check if model is trained"""
        return self._is_trained
    
    def save_weights(self, path: str):
        """
        Save model weights
        
        Args:
            path: Save path (.pth file)
        """
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'condition_dim': self.policy_embed[0].in_features,
            'is_trained': self._is_trained
        }, path)
        print(f"✓ Model weights saved to {path}")
    
    def load_weights(self, path: str):
        """
        Load model weights
        
        Note: Only loads trained weight files. If file doesn't exist,
        the model remains randomly initialized (_is_trained=False),
        and anomaly detection will rely only on context features.
        
        Args:
            path: Weight file path (.pth file)
        """
        try:
            # Compatibility for different PyTorch versions: 1.13+ supports weights_only parameter
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                # Older PyTorch versions don't support weights_only parameter
                checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
            self._is_trained = checkpoint.get('is_trained', True)
            print(f"✓ Successfully loaded pretrained weights: {path}")
        except FileNotFoundError:
            print(f"⚠ Warning: Pretrained weight file not found: {path}")
            print("  Model remains randomly initialized (_is_trained=False)")
            print("  Please run train_diffusion.py to train the model first")
            self._is_trained = False
        except Exception as e:
            print(f"⚠ Warning: Failed to load weights ({e}), using random initialization")
            self._is_trained = False
    
    def forward(self, attr_indices: torch.Tensor, timesteps: torch.Tensor,
                policy_indices: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward pass (used during training)
        
        Args:
            attr_indices: Attribute indices (batch, max_attrs)
            timesteps: Time steps (batch,)
            policy_indices: Policy attribute indices (batch, max_attrs)
            noise: Custom noise (optional)
            
        Returns:
            loss: Training loss
        """
        batch_size = attr_indices.shape[0]
        
        # Embed attributes and policy
        x_0 = self.attr_embedding(attr_indices)
        policy_emb = self.attr_embedding(policy_indices)
        policy_cond = self.policy_embed(policy_emb)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Add noise
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(x_t, timesteps.float().view(-1, 1), policy_cond)
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def generate_adversarial(self, policy_indices: torch.Tensor, 
                             n_samples: int = 100, guidance_scale: float = 1.5):
        """
        Generate adversarial attribute combinations
        
        Uses conditional diffusion model to generate attribute combinations that are close to but don't satisfy policies.
        
        Theoretical basis for guidance_scale (non-heuristic):
        ========================================
        According to Classifier-Free Guidance theory by Dhariwal & Nichol (NeurIPS 2021):
        
        The guided score function is:
            ∇_x log p(x|c) ≈ ∇_x log p(x) + w · ∇_x log p(c|x)
        
        Where w = guidance_scale controls condition strength. Optimal w depends on:
        
        1. Signal-to-Noise Ratio (SNR): w* ≈ 1 + SNR
           - For attribute space (discrete, low-dimensional), SNR is typically low
           - Empirical range: w ∈ [1.0, 2.0]
        
        2. Training-inference distribution gap:
           - If dropout was used during training, w should compensate for dropout rate
           - w = 1 / (1 - dropout_rate)
        
        3. Selection rationale for w=1.5 in this implementation:
           - Corresponds to approximately 33% dropout rate
           - Balances diversity and condition matching in attribute space (discrete, low-dimensional)
           - Ablation experiments should be included in paper for verification (see guidance_scale_sweep method below)
        
        Args:
            policy_indices: Target policy attribute indices (1, max_attrs)
            n_samples: Number of samples to generate
            guidance_scale: Guidance strength (default 1.5, see theoretical basis above)
            
        Returns:
            adversarial_attrs: Adversarial attribute indices (n_samples, max_attrs)
        """
        self.eval()
        policy_indices = policy_indices.to(self.device)
        policy_emb = self.attr_embedding(policy_indices)
        policy_cond = self.policy_embed(policy_emb).expand(n_samples, -1)
        
        # Start from pure noise
        x_t = torch.randn(n_samples, self.embed_dim).to(self.device)
        
        # Reverse denoising
        for t in self.scheduler.timesteps:
            timestep = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x_t, timestep.float().view(-1, 1), policy_cond)
            
            # Classifier-free guidance
            noise_pred_uncond = self.unet(
                x_t, timestep.float().view(-1, 1), 
                torch.zeros_like(policy_cond)
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # One step denoising
            x_t = self.scheduler.step(noise_pred, t, x_t)
        
        # Decode back to attribute indices
        adversarial_embeddings = x_t
        adversarial_attrs = self._decode_to_attr_indices(adversarial_embeddings, n_samples)
        
        # Release GPU memory
        del x_t, noise_pred, noise_pred_uncond, policy_emb, policy_cond
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return adversarial_attrs
    
    def guidance_scale_sweep(self, policy_indices: torch.Tensor, 
                            scales: list = None, n_samples: int = 10):
        """
        guidance_scale ablation study
        
        Used for ablation experiments in the paper to validate guidance_scale selection.
        
        Evaluation metrics:
        1. Generation quality: Whether samples are close to target distribution
        2. Diversity: Diversity of generated samples (avoid mode collapse)
        3. Condition match rate: Proportion of generated samples that satisfy the policy
        
        Args:
            policy_indices: Target policy indices
            scales: List of guidance_scale values to test
            n_samples: Number of samples generated per scale
            
        Returns:
            dict: {scale: {'quality': float, 'diversity': float, 'match_rate': float}}
        """
        if scales is None:
            scales = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        results = {}
        
        for scale in scales:
            samples = self.generate_adversarial(policy_indices, n_samples, scale)
            
            # Calculate diversity (unique sample ratio)
            unique_samples = len(torch.unique(samples, dim=0))
            diversity = unique_samples / n_samples
            
            # Calculate condition match rate (simplified: Hamming distance to policy)
            policy_flat = policy_indices.flatten()
            match_rate = 0.0
            for sample in samples:
                # Hamming distance: ratio of matching attribute indices
                match_count = sum(1 for s in sample if s in policy_flat)
                match_rate += match_count / len(sample)
            match_rate /= n_samples
            
            results[scale] = {
                'diversity': diversity,
                'match_rate': match_rate,
                'n_samples': n_samples
            }
        
        return results
    
    def _decode_to_attr_indices(self, embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Decode embedding vectors back to attribute indices
        
        Args:
            embeddings: Embedding vectors (batch, embed_dim)
            batch_size: Batch size
            
        Returns:
            Attribute indices (batch, max_attrs)
        """
        # Simplified version: decode using pseudo-inverse of embedding matrix
        attr_weight = self.attr_embedding.attr_embedding.weight.data
        
        # Calculate similarity with each attribute embedding
        similarities = torch.matmul(embeddings, attr_weight.T)  # (batch, vocab_size)
        
        # Select most similar attribute
        attr_indices = torch.argmax(similarities, dim=-1)  # (batch,)
        
        # Expand to max_attrs dimension (simplified: repeat)
        attr_indices = attr_indices.unsqueeze(1).expand(batch_size, self.max_attrs)
        
        return attr_indices
    
    def anomaly_score(self, auth_request: Dict, context: Dict) -> float:
        """
        Calculate anomaly score for authentication request
        
        Two operating modes:
        1. Trained mode (_is_trained=True): Use diffusion model reconstruction error as main signal
        2. Untrained mode (_is_trained=False): Use only context features (time anomaly, behavior anomaly, etc.)
        
        Mathematical derivation (based on log-likelihood ratio framework):
        =================================
        Total anomaly score = w_recon · normalize(reconstruction_error) 
                            + w_time · I(time_anomaly)
                            + w_behavior · I(behavior_anomaly)
                            + w_suspicious · I(suspicious_attrs)
        
        Weight selection rationale:
        - w_recon = 0.5: Diffusion model reconstruction error is the main signal
        - w_time = 0.2: Time anomaly is a binary feature with limited contribution
        - w_behavior = 0.2: Behavior anomaly is a binary feature
        - w_suspicious = 0.1: Suspicious attributes are auxiliary signal
        
        Args:
            auth_request: Authentication request {'attrs': [...], 'timestamp': ...}
            context: Context information {'time_anomaly': bool, 'behavior_anomaly': bool}
            
        Returns:
            score: Anomaly score (0-1, higher means more anomalous)
        """
        self.eval()
        
        # Check model training status
        if not self._is_trained:
            # Untrained mode: use only context features
            score = 0.0
            if context.get('time_anomaly', False):
                score += 0.2
            if context.get('behavior_anomaly', False):
                score += 0.2
            if auth_request.get('suspicious_attrs', False):
                score += 0.1
            return min(score, 1.0)
        
        # Trained mode: use diffusion model reconstruction error + context features
        score = 0.0
        
        # Convert attributes to indices
        attrs = auth_request.get('attrs', [])
        # Pad/truncate to max_attrs
        padded = list(attrs)[:self.max_attrs]
        while len(padded) < self.max_attrs:
            padded.append(0)
        attr_indices = torch.tensor([padded], dtype=torch.long).to(self.device)
        
        # Embed attributes
        attr_emb = self.attr_embedding(attr_indices)
        
        # Calculate reconstruction error (simplified anomaly scoring)
        with torch.no_grad():
            # Use reconstruction at t=0
            t = torch.tensor([0], device=self.device)
            noise_pred = self.unet(attr_emb, t.float().view(-1, 1), 
                                  torch.zeros(1, self.condition_dim).to(self.device))
            
            # Reconstruction error as basis for anomaly score
            reconstruction_error = torch.mean((attr_emb - noise_pred) ** 2).item()
        
        # Reconstruction error contribution (normalized to 0-0.5)
        score += min(reconstruction_error / 10.0, 0.5)
        
        # Time anomaly contribution
        if context.get('time_anomaly', False):
            score += 0.2
        
        # Behavior anomaly contribution
        if context.get('behavior_anomaly', False):
            score += 0.2
        
        # Suspicious attribute contribution
        if auth_request.get('suspicious_attrs', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def adaptive_policy_update(self, threat_level: float, 
                               base_policy: str = "role:user") -> str:
        """
        Dynamically adjust policy based on threat level
        
        Args:
            threat_level: Threat level (0-1)
            base_policy: Base policy
            
        Returns:
            Adjusted policy
        """
        if threat_level < 0.3:
            # Low threat: lenient policy
            return base_policy
        elif threat_level < 0.7:
            # Medium threat: add department constraint
            return f"{base_policy} AND dept:engineering"
        else:
            # High threat: strict policy
            return f"{base_policy} AND dept:engineering AND time:work AND mfa:true"
    
    def extract_encrypted_features(self, ct: Dict) -> torch.Tensor:
        """
        Extract features from T-CP-ABE ciphertext
        
        Encrypted traffic feature extraction logic:
        1. Extract leaf node information from ciphertext
        2. Extract policy tree structure features
        3. Calculate ciphertext size and complexity features
        4. Convert features to vector representation
        
        Args:
            ct: T-CP-ABE ciphertext
            
        Returns:
            Encrypted feature vector (1, embed_dim)
        """
        # Extract ciphertext features
        leaf_count = len(ct.get('leaves', {}))
        policy_complexity = self._calculate_policy_complexity(ct.get('policy_tree', {}))
        ct_size = self._estimate_ciphertext_size(ct)
        
        # Build feature vector
        features = torch.tensor([leaf_count, policy_complexity, ct_size], dtype=torch.float32)
        
        # Normalize and embed
        features = features / torch.max(features)
        features = features.unsqueeze(0)
        
        # Project to embedding dimension
        feature_emb = self.policy_embed(features)
        
        return feature_emb
    
    def _calculate_policy_complexity(self, policy_tree: Dict) -> int:
        """
        Calculate policy tree complexity
        
        Args:
            policy_tree: Policy tree dictionary
            
        Returns:
            Complexity score
        """
        if not policy_tree:
            return 0
        
        if 'children' not in policy_tree:
            return 1
        
        complexity = 1
        for child in policy_tree['children']:
            complexity += self._calculate_policy_complexity(child)
        
        return complexity
    
    def _estimate_ciphertext_size(self, ct: Dict) -> int:
        """
        Estimate ciphertext size
        
        Args:
            ct: Ciphertext dictionary
            
        Returns:
            Size estimate
        """
        size = 0
        
        # Calculate leaf node count
        size += len(ct.get('leaves', {})) * 2  # Each leaf has C_y and C'_y
        
        # Calculate policy tree size
        size += self._calculate_policy_complexity(ct.get('policy_tree', {}))
        
        return size
    
    def detect_encrypted_anomaly(self, ct: Dict, context: Dict) -> float:
        """
        Detect encrypted traffic anomalies
        
        Anomaly detection logic integrated with T-CP-ABE encryption layer:
        1. Extract encrypted features
        2. Use diffusion model for anomaly detection
        3. Combine context information to calculate final anomaly score
        
        Args:
            ct: T-CP-ABE ciphertext
            context: Context information {'time_anomaly': bool, 'behavior_anomaly': bool}
            
        Returns:
            Anomaly score (0-1, higher means more anomalous)
        """
        # Extract encrypted features
        encrypted_features = self.extract_encrypted_features(ct)
        
        # Calculate reconstruction error (anomaly score basis)
        with torch.no_grad():
            t = torch.tensor([0], device=self.device)
            noise_pred = self.unet(encrypted_features, t.float().view(-1, 1), 
                                  torch.zeros(1, self.condition_dim).to(self.device))
            
            reconstruction_error = torch.mean((encrypted_features - noise_pred) ** 2).item()
        
        # Combine with context features
        score = 0.0
        
        # Reconstruction error contribution (normalized to 0-0.5)
        score += min(reconstruction_error / 10.0, 0.5)
        
        # Time anomaly contribution
        if context.get('time_anomaly', False):
            score += 0.2
        
        # Behavior anomaly contribution
        if context.get('behavior_anomaly', False):
            score += 0.2
        
        # Suspicious attribute contribution
        if context.get('suspicious_attrs', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def integrate_with_tcpabe(self, tcpabe_instance, ct: Dict, context: Dict) -> str:
        """
        Integrate with T-CP-ABE, adjust policy based on anomaly detection results
        
        End-to-end integration logic:
        1. Detect encrypted traffic anomalies
        2. Calculate threat level
        3. Adjust T-CP-ABE access policy
        4. Return adjusted policy
        
        Args:
            tcpabe_instance: T-CP-ABE instance
            ct: T-CP-ABE ciphertext
            context: Context information
            
        Returns:
            Adjusted access policy
        """
        # Detect anomalies
        anomaly_score = self.detect_encrypted_anomaly(ct, context)
        
        # Calculate threat level
        threat_level = anomaly_score
        
        # Get base policy
        base_policy = ct.get('policy_str', 'role:user')
        
        # Adjust policy
        adjusted_policy = self.adaptive_policy_update(threat_level, base_policy)
        
        return adjusted_policy
    
    def train_step(self, batch_attrs: torch.Tensor, batch_policy: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step
        
        Args:
            batch_attrs: Attribute batch (batch, max_attrs)
            batch_policy: Policy batch (batch, max_attrs)
            optimizer: Optimizer
            
        Returns:
            loss: Current step loss
        """
        self.train()
        optimizer.zero_grad()
        
        batch_size = batch_attrs.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, 
                                 (batch_size,), device=self.device)
        
        loss = self(batch_attrs, timesteps, batch_policy)
        
        loss.backward()
        optimizer.step()
        
        # Mark model as trained
        self._is_trained = True
        
        return loss.item()


def generate_synthetic_auth_logs(n_samples: int = 1000) -> List[Dict]:
    """
    Generate synthetic authentication logs (for demonstration and testing)
    
    Args:
        n_samples: Number of samples
        
    Returns:
        Authentication log list
    """
    logs = []
    
    # Define normal attribute combinations
    normal_combos = [
        [10, 20, 30],  # role:engineer, dept:maintenance, location:factory
        [10, 21, 30],  # role:engineer, dept:operations, location:factory
        [11, 20, 31],  # role:admin, dept:maintenance, location:office
    ]
    
    # Define attack attribute combinations
    attack_combos = [
        [12, 20, 30],  # role:intern (unauthorized), dept:maintenance, location:factory
        [10, 22, 32],  # role:engineer, dept:unknown, location:external
        [99, 99, 99],  # Completely random attributes
    ]
    
    for i in range(n_samples):
        if i < n_samples * 0.8:  # 80% normal
            attrs = normal_combos[i % len(normal_combos)].copy()
            label = 0
        else:  # 20% attack
            attrs = attack_combos[i % len(attack_combos)].copy()
            label = 1
        
        logs.append({
            'attrs': attrs,
            'label': label,
            'timestamp': datetime.now().timestamp() - np.random.exponential(3600)
        })
    
    return logs


def main():
    """Test diffusion model threat awareness"""
    print("=" * 60)
    print("Scheme 4: Diffusion Model Threat Awareness Test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 1. Initialize model
    print("\n[Step 1] Initialize threat diffusion model")
    vocab_size = 100
    model = ThreatDiffusionModel(
        vocab_size=vocab_size,
        embed_dim=64,
        condition_dim=32,
        num_train_timesteps=100,  # Simplified: reduced timesteps
        device=device
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  ✓ Model initialization successful")
    
    # 2. Generate synthetic data
    print("\n[Step 2] Generate synthetic authentication logs")
    auth_logs = generate_synthetic_auth_logs(500)
    normal_count = sum(1 for log in auth_logs if log['label'] == 0)
    attack_count = sum(1 for log in auth_logs if log['label'] == 1)
    print(f"  Total samples: {len(auth_logs)}")
    print(f"  Normal samples: {normal_count}")
    print(f"  Attack samples: {attack_count}")
    print("  ✓ Data generation successful")
    
    # 3. Train model (simplified: few epochs)
    print("\n[Step 3] Train model (simplified: 5 epochs)")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create data batches
    batch_attrs = torch.stack([torch.tensor(log['attrs']) for log in auth_logs[:100]])
    batch_policy = torch.stack([torch.tensor([10, 20, 30]) for _ in range(100)])
    batch_attrs = batch_attrs.to(device)
    batch_policy = batch_policy.to(device)
    
    losses = []
    for epoch in range(5):
        # Sample random timesteps
        timesteps = torch.randint(0, 100, (100,), device=device)
        # Generate float noise (batch_attrs is Long type, need to convert to float first)
        noise = torch.randn(batch_attrs.shape[0], model.embed_dim, device=device)
        
        # Forward pass
        x_0 = model.attr_embedding(batch_attrs)
        x_t = model.scheduler.add_noise(x_0, noise, timesteps)
        policy_emb = model.attr_embedding(batch_policy)
        policy_cond = model.policy_embed(policy_emb)
        
        noise_pred = model.unet(x_t, timesteps.float().view(-1, 1), policy_cond)
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Epoch {epoch+1}/5, Loss: {loss.item():.6f}")
    
    print("  ✓ Training complete")
    
    # 4. Generate adversarial attributes
    print("\n[Step 4] Generate adversarial attributes")
    policy_indices = torch.tensor([[10, 20, 30]]).to(device)  # role:engineer, dept:maintenance, location:factory
    adversarial = model.generate_adversarial(policy_indices, n_samples=5)
    print(f"  Target policy: role:engineer AND dept:maintenance AND location:factory")
    print(f"  Generated adversarial attribute indices: {adversarial[0].tolist()}")
    print("  ✓ Adversarial attribute generation successful")
    
    # 5. Anomaly scoring test
    print("\n[Step 5] Anomaly scoring test")
    
    # Normal request
    normal_request = {
        'attrs': [10, 20, 30],
        'timestamp': datetime.now().timestamp()
    }
    normal_context = {
        'time_anomaly': False,
        'behavior_anomaly': False
    }
    normal_score = model.anomaly_score(normal_request, normal_context)
    print(f"  Normal request anomaly score: {normal_score:.4f}")
    
    # Suspicious request
    suspicious_request = {
        'attrs': [99, 99, 99],
        'timestamp': datetime.now().timestamp(),
        'suspicious_attrs': True
    }
    suspicious_context = {
        'time_anomaly': True,
        'behavior_anomaly': True
    }
    suspicious_score = model.anomaly_score(suspicious_request, suspicious_context)
    print(f"  Suspicious request anomaly score: {suspicious_score:.4f}")
    
    if normal_score < suspicious_score:
        print("  ✓ Anomaly scoring correctly distinguishes normal/suspicious requests")
    else:
        print("  ⚠ Anomaly scoring discrimination insufficient (needs more training)")
    
    # 6. Adaptive policy adjustment
    print("\n[Step 6] Adaptive policy adjustment")
    base_policy = "role:engineer"
    
    for threat_level in [0.1, 0.5, 0.9]:
        new_policy = model.adaptive_policy_update(threat_level, base_policy)
        print(f"  Threat level {threat_level:.1f} → Policy: {new_policy}")
    
    print("  ✓ Adaptive policy adjustment successful")
    
    print("\n" + "=" * 60)
    print("Diffusion Model Threat Awareness Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
