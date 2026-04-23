"""
SELF-PRUNING NEURAL NETWORK - FIXED FOR WINDOWS GPU
Complete working solution - No external dependencies issues
Run: python final_solution.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# PART 1: PRUNABLE LINEAR LAYER
# ============================================================================

class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gates for each weight.
    Each weight has a gate that determines if it gets pruned.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight initialization (Xavier/He)
        stdv = 1.0 / (in_features ** 0.5)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * stdv)
        
        # Gate scores - initialized so sigmoid gives ~0.7-0.8
        # This means most weights start as "kept"
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * 1.5)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def get_gates(self):
        """Convert gate_scores to gates in [0,1] range"""
        return torch.sigmoid(self.gate_scores)
    
    def get_pruned_weights(self):
        """Apply gates to weights"""
        gates = self.get_gates()
        return self.weight * gates
    
    def forward(self, x):
        """Forward pass using pruned weights"""
        pruned_weights = self.get_pruned_weights()
        return F.linear(x, pruned_weights, self.bias)
    
    def get_sparsity(self, threshold=0.01):
        """Calculate percentage of pruned weights in this layer"""
        gates = torch.sigmoid(self.gate_scores)
        return (gates < threshold).float().mean().item()
    
    def get_gate_values(self):
        """Return all gate values for analysis"""
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()


# ============================================================================
# PART 2: SELF-PRUNING NETWORK
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    Complete neural network using prunable layers
    Architecture: 3072 → 1024 → 512 → 256 → 10
    """
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # All linear layers are prunable!
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4 = PrunableLinear(256, 10)
        
        # Store all prunable layers for easy access
        self.prunable_layers = [self.fc1, self.fc2, self.fc3, self.fc4]
    
    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        return x
    
    def get_all_gates(self):
        """Collect all gate values from all prunable layers"""
        all_gates = []
        for layer in self.prunable_layers:
            all_gates.append(layer.get_gate_values())
        return torch.cat(all_gates)
    
    def compute_sparsity_loss(self):
        """
        Calculate L1 norm of all gates (sum of gate values)
        This encourages gates to become exactly zero
        """
        total_gate_sum = 0.0
        total_gate_count = 0
        
        for layer in self.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores)
            total_gate_sum += gates.sum()
            total_gate_count += gates.numel()
        
        # Return average gate value (normalized)
        return total_gate_sum / total_gate_count
    
    def compute_sparsity_percentage(self, threshold=0.01):
        """Calculate overall network sparsity percentage"""
        all_gates = self.get_all_gates()
        return (all_gates < threshold).float().mean().item() * 100
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_active_parameters(self, threshold=0.01):
        """Count parameters that survived pruning"""
        active = 0
        for layer in self.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores)
            active += (gates > threshold).sum().item()
        return active


# ============================================================================
# PART 3: DATA LOADING
# ============================================================================

def get_data_loaders(batch_size=256):
    """Load CIFAR-10 dataset"""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split train into train/validation (90/10)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    print(f"✓ Data loaded: {len(trainset)} train, {len(valset)} val, {len(testset)} test")
    return trainloader, valloader, testloader


# ============================================================================
# PART 4: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, trainloader, optimizer, criterion, lambda_sparsity, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_sparsity_loss = 0
    
    pbar = tqdm(trainloader, desc="Training", leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        class_loss = criterion(outputs, targets)
        
        # Sparsity regularization
        sparsity_loss = model.compute_sparsity_loss()
        
        # Total loss = classification loss + lambda * sparsity loss
        loss = class_loss + lambda_sparsity * sparsity_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    n = len(trainloader)
    return (total_loss/n, total_class_loss/n, total_sparsity_loss/n)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(dataloader), 100.*correct/total


# ============================================================================
# PART 5: MAIN TRAINING LOOP
# ============================================================================

def train_model(lambda_sparsity, num_epochs=15):
    """Train model with given sparsity penalty"""
    
    print(f"\n{'='*60}")
    print(f"Training with λ = {lambda_sparsity}")
    print(f"{'='*60}")
    
    # Load data
    trainloader, valloader, testloader = get_data_loaders(batch_size=256)
    
    # Initialize model
    model = SelfPruningNetwork().to(device)
    total_params = model.count_parameters()
    print(f"✓ Model has {total_params:,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [],
        'sparsity': [], 'class_loss': [], 'sparse_loss': []
    }
    
    best_accuracy = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, class_loss, sparse_loss = train_epoch(
            model, trainloader, optimizer, criterion, lambda_sparsity, device
        )
        
        # Validate
        val_loss, val_accuracy = evaluate(model, valloader, criterion, device)
        
        # Calculate sparsity
        sparsity = model.compute_sparsity_percentage()
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['sparsity'].append(sparsity)
        history['class_loss'].append(class_loss)
        history['sparse_loss'].append(sparse_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"  Sparsity: {sparsity:.1f}% | λ: {lambda_sparsity}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'best_model_lambda_{lambda_sparsity}.pt')
            print(f"  ✓ Saved best model (acc: {val_accuracy:.2f}%)")
    
    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time/60:.1f} minutes")
    
    # Final test evaluation
    test_loss, test_accuracy = evaluate(model, testloader, criterion, device)
    final_sparsity = model.compute_sparsity_percentage()
    
    # Calculate compression
    active_params = model.count_active_parameters()
    compression = total_params / active_params if active_params > 0 else 1
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS for λ = {lambda_sparsity}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Sparsity: {final_sparsity:.2f}%")
    print(f"Active Parameters: {active_params:,} / {total_params:,}")
    print(f"Compression Ratio: {compression:.2f}x")
    print(f"{'='*60}")
    
    return model, history, test_accuracy, final_sparsity


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_gate_distribution(model, lambda_val, save_path=None):
    """Plot histogram of gate values"""
    gates = model.get_all_gates().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(gates, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0.01, color='red', linestyle='--', linewidth=2, 
                    label='Pruning Threshold (0.01)')
    axes[0].set_xlabel('Gate Value', fontsize=12)
    axes[0].set_ylabel('Number of Weights', fontsize=12)
    axes[0].set_title(f'Gate Distribution (λ = {lambda_val})', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics
    pruned = (gates < 0.01).mean() * 100
    kept = (gates > 0.9).mean() * 100
    mid = ((gates >= 0.01) & (gates <= 0.9)).mean() * 100
    
    stats_text = f"Pruned (<0.01): {pruned:.1f}%\nKept (>0.9): {kept:.1f}%\nMid-range: {mid:.1f}%"
    axes[1].text(0.5, 0.5, stats_text, transform=axes[1].transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1].set_title('Sparsity Statistics', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return pruned, kept, mid


def plot_training_curves(results_dict, save_path=None):
    """Plot training curves comparing different lambda values"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (lambda_val, data) in enumerate(results_dict.items()):
        history = data['history']
        epochs = range(1, len(history['val_accuracy']) + 1)
        
        axes[0].plot(epochs, history['val_accuracy'], color=colors[i], 
                    label=f'λ={lambda_val}', linewidth=2, marker='o', markersize=3)
        axes[1].plot(epochs, history['sparsity'], color=colors[i], 
                    label=f'λ={lambda_val}', linewidth=2, marker='s', markersize=3)
        axes[2].plot(epochs, history['sparse_loss'], color=colors[i], 
                    label=f'λ={lambda_val}', linewidth=2, marker='^', markersize=3)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy over Time', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Sparsity (%)', fontsize=12)
    axes[1].set_title('Sparsity over Time', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Sparsity Loss', fontsize=12)
    axes[2].set_title('Sparsity Regularization', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    """Run complete experiment"""
    
    print("="*70)
    print("🧠 SELF-PRUNING NEURAL NETWORK - COMPLETE SOLUTION")
    print("="*70)
    print(f"Device: {device}")
    print("="*70)
    
    # Different lambda values to test
    lambda_values = [0.0001, 0.001, 0.005, 0.01]
    
    results = {}
    results_table = []
    
    for lambda_val in lambda_values:
        model, history, test_acc, sparsity = train_model(
            lambda_sparsity=lambda_val,
            num_epochs=15  # Use 15 epochs for faster demo, increase to 25 for better results
        )
        
        results[lambda_val] = {
            'model': model,
            'history': history,
            'test_accuracy': test_acc,
            'sparsity': sparsity
        }
        
        results_table.append({
            'Lambda': lambda_val,
            'Test Accuracy (%)': round(test_acc, 2),
            'Sparsity (%)': round(sparsity, 2)
        })
        
        # Plot gate distribution
        pruned, kept, mid = plot_gate_distribution(
            model, lambda_val, 
            save_path=f'gate_distribution_{lambda_val}.png'
        )
        
        print(f"\n📊 Gate Distribution for λ={lambda_val}:")
        print(f"   Pruned: {pruned:.1f}%, Kept: {kept:.1f}%, Mid: {mid:.1f}%")
    
    # Create results table
    df = pd.DataFrame(results_table)
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Save results
    df.to_csv('pruning_results.csv', index=False)
    print("\n✓ Results saved to 'pruning_results.csv'")
    
    # Plot comparison curves
    plot_training_curves(results, save_path='training_curves.png')
    
    # Best model analysis
    best_idx = df['Test Accuracy (%)'].idxmax()
    best_lambda = df.loc[best_idx, 'Lambda']
    best_accuracy = df.loc[best_idx, 'Test Accuracy (%)']
    best_sparsity = df.loc[best_idx, 'Sparsity (%)']
    
    print("\n" + "="*70)
    print("🏆 BEST MODEL ANALYSIS")
    print("="*70)
    print(f"Best λ: {best_lambda}")
    print(f"Test Accuracy: {best_accuracy}%")
    print(f"Sparsity: {best_sparsity}%")
    
    # Verify pruning success
    best_model = results[best_lambda]['model']
    gates = best_model.get_all_gates().numpy()
    spike_at_zero = (gates < 0.01).mean()
    spike_at_one = (gates > 0.9).mean()
    
    print("\n✅ PRUNING SUCCESS VERIFICATION:")
    print(f"   Bimodal distribution: {spike_at_zero*100:.1f}% at 0, {spike_at_one*100:.1f}% at 1")
    
    if spike_at_zero > 0.2 and spike_at_one > 0.1:
        print("   ✓ SUCCESS! Network shows clear pruning pattern")
    else:
        print("   ⚠ Consider training for more epochs for better separation")
    
    # Theoretical explanation
    print("\n" + "="*70)
    print("📖 WHY L1 PENALTY CREATES SPARSITY")
    print("="*70)
    print("""   
    Mathematical Explanation:
    
    Let g = σ(s) where σ is sigmoid, s is gate_score.
    L1 penalty = Σ|g| = Σg (since g > 0)
    
    Gradient: ∂L/∂s = g(1-g)
    
    Key insight: When g is small, gradient is small, but the 
    subgradient allows g to stay at 0 once reached.
    
    Comparison:
    - L1: Constant pressure towards 0 → EXACT ZEROS ✓
    - L2: Vanishing pressure near 0 → small values only ✗
    
    This is why L1 creates true sparsity!
    """)
    
    return results


if __name__ == "__main__":
    # Clear GPU cache if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run main experiment
    results = main()
    
    print("\n" + "="*70)
    print("🎉 EXPERIMENT COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("  - pruning_results.csv (results table)")
    print("  - training_curves.png (comparison plots)")
    print("  - gate_distribution_*.png (gate histograms)")
    print("  - best_model_lambda_*.pt (model checkpoints)")
    print("="*70)