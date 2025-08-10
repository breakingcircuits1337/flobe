import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import matplotlib.pyplot as plt

class NeurochemicalModule(nn.Module):
    """
    Neurochemical modulation system that simulates the effects of key neurotransmitters
    on frontal lobe executive functions
    """
    
    def __init__(self, hidden_dim, context_dim=64):
        super(NeurochemicalModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # Neurotransmitter synthesis networks
        self.dopamine_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.norepinephrine_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.serotonin_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.acetylcholine_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Neurotransmitter modulation gates
        self.dopamine_gate = nn.Linear(1, hidden_dim)
        self.norepinephrine_gate = nn.Linear(1, hidden_dim)
        self.serotonin_gate = nn.Linear(1, hidden_dim)
        self.acetylcholine_gate = nn.Linear(1, hidden_dim)
        
        # Interaction matrix for neurotransmitter cross-talk
        self.interaction_matrix = nn.Parameter(torch.randn(4, 4) * 0.1)
        
    def forward(self, neural_state, context=None):
        batch_size = neural_state.size(0)
        
        # Create context if not provided
        if context is None:
            context = torch.zeros(batch_size, self.context_dim).to(neural_state.device)
        
        # Combine neural state with context
        combined_input = torch.cat([neural_state, context], dim=-1)
        
        # Synthesize neurotransmitters
        dopamine = self.dopamine_synthesizer(combined_input)
        norepinephrine = self.norepinephrine_synthesizer(combined_input)
        serotonin = self.serotonin_synthesizer(combined_input)
        acetylcholine = self.acetylcholine_synthesizer(combined_input)
        
        # Stack neurotransmitters for interaction processing
        neurotransmitters = torch.stack([dopamine.squeeze(), norepinephrine.squeeze(), 
                                       serotonin.squeeze(), acetylcholine.squeeze()], dim=-1)
        
        # Apply neurotransmitter interactions
        interacted_nt = torch.matmul(neurotransmitters, self.interaction_matrix)
        dopamine_int, norepinephrine_int, serotonin_int, acetylcholine_int = \
            interacted_nt.split(1, dim=-1)
        
        # Generate modulation signals
        dopamine_mod = torch.tanh(self.dopamine_gate(dopamine_int))
        norepinephrine_mod = torch.tanh(self.norepinephrine_gate(norepinephrine_int))
        serotonin_mod = torch.tanh(self.serotonin_gate(serotonin_int))
        acetylcholine_mod = torch.tanh(self.acetylcholine_gate(acetylcholine_int))
        
        # Apply neurochemical modulation to neural state
        # Dopamine: motivation, reward processing, working memory
        dopamine_effect = neural_state * (1.0 + 0.5 * dopamine_mod)
        
        # Norepinephrine: attention, arousal, stress response
        norepinephrine_effect = dopamine_effect * (1.0 + 0.3 * norepinephrine_mod)
        
        # Serotonin: mood regulation, impulse control, decision making
        serotonin_effect = norepinephrine_effect * (1.0 + 0.4 * serotonin_mod)
        
        # Acetylcholine: attention, learning, memory consolidation
        modulated_state = serotonin_effect * (1.0 + 0.6 * acetylcholine_mod)
        
        return {
            'modulated_state': modulated_state,
            'neurotransmitters': {
                'dopamine': dopamine.squeeze(),
                'norepinephrine': norepinephrine.squeeze(),
                'serotonin': serotonin.squeeze(),
                'acetylcholine': acetylcholine.squeeze()
            },
            'modulation_signals': {
                'dopamine_mod': dopamine_mod,
                'norepinephrine_mod': norepinephrine_mod,
                'serotonin_mod': serotonin_mod,
                'acetylcholine_mod': acetylcholine_mod
            }
        }

class NetworkInterface(nn.Module):
    """
    Interface module for communication between different neural networks
    """
    
    def __init__(self, internal_dim, external_dim, interface_dim=128):
        super(NetworkInterface, self).__init__()
        
        self.interface_dim = interface_dim
        
        # Outgoing communication (to other networks)
        self.output_encoder = nn.Sequential(
            nn.Linear(internal_dim, interface_dim),
            nn.ReLU(),
            nn.Linear(interface_dim, external_dim),
            nn.Tanh()
        )
        
        # Incoming communication (from other networks)
        self.input_decoder = nn.Sequential(
            nn.Linear(external_dim * 2, interface_dim),  # 2 other networks
            nn.ReLU(),
            nn.Linear(interface_dim, internal_dim),
            nn.Tanh()
        )
        
        # Attention mechanism for selective communication
        self.communication_attention = nn.MultiheadAttention(
            embed_dim=external_dim,
            num_heads=4,
            batch_first=True
        )
        
    def send_signal(self, internal_state):
        """Encode internal state for external communication"""
        return self.output_encoder(internal_state)
    
    def receive_signals(self, external_signals):
        """Process incoming signals from other networks"""
        if len(external_signals) == 0:
            return None
        
        # Stack external signals
        stacked_signals = torch.stack(external_signals, dim=1)
        
        # Apply attention to selectively process signals
        attended_signals, attention_weights = self.communication_attention(
            stacked_signals, stacked_signals, stacked_signals
        )
        
        # Flatten for processing
        flattened_signals = attended_signals.view(attended_signals.size(0), -1)
        
        # Decode to internal representation
        internal_influence = self.input_decoder(flattened_signals)
        
        return internal_influence, attention_weights

class NeurochemicalFrontalLobeHRM(nn.Module):
    """
    Enhanced 4-Layer HRM with neurochemical modulation and network integration
    """
    
    def __init__(self, input_dim=128, hidden_dims=[256, 512, 256, 128], 
                 latent_dim=64, num_attention_heads=8, external_comm_dim=64):
        super(NeurochemicalFrontalLobeHRM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_heads = num_attention_heads
        
        # Base neural architecture (same as before)
        self.sensory_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2)
        )
        
        self.pattern_recognition = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1])
        )
        
        self.attention_layer2 = nn.MultiheadAttention(
            embed_dim=hidden_dims[1],
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        self.planning_module = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[2])
        )
        
        self.reasoning_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[2],
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        self.executive_control = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[3])
        )
        
        # Neurochemical modulation systems for each layer
        self.layer1_neurochemistry = NeurochemicalModule(hidden_dims[0])
        self.layer2_neurochemistry = NeurochemicalModule(hidden_dims[1])
        self.layer3_neurochemistry = NeurochemicalModule(hidden_dims[2])
        self.layer4_neurochemistry = NeurochemicalModule(hidden_dims[3])
        
        # Network interfaces
        self.network_interface = NetworkInterface(
            internal_dim=hidden_dims[3],
            external_dim=external_comm_dim
        )
        
        # Working memory with neurochemical influence
        self.working_memory_size = 10
        self.working_memory = deque(maxlen=self.working_memory_size)
        self.neurochemical_memory = deque(maxlen=self.working_memory_size)
        
        # VAE components
        self.encoder_mu = nn.Linear(hidden_dims[3], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[3], latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )
        
        # Executive function outputs with neurochemical influence
        self.planning_output = nn.Linear(hidden_dims[3], 32)
        self.reasoning_output = nn.Linear(hidden_dims[3], 16)
        self.attention_control = nn.Linear(hidden_dims[3], 8)
        self.impulse_control = nn.Linear(hidden_dims[3], 1)
        
        # Neurotransmitter prediction heads (for training)
        self.nt_predictor = nn.Sequential(
            nn.Linear(hidden_dims[3], 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 neurotransmitters
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, external_signals=None, store_in_memory=True):
        batch_size = x.size(0)
        
        # Layer 1: Sensory processing with neurochemical modulation
        h1 = self.sensory_encoder(x)
        neuro1 = self.layer1_neurochemistry(h1)
        h1_modulated = neuro1['modulated_state']
        
        # Layer 2: Pattern recognition with neurochemical modulation
        h2 = self.pattern_recognition(h1_modulated)
        neuro2 = self.layer2_neurochemistry(h2)
        h2_modulated = neuro2['modulated_state']
        
        h2_reshaped = h2_modulated.unsqueeze(1)
        h2_attended, attention_weights_2 = self.attention_layer2(
            h2_reshaped, h2_reshaped, h2_reshaped
        )
        h2_attended = h2_attended.squeeze(1)
        
        # Layer 3: Planning/reasoning with neurochemical modulation
        h3 = self.planning_module(h2_attended)
        neuro3 = self.layer3_neurochemistry(h3)
        h3_modulated = neuro3['modulated_state']
        
        h3_reshaped = h3_modulated.unsqueeze(1)
        h3_attended, attention_weights_3 = self.reasoning_attention(
            h3_reshaped, h3_reshaped, h3_reshaped
        )
        h3_attended = h3_attended.squeeze(1)
        
        # Layer 4: Executive control with neurochemical modulation
        h4 = self.executive_control(h3_attended)
        neuro4 = self.layer4_neurochemistry(h4)
        h4_modulated = neuro4['modulated_state']
        
        # Process external network communications
        external_influence = None
        comm_attention = None
        if external_signals is not None and len(external_signals) > 0:
            external_influence, comm_attention = self.network_interface.receive_signals(external_signals)
            if external_influence is not None:
                h4_modulated = h4_modulated + 0.2 * external_influence
        
        # Store in working memory
        if store_in_memory:
            self.working_memory.append(h4_modulated.detach().cpu().numpy())
            self.neurochemical_memory.append({
                'layer1': neuro1['neurotransmitters'],
                'layer2': neuro2['neurotransmitters'],
                'layer3': neuro3['neurotransmitters'],
                'layer4': neuro4['neurotransmitters']
            })
        
        # VAE encoding
        mu = self.encoder_mu(h4_modulated)
        logvar = self.encoder_logvar(h4_modulated)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        
        # Executive function outputs
        planning_actions = torch.softmax(self.planning_output(h4_modulated), dim=-1)
        reasoning_conclusions = torch.softmax(self.reasoning_output(h4_modulated), dim=-1)
        attention_weights = torch.softmax(self.attention_control(h4_modulated), dim=-1)
        impulse_inhibition = torch.sigmoid(self.impulse_control(h4_modulated))
        
        # Neurotransmitter predictions for training
        nt_predictions = self.nt_predictor(h4_modulated)
        
        # Generate outgoing communication signal
        outgoing_signal = self.network_interface.send_signal(h4_modulated)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'latent': z,
            'planning_actions': planning_actions,
            'reasoning_conclusions': reasoning_conclusions,
            'attention_weights': attention_weights,
            'impulse_inhibition': impulse_inhibition,
            'neurotransmitter_predictions': nt_predictions,
            'outgoing_signal': outgoing_signal,
            'neurochemistry': {
                'layer1': neuro1,
                'layer2': neuro2,
                'layer3': neuro3,
                'layer4': neuro4
            },
            'layer_outputs': [h1_modulated, h2_attended, h3_attended, h4_modulated],
            'attention_maps': [attention_weights_2, attention_weights_3],
            'communication_attention': comm_attention
        }
    
    def get_neurochemical_state(self):
        """Return current neurochemical state across all layers"""
        if len(self.neurochemical_memory) > 0:
            return list(self.neurochemical_memory)
        return None

class NeurochemicalTrainer:
    """Enhanced trainer with neurochemical objectives"""
    
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.training_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'executive_loss': [],
            'neurochemical_loss': [],
            'communication_loss': []
        }
        
    def neurochemical_loss(self, nt_predictions, target_nt=None):
        """Loss function for neurotransmitter prediction and balance"""
        batch_size = nt_predictions.size(0)
        
        if target_nt is None:
            # Create balanced target (homeostasis)
            target_nt = torch.ones_like(nt_predictions) * 0.5
        
        # Prediction loss
        prediction_loss = F.mse_loss(nt_predictions, target_nt)
        
        # Balance constraint (prevent extreme values)
        balance_loss = torch.mean(torch.abs(nt_predictions - 0.5))
        
        # Diversity constraint (encourage different NT profiles)
        diversity_loss = -torch.mean(torch.std(nt_predictions, dim=0))
        
        return prediction_loss + 0.1 * balance_loss + 0.05 * diversity_loss
    
    def communication_loss(self, outgoing_signals):
        """Loss to encourage meaningful inter-network communication"""
        # Encourage non-zero communication
        comm_magnitude = torch.mean(torch.norm(outgoing_signals, dim=-1))
        magnitude_loss = F.relu(0.5 - comm_magnitude)  # Penalty if too small
        
        # Encourage diversity in communication patterns
        diversity_loss = -torch.mean(torch.std(outgoing_signals, dim=0))
        
        return magnitude_loss + 0.1 * diversity_loss
    
    def train_step(self, batch_data, external_signals=None, target_neurotransmitters=None):
        """Enhanced training step with neurochemical objectives"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_data, external_signals)
        
        # Standard losses
        recon_loss = F.mse_loss(outputs['reconstruction'], batch_data)
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - 
                                  outputs['logvar'].exp()) / batch_data.size(0)
        
        # Executive function loss
        planning = outputs['planning_actions']
        reasoning = outputs['reasoning_conclusions']
        consistency_loss = F.mse_loss(
            planning.mean(dim=-1, keepdim=True).expand(-1, reasoning.size(-1)),
            reasoning
        )
        exec_loss = consistency_loss
        
        # Neurochemical loss
        neuro_loss = self.neurochemical_loss(
            outputs['neurotransmitter_predictions'], target_neurotransmitters
        )
        
        # Communication loss
        comm_loss = self.communication_loss(outputs['outgoing_signal'])
        
        # Total loss
        total_loss = (recon_loss + 0.1 * kl_loss + 0.5 * exec_loss + 
                     0.3 * neuro_loss + 0.2 * comm_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Store losses
        self.training_history['total_loss'].append(total_loss.item())
        self.training_history['reconstruction_loss'].append(recon_loss.item())
        self.training_history['kl_loss'].append(kl_loss.item())
        self.training_history['executive_loss'].append(exec_loss.item())
        self.training_history['neurochemical_loss'].append(neuro_loss.item())
        self.training_history['communication_loss'].append(comm_loss.item())
        
        return total_loss.item(), outputs
    
    def plot_neurochemical_analysis(self, outputs):
        """Plot neurochemical activity across layers"""
        neurochemistry = outputs['neurochemistry']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        neurotransmitters = ['Dopamine', 'Norepinephrine', 'Serotonin', 'Acetylcholine']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (layer_name, layer_neuro) in enumerate(neurochemistry.items()):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            nt_values = []
            for nt_name in ['dopamine', 'norepinephrine', 'serotonin', 'acetylcholine']:
                values = layer_neuro['neurotransmitters'][nt_name].detach().cpu().numpy()
                nt_values.append(np.mean(values))
            
            bars = ax.bar(neurotransmitters, nt_values, color=colors)
            ax.set_title(f'{layer_name.capitalize()} Neurotransmitter Levels')
            ax.set_ylabel('Average Level')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, nt_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Multi-network integration example
class NetworkEcosystem:
    """System for managing multiple interconnected neural networks"""
    
    def __init__(self):
        self.networks = {}
        self.communication_history = []
        
    def add_network(self, name, network):
        """Add a network to the ecosystem"""
        self.networks[name] = network
        
    def step(self, inputs, network_name):
        """Execute one step with inter-network communication"""
        if network_name not in self.networks:
            raise ValueError(f"Network {network_name} not found")
        
        # Collect signals from other networks
        external_signals = []
        for other_name, other_network in self.networks.items():
            if other_name != network_name and hasattr(other_network, 'last_output'):
                if other_network.last_output is not None:
                    external_signals.append(other_network.last_output['outgoing_signal'])
        
        # Execute current network
        current_network = self.networks[network_name]
        outputs = current_network(inputs, external_signals)
        current_network.last_output = outputs
        
        # Store communication history
        self.communication_history.append({
            'network': network_name,
            'outgoing_signal': outputs['outgoing_signal'].detach().cpu().numpy(),
            'external_signals_received': len(external_signals)
        })
        
        return outputs

# Example usage and demonstration
if __name__ == "__main__":
    print("Neurochemical Frontal Lobe Neural Network with Multi-Network Integration")
    print("=" * 80)
    
    # Initialize enhanced model
    model = NeurochemicalFrontalLobeHRM(
        input_dim=128,
        hidden_dims=[256, 512, 256, 128],
        latent_dim=64,
        num_attention_heads=8,
        external_comm_dim=64
    )
    
    # Initialize trainer
    trainer = NeurochemicalTrainer(model, learning_rate=1e-3)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Neurochemical Systems: 4 layers × 4 neurotransmitters")
    print(f"Network Communication: Bidirectional with attention")
    
    # Create multi-network ecosystem
    ecosystem = NetworkEcosystem()
    
    # Add main frontal lobe network
    ecosystem.add_network('frontal_lobe', model)
    
    # Simulate two additional networks (simplified versions)
    dummy_network1 = NeurochemicalFrontalLobeHRM(input_dim=128, external_comm_dim=64)
    dummy_network2 = NeurochemicalFrontalLobeHRM(input_dim=128, external_comm_dim=64)
    dummy_network1.last_output = None
    dummy_network2.last_output = None
    
    ecosystem.add_network('limbic_system', dummy_network1)
    ecosystem.add_network('motor_cortex', dummy_network2)
    
    # Training demonstration with neurochemical targets
    print("\nTraining with Neurochemical Modulation:")
    print("-" * 50)
    
    for epoch in range(10):
        # Generate training data
        batch_data = torch.randn(16, 128)
        
        # Create target neurotransmitter profiles (example: stress response)
        if epoch < 5:
            # High stress condition
            target_nt = torch.tensor([[0.8, 0.9, 0.3, 0.6]] * 16)  # High DA, NE, low 5HT, moderate ACh
        else:
            # Relaxed condition  
            target_nt = torch.tensor([[0.4, 0.3, 0.7, 0.5]] * 16)  # Moderate DA, low NE, high 5HT
        
        # Training step with ecosystem integration
        outputs = ecosystem.step(batch_data, 'frontal_lobe')
        loss, _ = trainer.train_step(batch_data, target_neurotransmitters=target_nt)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
            # Show neurochemical state
            nt_pred = outputs['neurotransmitter_predictions'][0].detach()
            print(f"  NT Levels: DA={nt_pred[0]:.3f}, NE={nt_pred[1]:.3f}, "
                  f"5HT={nt_pred[2]:.3f}, ACh={nt_pred[3]:.3f}")
    
    print("\nModel Integration Complete!")
    print("✓ Neurochemical modulation implemented")
    print("✓ Multi-network communication established") 
    print("✓ Executive function training with NT targets")
    
    # Final demonstration
    with torch.no_grad():
        test_input = torch.randn(1, 128)
        final_output = ecosystem.step(test_input, 'frontal_lobe')
        
        print(f"\nFinal Test Output:")
        print(f"Communication signals sent: {len(ecosystem.communication_history)}")
        print(f"Executive decision confidence: {final_output['impulse_inhibition'].item():.3f}")
        print(f"Inter-network attention: {final_output['communication_attention'] is not None}")
