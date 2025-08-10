class MultimodalNeurochemicalTrainer:
    """Enhanced trainer with multimodal and neurochemical objectives"""
    
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.training_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'executive_loss': [],
            'neurochemical_loss': [],
            'communication_loss': [],
            'multimodal_loss': [],
            'visual_loss': [],
            'audio_loss': []
        }
        
    def multimodal_loss(self, outputs, visual_targets=None, audio_targets=None):
        """Loss function for multimodal learning objectives"""
        total_loss = 0.0
        loss_count = 0
        
        # Visual processing loss
        if 'visual_processing' in outputs and visual_targets is not None:
            visual_output = outputs['visual_processing']
            
            # Visual attention consistency loss
            if 'attention_weights' in visual_output:
                visual_attention_loss = -torch.mean(
                    visual_output['attention_weights'] * 
                    torch.log(visual_output['attention_weights'] + 1e-8)
                )
                total_loss += visual_attention_loss
                loss_count += 1
            
            # Visual feature consistency
            ventral_dorsal_consistency = F.mse_loss(
                visual_output['ventral_features'].mean(dim=-1, keepdim=True).expand_as(visual_output['dorsal_features']),
                visual_output['dorsal_features']
            )
            total_loss += 0.1 * ventral_dorsal_consistency
            loss_count += 1
        
        # Audio processing loss
        if 'audio_processing' in outputs and audio_targets is not None:
            audio_output = outputs['audio_processing']
            
            # Audio attention consistency
            if 'attention_weights' in audio_output:
                audio_attention_loss = -torch.mean(
                    audio_output['attention_weights'] * 
                    torch.log(audio_output['attention_weights'] + 1e-8)
                )
                total_loss += audio_attention_loss
                loss_count += 1
            
            # Spectral-temporal consistency
            spectral_temporal_consistency = F.mse_loss(
                audio_output['spectral_features'].mean(dim=-1, keepdim=True).expand_as(audio_output['temporal_features']),
                audio_output['temporal_features']
            )
            total_loss += 0.1 * spectral_temporal_consistency
            loss_count += 1
            
            # Pitch and rhythm prediction loss (if targets available)
            if audio_targets is not None and 'pitch' in audio_targets:
                pitch_loss = F.cross_entropy(audio_output['pitch_prediction'], audio_targets['pitch'])
                total_loss += pitch_loss
                loss_count += 1
            
            if audio_targets is not None and 'rhythm' in audio_targets:
                rhythm_loss = F.cross_entropy(audio_output['rhythm_prediction'], audio_targets['rhythm'])
                total_loss += rhythm_loss  
                loss_count += 1
        
        # Multimodal integration loss
        if 'multimodal_integration' in outputs:
            multimodal_output = outputs['multimodal_integration']
            
            # Cross-modal attention consistency
            if 'attention_weights' in multimodal_output:
                multimodal_attention_loss = -torch.mean(
                    multimodal_output['attention_weights'] * 
                    torch.log(multimodal_output['attention_weights'] + 1e-8)
                )
                total_loss += multimodal_attention_loss
                loss_count += 1
        
        return total_loss / max(loss_count, 1)
    
    def train_step(self, cognitive_data, visual_data=None, audio_data=None, 
                   external_signals=None, target_neurotransmitters=None,
                   visual_targets=None, audio_targets=None):
        """Enhanced training step with multimodal inputs"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(cognitive_data, visual_data, audio_data, external_signals)
        
        # Standard losses
        reconstruction_target = torch.cat([cognitive_data, outputs['multimodal_integration']['integrated_features']], dim=-1)
        recon_loss = F.mse_loss(outputs['reconstruction'], reconstruction_target)
        
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - 
                                  outputs['logvar'].exp()) / cognitive_data.size(0)
        
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
        
        # Multimodal loss
        multimodal_loss_val = self.multimodal_loss(outputs, visual_targets, audio_targets)
        
        # Visual-specific loss
        visual_loss_val = 0.0
        if 'visual_processing' in outputs:
            # Visual attention control consistency
            visual_ctrl = outputs['visual_attention_control']
            visual_consistency = -torch.mean(visual_ctrl * torch.log(visual_ctrl + 1e-8))
            visual_loss_val = visual_consistency
        
        # Audio-specific loss  
        audio_loss_val = 0.0
        if 'audio_processing' in outputs:
            # Audio attention control consistency
            audio_ctrl = outputs['audio_attention_control']
            audio_consistency = -torch.mean(audio_ctrl * torch.log(audio_ctrl + 1e-8))
            audio_loss_val = audio_consistency
        
        # Total loss with weighted combination
        total_loss = (recon_loss + 0.1 * kl_loss + 0.5 * exec_loss + 
                     0.3 * neuro_loss + 0.2 * comm_loss + 0.4 * multimodal_loss_val +
                     0.2 * visual_loss_val + 0.2 * audio_loss_val)
        
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
        self.training_history['multimodal_loss'].append(multimodal_loss_val.item())
        self.training_history['visual_loss'].append(visual_loss_val if isinstance(visual_loss_val, float) else visual_loss_val.item())
        self.training_history['audio_loss'].append(audio_loss_val if isinstance(audio_loss_val, float) else audio_loss_val.item())
        
        return total_loss.item(), outputs
    
    def plot_multimodal_analysis(self, outputs):
        """Plot comprehensive multimodal analysis"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Neurochemical analysis (top row)
        if 'neurochemistry' in outputs:
            neurochemistry = outputs['neurochemistry']
            neurotransmitters = ['Dopamine', 'Norepinephrine', 'Serotonin', 'Acetylcholine']
            colors = ['red', 'blue', 'green', 'orange']
            
            # Layer 4 neurochemistry (most relevant for executive control)
            layer4_neuro = neurochemistry['layer4']
            ax = axes[0, 0]
            
            nt_values = []
            for nt_name in ['dopamine', 'norepinephrine', 'serotonin', 'acetylcholine']:
                values = layer4_neuro['neurotransmitters'][nt_name].detach().cpu().numpy()
                nt_values.append(np.mean(values))
            
            bars = ax.bar(neurotransmitters, nt_values, color=colors)
            ax.set_title('Executive Layer Neurotransmitters')
            ax.set_ylabel('Average Level')
            ax.set_ylim(0, 1)
            
            for bar, value in zip(bars, nt_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Executive functions analysis
        ax = axes[0, 1]
        exec_functions = ['Planning', 'Reasoning', 'Attention', 'Impulse Control']
        exec_values = [
            torch.mean(outputs['planning_actions']).item(),
            torch.mean(outputs['reasoning_conclusions']).item(), 
            torch.mean(outputs['attention_weights']).item(),
            torch.mean(outputs['impulse_inhibition']).item()
        ]
        
        ax.bar(exec_functions, exec_values, color=['purple', 'brown', 'pink', 'gray'])
        ax.set_title('Executive Function Levels')
        ax.set_ylabel('Average Activation')
        ax.set_ylim(0, 1)
        
        # Multimodal integration
        ax = axes[0, 2]
        if 'multimodal_integration' in outputs:
            multimodal_features = outputs['multimodal_integration']['integrated_features']
            feature_variance = torch.var(multimodal_features, dim=0).detach().cpu().numpy()
            ax.plot(feature_variance[:50])  # Plot first 50 features
            ax.set_title('Multimodal Feature Variance')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Variance')
        
        # Visual processing analysis (middle row)
        if 'visual_processing' in outputs:
            visual_output = outputs['visual_processing']
            
            # Ventral vs Dorsal stream comparison
            ax = axes[1, 0]
            ventral_mean = torch.mean(visual_output['ventral_features']).item()
            dorsal_mean = torch.mean(visual_output['dorsal_features']).item()
            
            ax.bar(['Ventral Stream\n(What)', 'Dorsal Stream\n(Where/How)'], 
                  [ventral_mean, dorsal_mean], color=['lightblue', 'lightgreen'])
            ax.set_title('Visual Stream Activation')
            ax.set_ylabel('Mean Activation')
            
            # Visual attention weights
            ax = axes[1, 1]
            if 'attention_weights' in visual_output:
                attention = visual_output['attention_weights'].detach().cpu().numpy()
                ax.imshow(attention[0], cmap='hot', interpolation='nearest')
                ax.set_title('Visual Attention Map')
                ax.set_xlabel('Attention Head')
                ax.set_ylabel('Spatial Region')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Visual Input', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'No Visual Input', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Visual attention control from executive system
        ax = axes[1, 2]
        visual_ctrl = outputs['visual_attention_control'].detach().cpu().numpy()
        ax.bar(range(len(visual_ctrl[0])), visual_ctrl[0], color='skyblue')
        ax.set_title('Executive Visual Attention Control')
        ax.set_xlabel('Visual Region')
        ax.set_ylabel('Control Signal')
        
        # Audio processing analysis (bottom row)
        if 'audio_processing' in outputs:
            audio_output = outputs['audio_processing']
            
            # Spectral vs Temporal processing
            ax = axes[2, 0]
            spectral_mean = torch.mean(audio_output['spectral_features']).item()
            temporal_mean = torch.mean(audio_output['temporal_features']).item()
            
            ax.bar(['Spectral Processing\n(Frequency)', 'Temporal Processing\n(Rhythm)'], 
                  [spectral_mean, temporal_mean], color=['lightcoral', 'lightyellow'])
            ax.set_title('Audio Stream Activation')
            ax.set_ylabel('Mean Activation')
            
            # Pitch and rhythm predictions
            ax = axes[2, 1]
            pitch_pred = audio_output['pitch_prediction'][0].detach().cpu().numpy()
            rhythm_pred = audio_output['rhythm_prediction'][0].detach().cpu().numpy()
            
            x = np.arange(len(pitch_pred))
            width = 0.35
            ax.bar(x - width/2, pitch_pred, width, label='Pitch', color='mediumseagreen')
            if len(rhythm_pred) <= len(pitch_pred):
                ax.bar(x[:len(rhythm_pred)] + width/2, rhythm_pred, width, label='Rhythm', color='orange')
            ax.set_title('Audio Feature Predictions')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Prediction Strength')
            ax.legend()
        else:
            axes[2, 0].text(0.5, 0.5, 'No Audio Input', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 1].text(0.5, 0.5, 'No Audio Input', ha='center', va='center', transform=axes[2, 1].transAxes)
        
        # Audio attention control from executive system
        ax = axes[2, 2]
        audio_ctrl = outputs['audio_attention_control'].detach().cpu().numpy()
        ax.bar(range(len(audio_ctrl[0])), audio_ctrl[0], color='lightpink')
        ax.set_title('Executive Audio Attention Control')
        ax.set_xlabel('Audio Frequency Band')
        ax.set_ylabel('Control Signal')
        
        plt.tight_layout()
        plt.show()

class MultimodalNetworkEcosystem:
    """Enhanced ecosystem for managing multimodal neural networks"""
    
    def __init__(self):
        self.networks = {}
        self.communication_history = []
        self.multimodal_history = []
        
    def add_network(self, name, network):
        """Add a network to the ecosystem"""
        self.networks[name] = network
        network.last_output = None
        
    def step(self, cognitive_inputs, visual_inputs=None, audio_inputs=None, 
             network_name='frontal_lobe'):
        """Execute one step with multimodal inter-network communication"""
        if network_name not in self.networks:
            raise ValueError(f"Network {network_name} not found")
        
        # Collect signals from other networks
        external_signals = []
        for other_name, other_network in self.networks.items():
            if other_name != network_name and hasattr(other_network, 'last_output'):
                if other_network.last_output is not None:
                    external_signals.append(other_network.last_output['outgoing_signal'])
        
        # Execute current network with multimodal inputs
        current_network = self.networks[network_name]
        outputs = current_network(cognitive_inputs, visual_inputs, audio_inputs, external_signals)
        current_network.last_output = outputs
        
        # Store communication and multimodal history
        self.communication_history.append({
            'network': network_name,
            'outgoing_signal': outputs['outgoing_signal'].detach().cpu().numpy(),
            'external_signals_received': len(external_signals)
        })
        
        self.multimodal_history.append({
            'network': network_name,
            'modalities': {
                'visual': visual_inputs is not None,
                'audio': audio_inputs is not None,
                'cognitive': cognitive_inputs is not None
            },
            'integration_strength': torch.mean(outputs['multimodal_integration']['integrated_features']).item()
        })
        
        return outputsimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class VisualProcessingModule(nn.Module):
    """
    Visual processing network inspired by ventral and dorsal visual streams
    Integrates with frontal lobe for visual-executive integration
    """
    
    def __init__(self, input_channels=3, output_dim=256):
        super(VisualProcessingModule, self).__init__()
        
        self.output_dim = output_dim
        
        # Ventral stream - "what" pathway (object recognition)
        self.ventral_stream = nn.Sequential(
            # Early visual processing
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Feature extraction layers
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Dorsal stream - "where/how" pathway (spatial processing, movement)
        self.dorsal_stream = nn.Sequential(
            # Motion and spatial processing
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Specialized for motion detection
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Stream integration
        self.ventral_fc = nn.Linear(512 * 4 * 4, output_dim // 2)
        self.dorsal_fc = nn.Linear(256 * 4 * 4, output_dim // 2)
        
        # Visual attention mechanism
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Visual working memory
        self.visual_memory_size = 5
        self.visual_memory = deque(maxlen=self.visual_memory_size)
        
    def forward(self, visual_input, store_memory=True):
        batch_size = visual_input.size(0)
        
        # Process through dual streams
        ventral_features = self.ventral_stream(visual_input)
        dorsal_features = self.dorsal_stream(visual_input)
        
        # Flatten and project
        ventral_flat = ventral_features.view(batch_size, -1)
        dorsal_flat = dorsal_features.view(batch_size, -1)
        
        ventral_projected = self.ventral_fc(ventral_flat)
        dorsal_projected = self.dorsal_fc(dorsal_flat)
        
        # Combine streams
        combined_visual = torch.cat([ventral_projected, dorsal_projected], dim=-1)
        
        # Apply visual attention
        visual_reshaped = combined_visual.unsqueeze(1)
        attended_visual, attention_weights = self.visual_attention(
            visual_reshaped, visual_reshaped, visual_reshaped
        )
        visual_output = attended_visual.squeeze(1)
        
        # Store in visual memory
        if store_memory:
            self.visual_memory.append(visual_output.detach().cpu().numpy())
        
        return {
            'visual_features': visual_output,
            'ventral_features': ventral_projected,
            'dorsal_features': dorsal_projected,
            'attention_weights': attention_weights,
            'raw_features': {'ventral': ventral_features, 'dorsal': dorsal_features}
        }

class AudioProcessingModule(nn.Module):
    """
    Audio processing network inspired by auditory cortex organization
    Processes spectral and temporal audio features
    """
    
    def __init__(self, input_dim=128, output_dim=256, sample_rate=16000):
        super(AudioProcessingModule, self).__init__()
        
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        
        # Spectral processing pathway (frequency analysis)
        self.spectral_processor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Temporal processing pathway (rhythm, timing)
        self.temporal_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=51, stride=4, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Audio feature integration
        self.spectral_fc = nn.Linear(256 * 32, output_dim // 2)
        self.temporal_fc = nn.Linear(128 * 32, output_dim // 2)
        
        # Audio attention for selective listening
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=6,
            batch_first=True
        )
        
        # Auditory working memory
        self.audio_memory_size = 5
        self.audio_memory = deque(maxlen=self.audio_memory_size)
        
        # Pitch and rhythm detection heads
        self.pitch_detector = nn.Linear(output_dim, 12)  # 12 semitones
        self.rhythm_detector = nn.Linear(output_dim, 8)  # 8 rhythmic patterns
        
    def forward(self, audio_input, store_memory=True):
        batch_size = audio_input.size(0)
        
        # Ensure proper input shape for 1D convolution
        if len(audio_input.shape) == 2:
            audio_input = audio_input.unsqueeze(1)  # Add channel dimension
        
        # Process through dual pathways
        spectral_features = self.spectral_processor(audio_input)
        temporal_features = self.temporal_processor(audio_input)
        
        # Flatten and project
        spectral_flat = spectral_features.view(batch_size, -1)
        temporal_flat = temporal_features.view(batch_size, -1)
        
        spectral_projected = self.spectral_fc(spectral_flat)
        temporal_projected = self.temporal_fc(temporal_flat)
        
        # Combine pathways
        combined_audio = torch.cat([spectral_projected, temporal_projected], dim=-1)
        
        # Apply audio attention
        audio_reshaped = combined_audio.unsqueeze(1)
        attended_audio, attention_weights = self.audio_attention(
            audio_reshaped, audio_reshaped, audio_reshaped
        )
        audio_output = attended_audio.squeeze(1)
        
        # Specialized audio analysis
        pitch_prediction = torch.softmax(self.pitch_detector(audio_output), dim=-1)
        rhythm_prediction = torch.softmax(self.rhythm_detector(audio_output), dim=-1)
        
        # Store in auditory memory
        if store_memory:
            self.audio_memory.append(audio_output.detach().cpu().numpy())
        
        return {
            'audio_features': audio_output,
            'spectral_features': spectral_projected,
            'temporal_features': temporal_projected,
            'attention_weights': attention_weights,
            'pitch_prediction': pitch_prediction,
            'rhythm_prediction': rhythm_prediction
        }

class MultimodalIntegrationModule(nn.Module):
    """
    Integration module for combining visual, audio, and cognitive features
    Implements cross-modal attention and binding
    """
    
    def __init__(self, visual_dim=256, audio_dim=256, cognitive_dim=128, 
                 integrated_dim=512):
        super(MultimodalIntegrationModule, self).__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.cognitive_dim = cognitive_dim
        self.integrated_dim = integrated_dim
        
        # Modal projection layers
        self.visual_projector = nn.Linear(visual_dim, integrated_dim // 3)
        self.audio_projector = nn.Linear(audio_dim, integrated_dim // 3)
        self.cognitive_projector = nn.Linear(cognitive_dim, integrated_dim // 3)
        
        # Cross-modal attention mechanisms
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=integrated_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Binding network for temporal coherence
        self.binding_network = nn.Sequential(
            nn.Linear(integrated_dim, integrated_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(integrated_dim * 2, integrated_dim),
            nn.LayerNorm(integrated_dim)
        )
        
        # Multimodal working memory
        self.multimodal_memory = deque(maxlen=8)
        
    def forward(self, visual_features=None, audio_features=None, 
                cognitive_features=None, store_memory=True):
        
        modal_features = []
        
        # Project available modalities
        if visual_features is not None:
            visual_proj = self.visual_projector(visual_features)
            modal_features.append(visual_proj)
        else:
            modal_features.append(torch.zeros(
                cognitive_features.size(0), self.integrated_dim // 3
            ).to(cognitive_features.device))
        
        if audio_features is not None:
            audio_proj = self.audio_projector(audio_features)
            modal_features.append(audio_proj)
        else:
            modal_features.append(torch.zeros(
                cognitive_features.size(0), self.integrated_dim // 3
            ).to(cognitive_features.device))
        
        if cognitive_features is not None:
            cognitive_proj = self.cognitive_projector(cognitive_features)
            modal_features.append(cognitive_proj)
        else:
            modal_features.append(torch.zeros(
                visual_features.size(0), self.integrated_dim // 3
            ).to(visual_features.device))
        
        # Concatenate modal features
        integrated_features = torch.cat(modal_features, dim=-1)
        
        # Apply cross-modal attention
        integrated_reshaped = integrated_features.unsqueeze(1)
        attended_features, attention_weights = self.cross_modal_attention(
            integrated_reshaped, integrated_reshaped, integrated_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Apply binding network
        bound_features = self.binding_network(attended_features)
        
        # Store in multimodal memory
        if store_memory:
            self.multimodal_memory.append({
                'features': bound_features.detach().cpu().numpy(),
                'modalities_present': {
                    'visual': visual_features is not None,
                    'audio': audio_features is not None,
                    'cognitive': cognitive_features is not None
                }
            })
        
        return {
            'integrated_features': bound_features,
            'modal_projections': modal_features,
            'attention_weights': attention_weights
        }
    """
    Neurochemical modulation system that simulates the effects of key neurotransmitters
    on frontal lobe executive functions
    """
    
    def __init__(self, hidden_dim, context_dim=64):
        super(NeurochemicalModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # Neurotransmitter synthesis networks - ensure consistent output shapes
        self.dopamine_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Always outputs (batch, 1)
            nn.Sigmoid()
        )
        
        self.norepinephrine_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Always outputs (batch, 1)
            nn.Sigmoid()
        )
        
        self.serotonin_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Always outputs (batch, 1)
            nn.Sigmoid()
        )
        
        self.acetylcholine_synthesizer = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Always outputs (batch, 1)
            nn.Sigmoid()
        )
        
        # Neurotransmitter modulation gates - expect (batch, 1) input
        self.dopamine_gate = nn.Linear(1, hidden_dim)
        self.norepinephrine_gate = nn.Linear(1, hidden_dim)
        self.serotonin_gate = nn.Linear(1, hidden_dim)
        self.acetylcholine_gate = nn.Linear(1, hidden_dim)
        
        # Interaction matrix for neurotransmitter cross-talk: (4, 4)
        self.interaction_matrix = nn.Parameter(torch.randn(4, 4) * 0.1)
        
    def forward(self, neural_state, context=None):
        batch_size = neural_state.size(0)
        
        # Create context if not provided
        if context is None:
            context = torch.zeros(batch_size, self.context_dim).to(neural_state.device)
        
        # Combine neural state with context
        combined_input = torch.cat([neural_state, context], dim=-1)
        
        # Synthesize neurotransmitters
        dopamine = self.dopamine_synthesizer(combined_input)  # (batch, 1)
        norepinephrine = self.norepinephrine_synthesizer(combined_input)  # (batch, 1)
        serotonin = self.serotonin_synthesizer(combined_input)  # (batch, 1)
        acetylcholine = self.acetylcholine_synthesizer(combined_input)  # (batch, 1)
        
        # Stack neurotransmitters safely - keeping (batch, 1) shape
        neurotransmitters = torch.cat([dopamine, norepinephrine, 
                                     serotonin, acetylcholine], dim=-1)  # (batch, 4)
        
        # Apply neurotransmitter interactions
        interacted_nt = torch.matmul(neurotransmitters, self.interaction_matrix)  # (batch, 4)
        dopamine_int, norepinephrine_int, serotonin_int, acetylcholine_int = \
            interacted_nt.split(1, dim=-1)  # Each: (batch, 1)
        
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
                'dopamine': dopamine.squeeze(-1),  # Safe squeeze only last dim: (batch, 1) -> (batch,)
                'norepinephrine': norepinephrine.squeeze(-1),
                'serotonin': serotonin.squeeze(-1),
                'acetylcholine': acetylcholine.squeeze(-1)
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

class MultimodalNeurochemicalFrontalLobeHRM(nn.Module):
    """
    Enhanced 4-Layer HRM with multimodal processing (vision + audio),
    neurochemical modulation, and network integration
    """
    
    def __init__(self, input_dim=128, hidden_dims=[256, 512, 256, 128], 
                 latent_dim=64, num_attention_heads=8, external_comm_dim=64,
                 visual_input_size=(3, 224, 224), audio_input_size=16000):
        super(MultimodalNeurochemicalFrontalLobeHRM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_heads = num_attention_heads
        
        # Multimodal processing modules
        self.visual_processor = VisualProcessingModule(
            input_channels=visual_input_size[0], output_dim=256
        )
        self.audio_processor = AudioProcessingModule(
            input_dim=audio_input_size, output_dim=256
        )
        
        # Multimodal integration
        self.multimodal_integrator = MultimodalIntegrationModule(
            visual_dim=256, audio_dim=256, cognitive_dim=hidden_dims[3],
            integrated_dim=512
        )
        
        # Enhanced input processing to handle multimodal input
        self.sensory_encoder = nn.Sequential(
            nn.Linear(input_dim + 512, hidden_dims[0]),  # +512 for multimodal features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2)
        )
        
        # Base neural architecture (same structure as before)
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
        
        # Enhanced working memory
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
            nn.Linear(hidden_dims[0], input_dim + 512),  # Include multimodal reconstruction
            nn.Sigmoid()
        )
        
        # Executive function outputs with multimodal influence
        self.planning_output = nn.Linear(hidden_dims[3], 32)
        self.reasoning_output = nn.Linear(hidden_dims[3], 16)
        self.attention_control = nn.Linear(hidden_dims[3], 8)
        self.impulse_control = nn.Linear(hidden_dims[3], 1)
        
        # Multimodal-specific outputs
        self.visual_attention_control = nn.Linear(hidden_dims[3], 4)  # Visual attention regions
        self.audio_attention_control = nn.Linear(hidden_dims[3], 4)   # Audio attention frequencies
        
        # Neurotransmitter prediction heads
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
    
    def forward(self, cognitive_input, visual_input=None, audio_input=None, 
                external_signals=None, store_in_memory=True):
        batch_size = cognitive_input.size(0)
        
        # Process multimodal inputs
        visual_features = None
        audio_features = None
        
        if visual_input is not None:
            visual_output = self.visual_processor(visual_input)
            visual_features = visual_output['visual_features']
        
        if audio_input is not None:
            audio_output = self.audio_processor(audio_input)
            audio_features = audio_output['audio_features']
        
        # Integrate multimodal information
        multimodal_output = self.multimodal_integrator(
            visual_features, audio_features, None  # Cognitive features added later
        )
        multimodal_features = multimodal_output['integrated_features']
        
        # Combine cognitive input with multimodal features
        combined_input = torch.cat([cognitive_input, multimodal_features], dim=-1)
        
        # Layer 1: Enhanced sensory processing
        h1 = self.sensory_encoder(combined_input)
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
        
        # Re-integrate with cognitive features for multimodal binding
        final_multimodal = self.multimodal_integrator(
            visual_features, audio_features, h4_modulated
        )
        
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
        
        # Multimodal attention controls
        visual_attention_ctrl = torch.softmax(self.visual_attention_control(h4_modulated), dim=-1)
        audio_attention_ctrl = torch.softmax(self.audio_attention_control(h4_modulated), dim=-1)
        
        # Neurotransmitter predictions
        nt_predictions = self.nt_predictor(h4_modulated)
        
        # Generate outgoing communication signal
        outgoing_signal = self.network_interface.send_signal(h4_modulated)
        
        result = {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'latent': z,
            'planning_actions': planning_actions,
            'reasoning_conclusions': reasoning_conclusions,
            'attention_weights': attention_weights,
            'impulse_inhibition': impulse_inhibition,
            'visual_attention_control': visual_attention_ctrl,
            'audio_attention_control': audio_attention_ctrl,
            'neurotransmitter_predictions': nt_predictions,
            'outgoing_signal': outgoing_signal,
            'neurochemistry': {
                'layer1': neuro1,
                'layer2': neuro2,
                'layer3': neuro3,
                'layer4': neuro4
            },
            'multimodal_integration': final_multimodal,
            'layer_outputs': [h1_modulated, h2_attended, h3_attended, h4_modulated],
            'attention_maps': [attention_weights_2, attention_weights_3],
            'communication_attention': comm_attention
        }
        
        # Add modality-specific outputs if available
        if visual_input is not None:
            result['visual_processing'] = visual_output
        if audio_input is not None:
            result['audio_processing'] = audio_output
            
        return result
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
    print("Multimodal Neurochemical Frontal Lobe Neural Network")
    print("=" * 80)
    
    # Initialize enhanced multimodal model
    model = MultimodalNeurochemicalFrontalLobeHRM(
        input_dim=128,
        hidden_dims=[256, 512, 256, 128],
        latent_dim=64,
        num_attention_heads=8,
        external_comm_dim=64,
        visual_input_size=(3, 224, 224),
        audio_input_size=16000
    )
    
    # Initialize multimodal trainer
    trainer = MultimodalNeurochemicalTrainer(model, learning_rate=1e-3)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Modalities: Vision + Audio + Cognitive")
    print(f"Neurochemical Systems: 4 layers × 4 neurotransmitters")
    print(f"Network Communication: Multimodal with attention")
    
    # Create enhanced multimodal ecosystem
    ecosystem = MultimodalNetworkEcosystem()
    ecosystem.add_network('frontal_lobe', model)
    
    # Create additional networks (simplified multimodal versions)
    visual_cortex = MultimodalNeurochemicalFrontalLobeHRM(
        input_dim=128, external_comm_dim=64, visual_input_size=(3, 224, 224)
    )
    auditory_cortex = MultimodalNeurochemicalFrontalLobeHRM(
        input_dim=128, external_comm_dim=64, audio_input_size=16000
    )
    
    ecosystem.add_network('visual_cortex', visual_cortex)
    ecosystem.add_network('auditory_cortex', auditory_cortex)
    
    print("\nTraining with Multimodal Inputs and Neurochemical Modulation:")
    print("-" * 70)
    
    # Training demonstration with multimodal data
    for epoch in range(12):
        # Generate multimodal training data
        cognitive_data = torch.randn(8, 128)
        
        # Generate visual data (simulate images)
        visual_data = None
        if epoch % 3 != 0:  # 2/3 of the time include visual data
            visual_data = torch.randn(8, 3, 224, 224)
        
        # Generate audio data (simulate audio waveforms)  
        audio_data = None
        if epoch % 4 != 0:  # 3/4 of the time include audio data
            audio_data = torch.randn(8, 16000)
        
        # Create scenario-specific neurotransmitter targets
        if epoch < 4:
            # High attention/focus scenario (e.g., learning)
            target_nt = torch.tensor([[0.6, 0.7, 0.5, 0.8]] * 8)  # Moderate DA, high NE, moderate 5HT, high ACh
            scenario = "Learning/Focus"
        elif epoch < 8:
            # Stress response scenario
            target_nt = torch.tensor([[0.8, 0.9, 0.3, 0.4]] * 8)  # High DA, very high NE, low 5HT, low ACh  
            scenario = "Stress Response"
        else:
            # Relaxed/creative scenario
            target_nt = torch.tensor([[0.4, 0.3, 0.8, 0.6]] * 8)  # Low DA, low NE, high 5HT, moderate ACh
            scenario = "Creative/Relaxed"
        
        # Create dummy audio targets for demonstration
        audio_targets = None
        if audio_data is not None:
            audio_targets = {
                'pitch': torch.randint(0, 12, (8,)),  # Random pitch classes
                'rhythm': torch.randint(0, 8, (8,))   # Random rhythm patterns
            }
        
        # Training step with multimodal ecosystem
        outputs = ecosystem.step(cognitive_data, visual_data, audio_data, 'frontal_lobe')
        loss, _ = trainer.train_step(
            cognitive_data, visual_data, audio_data,
            target_neurotransmitters=target_nt,
            audio_targets=audio_targets
        )
        
        if epoch % 2 == 0:
            modalities_active = []
            if visual_data is not None:
                modalities_active.append("Vision")
            if audio_data is not None:
                modalities_active.append("Audio")
            modalities_active.append("Cognitive")
            
            print(f"Epoch {epoch} ({scenario}): Loss = {loss:.4f}")
            print(f"  Active Modalities: {', '.join(modalities_active)}")
            
            # Show neurochemical state
            nt_pred = outputs['neurotransmitter_predictions'][0].detach()
            print(f"  NT Levels: DA={nt_pred[0]:.3f}, NE={nt_pred[1]:.3f}, "
                  f"5HT={nt_pred[2]:.3f}, ACh={nt_pred[3]:.3f}")
            
            # Show multimodal integration strength
            integration_strength = torch.mean(outputs['multimodal_integration']['integrated_features']).item()
            print(f"  Multimodal Integration: {integration_strength:.3f}")
            
            if visual_data is not None and 'visual_processing' in outputs:
                visual_attention = torch.mean(outputs['visual_attention_control']).item()
                print(f"  Visual Executive Control: {visual_attention:.3f}")
            
            if audio_data is not None and 'audio_processing' in outputs:
                audio_attention = torch.mean(outputs['audio_attention_control']).item()
                print(f"  Audio Executive Control: {audio_attention:.3f}")
            
            print()
    
    print("Multimodal Integration Complete!")
    print("✓ Vision processing integrated (ventral/dorsal streams)")
    print("✓ Audio processing integrated (spectral/temporal analysis)")  
    print("✓ Cross-modal attention and binding implemented")
    print("✓ Executive control of multimodal attention")
    print("✓ Neurochemical modulation across all modalities")
    print("✓ Multi-network communication with multimodal context")
    
    # Final comprehensive test
    print(f"\nFinal Multimodal Test:")
    print("-" * 40)
    
    with torch.no_grad():
        test_cognitive = torch.randn(1, 128)
        test_visual = torch.randn(1, 3, 224, 224)
        test_audio = torch.randn(1, 16000)
        
        final_output = ecosystem.step(test_cognitive, test_visual, test_audio, 'frontal_lobe')
        
        print(f"Modalities processed: Vision + Audio + Cognitive")
        print(f"Executive decisions made: {len(ecosystem.communication_history)}")
        print(f"Multimodal binding strength: {torch.mean(final_output['multimodal_integration']['integrated_features']).item():.3f}")
        print(f"Visual stream balance: Ventral={torch.mean(final_output['visual_processing']['ventral_features']).item():.3f}, "
              f"Dorsal={torch.mean(final_output['visual_processing']['dorsal_features']).item():.3f}")
        print(f"Audio stream balance: Spectral={torch.mean(final_output['audio_processing']['spectral_features']).item():.3f}, "
              f"Temporal={torch.mean(final_output['audio_processing']['temporal_features']).item():.3f}")
        print(f"Executive impulse control: {final_output['impulse_inhibition'].item():.3f}")
        
        # Show pitch and rhythm detection
        pitch_detection = final_output['audio_processing']['pitch_prediction'][0]
        dominant_pitch = torch.argmax(pitch_detection).item()
        print(f"Detected dominant pitch class: {dominant_pitch}/12 (confidence: {pitch_detection[dominant_pitch]:.3f})")
        
        rhythm_detection = final_output['audio_processing']['rhythm_prediction'][0]  
        dominant_rhythm = torch.argmax(rhythm_detection).item()
        print(f"Detected rhythm pattern: {dominant_rhythm}/8 (confidence: {rhythm_detection[dominant_rhythm]:.3f})")
        
    print(f"\n🧠 Complete Multimodal Frontal Lobe System Ready!")
    print(f"📊 Training history contains {len(trainer.training_history['total_loss'])} steps")
    print(f"🔗 Network ecosystem manages {len(ecosystem.networks)} interconnected networks")
    print(f"🎯 System supports vision, audio, cognitive processing with neurochemical modulation")
    
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
