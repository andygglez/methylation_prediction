
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim
import torch.nn.functional as F  # NUEVO: Para funciones adicionales

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor  # NUEVO: Callbacks útiles

import wandb
import os

# NUEVO: Importaciones para interpretabilidad
import matplotlib
matplotlib.use('Agg')  # NUEVO: Backend sin display para generar imágenes
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score  # NUEVO: Métricas adicionales

# CORREGIDO: Usar variable de entorno en lugar de hardcodear la API key
wandb_logger = WandbLogger(project="MethPrediction", log_model=True)  # MODIFICADO: log_model=True para guardar checkpoints

#### Dataset Class
class MethDataset(Dataset):
    def __init__(self, sequence, histone, methylation, coords, apply_log10=False):
        self.sequence = sequence
        self.histone = histone
        self.methylation = methylation
        self.transform = apply_log10
        self.coords = coords
        self.histone_names = ['H3K4me3', 'H3K36me2', 'H3K27me3', 'H3K9me3']

    def __len__(self):
        return self.methylation.shape[0]

    def __getitem__(self, idx):

        sequence = torch.from_numpy(self.sequence[idx])
        histone = self.histone.astype(np.float32)

        H3K4me3 = torch.from_numpy(histone[:, :, 0][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 0]+1e-4)[idx])
        H3K36me2 = torch.from_numpy(histone[:, :, 1][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 1]+1e-4)[idx])
        H3K27me3 = torch.from_numpy(histone[:, :, 2][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 2]+1e-4)[idx])
        H3K9me3 = torch.from_numpy(histone[:, :, 3][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 3]+1e-4)[idx])

        methylation = self.methylation[idx]
        coordinates = self.coords[idx]

        return sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates

class MethDataModule(pl.LightningDataModule):
    def __init__(self, npz_path, train_split=0.8, batch_size=32, apply_log10=True, num_workers=8):  # MODIFICADO: num_workers reducido por defecto
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = apply_log10
        self.num_workers = num_workers  # NUEVO: Parametrizado
        self.histone_names = ['H3K4me3', 'H3K36me2', 'H3K27me3', 'H3K9me3']

    def prepare_data(self):
        self.data = np.load(self.npz_path, allow_pickle=True)

    def setup(self, stage=None):
        split_index = int(self.train_split * self.data['dna'].shape[0])

        self.train_dataset = MethDataset(sequence = self.data['dna'][:split_index],
                                histone = self.data['histone'][:split_index],
                                methylation = self.data['methyl'][:split_index],
                                coords = self.data['coords'][:split_index],
                                apply_log10=self.transform)  # CORREGIDO: Usar self.transform

        self.test_dataset = MethDataset(sequence = self.data['dna'][split_index:],
                                histone = self.data['histone'][split_index:],
                                methylation = self.data['methyl'][split_index:],
                                coords = self.data['coords'][split_index:],
                                apply_log10=self.transform)  # CORREGIDO: Usar self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)  # MODIFICADO: shuffle=True y usar self.num_workers

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)  # MODIFICADO: Usar self.num_workers

# NUEVO: Clase para extraer y visualizar motivos de DNA
class DNAMotifVisualizer:
    """Visualiza filtros convolucionales como sequence logos"""

    @staticmethod
    def filter_to_pwm(filter_weights):
        """Convierte pesos de filtro a Position Weight Matrix"""
        # filter_weights: (out_channels, 4, kernel_size)
        pwm = F.softmax(filter_weights, dim=1)  # NUEVO: Normalizar sobre las 4 bases
        return pwm.detach().cpu().numpy()

    @staticmethod
    def plot_sequence_logo(pwm, title="DNA Motif", save_path=None):
        """Crea un logo de secuencia desde PWM"""
        bases = ['A', 'C', 'G', 'T']
        fig, ax = plt.subplots(figsize=(10, 3))  # NUEVO: Crear figura

        for pos in range(pwm.shape[1]):
            bottom = 0
            for base_idx, base in enumerate(bases):
                height = pwm[base_idx, pos]
                ax.bar(pos, height, bottom=bottom, label=base if pos == 0 else "",
                      color=['red', 'blue', 'orange', 'green'][base_idx], width=0.8)  # NUEVO: Colores estándar para bases
                ax.text(pos, bottom + height/2, base, ha='center', va='center', fontweight='bold')  # NUEVO: Letra de la base
                bottom += height

        ax.set_xlabel('Position')  # NUEVO: Etiqueta eje x
        ax.set_ylabel('Probability')  # NUEVO: Etiqueta eje y
        ax.set_title(title)  # NUEVO: Título
        ax.set_ylim(0, 1)  # NUEVO: Límite y

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # NUEVO: Guardar imagen
            plt.close()
        return fig

# NUEVO: Modelo mejorado con mejor arquitectura e interpretabilidad
class ImprovedModel(pl.LightningModule):
    def __init__(self,
                 DNA_kernel_sizes=(10, 10, 5),
                 DNA_strides=(2, 3, 3),
                 DNA_conv_channels=32,  # MODIFICADO: Más canales por defecto
                 embed_dim=128,  # NUEVO: Dimensión para attention
                 num_heads=8,  # NUEVO: Número de attention heads
                 dropout=0.3,  # NUEVO: Tasa de dropout
                 loss_fn=nn.MSELoss,
                 optimizer=torch.optim.Adam,
                 learning_rate=1e-3,
                 use_cross_attention=True):  # NUEVO: Opción para cross-attention
        super().__init__()

        self.save_hyperparameters()  # NUEVO: Guardar hiperparámetros para logging

        # Module parameters
        self.DNA_layer1_kernel_size, self.DNA_layer2_kernel_size, self.DNA_layer3_kernel_size = DNA_kernel_sizes
        self.DNA_conv_channels = DNA_conv_channels
        self.DNA_layer1_stride, self.DNA_layer2_stride, self.DNA_layer3_stride = DNA_strides
        self.embed_dim = embed_dim  # NUEVO
        self.num_heads = num_heads  # NUEVO
        self.dropout = dropout  # NUEVO
        self.use_cross_attention = use_cross_attention  # NUEVO

        self.loss_fn = loss_fn()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.first_epoch_loss = None
        self.first_test_loss = None

        # NUEVO: Métricas adicionales
        self.train_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()
        self.test_r2 = R2Score()
        self.train_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()

        # NUEVO: Para guardar outputs intermedios durante inferencia
        self.intermediates = {}
        self.save_intermediates = False

        ############## MEJORADO: Módulos con BatchNorm y Dropout
        self.dna_module = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=DNA_conv_channels,
                     kernel_size=self.DNA_layer1_kernel_size,
                     stride=self.DNA_layer1_stride, padding=0),
            nn.BatchNorm1d(DNA_conv_channels),  # NUEVO: Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),  # NUEVO: Dropout para regularización
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=DNA_conv_channels//2,
                     kernel_size=self.DNA_layer2_kernel_size,
                     stride=self.DNA_layer2_stride, padding=0),
            nn.BatchNorm1d(DNA_conv_channels//2),  # NUEVO
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.DNA_layer3_kernel_size,
                        stride=self.DNA_layer3_stride, padding=0)
        )

        # MEJORADO: Encoder compartido para histonas (más eficiente)
        self.histone_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=DNA_conv_channels,
                     kernel_size=self.DNA_layer1_kernel_size,
                     stride=self.DNA_layer1_stride, padding=0),
            nn.BatchNorm1d(DNA_conv_channels),  # NUEVO
            nn.ReLU(),
            nn.Dropout(dropout),  # NUEVO
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=DNA_conv_channels//2,
                     kernel_size=self.DNA_layer2_kernel_size,
                     stride=self.DNA_layer2_stride, padding=0),
            nn.BatchNorm1d(DNA_conv_channels//2),  # NUEVO
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.DNA_layer3_kernel_size,
                        stride=self.DNA_layer3_stride, padding=0)
        )

        # NUEVO: Proyecciones para ajustar a embed_dim
        self.dna_projection = nn.Linear(DNA_conv_channels//2, embed_dim)
        self.histone_projection = nn.Linear(DNA_conv_channels//2, embed_dim)

        # MEJORADO: Cross-Attention o Self-Attention configurable
        if use_cross_attention:
            self.attn_dna_to_histone = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                              batch_first=True, dropout=dropout)  # NUEVO: DNA query, histonas key/value
            self.attn_histone_to_dna = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                              batch_first=True, dropout=dropout)  # NUEVO: Histonas query, DNA key/value
        else:
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                             batch_first=True, dropout=dropout)  # MEJORADO: Parámetros configurables

        # NUEVO: Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # MEJORADO: FC layers con BatchNorm y Dropout
        if use_cross_attention:
            fc_input_dim = embed_dim * 2  # NUEVO: Concatenar outputs de ambas atenciones
        else:
            fc_input_dim = embed_dim * 5  # NUEVO: DNA + 4 histonas

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 256),  # MODIFICADO: Dimensiones ajustadas
            nn.BatchNorm1d(256),  # NUEVO
            nn.ReLU(),
            nn.Dropout(dropout),  # NUEVO
            nn.Linear(256, 128),  # NUEVO: Capa adicional
            nn.BatchNorm1d(128),  # NUEVO
            nn.ReLU(),
            nn.Dropout(dropout),  # NUEVO
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),  # NUEVO
            nn.Linear(64, 1),
            nn.Softplus()  # Para asegurar salida positiva
        )

    def forward(self, sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, return_attention=False):  # NUEVO: Parámetro return_attention
        # Procesar DNA
        sequence = sequence.to(torch.float32).permute(0, 2, 1)  # (B, C=4, L=500)
        dna_features = self.dna_module(sequence)  # (B, channels, L')

        if self.save_intermediates:
            self.intermediates['dna_features'] = dna_features.detach()  # NUEVO: Guardar features intermedias

        # Procesar histonas con encoder compartido (más eficiente)
        H3K4me3_features = self.histone_encoder(H3K4me3.unsqueeze(1))  # MODIFICADO: Usar encoder compartido
        H3K36me2_features = self.histone_encoder(H3K36me2.unsqueeze(1))
        H3K27me3_features = self.histone_encoder(H3K27me3.unsqueeze(1))
        H3K9me3_features = self.histone_encoder(H3K9me3.unsqueeze(1))

        if self.save_intermediates:
            self.intermediates['histone_features'] = {  # NUEVO: Guardar todas las features de histonas
                'H3K4me3': H3K4me3_features.detach(),
                'H3K36me2': H3K36me2_features.detach(),
                'H3K27me3': H3K27me3_features.detach(),
                'H3K9me3': H3K9me3_features.detach()
            }

        # NUEVO: Proyectar a embed_dim y preparar para attention
        dna_features = dna_features.permute(0, 2, 1)  # (B, L', channels)
        dna_embedded = self.dna_projection(dna_features)  # (B, L', embed_dim)

        # Concatenar todas las histonas
        histone_features = torch.stack([H3K4me3_features, H3K36me2_features,
                                       H3K27me3_features, H3K9me3_features], dim=1)  # (B, 4, channels, L')
        histone_features = histone_features.permute(0, 1, 3, 2).reshape(
            histone_features.size(0), -1, histone_features.size(2))  # (B, 4*L', channels)
        histone_embedded = self.histone_projection(histone_features)  # (B, 4*L', embed_dim)

        # MEJORADO: Cross-Attention bidireccional
        if self.use_cross_attention:
            # DNA como query, histonas como key/value
            dna_attended, attn_weights_1 = self.attn_dna_to_histone(
                dna_embedded, histone_embedded, histone_embedded)  # NUEVO

            # Histonas como query, DNA como key/value
            histone_attended, attn_weights_2 = self.attn_histone_to_dna(
                histone_embedded, dna_embedded, dna_embedded)  # NUEVO

            # Pooling sobre secuencia
            dna_pooled = dna_attended.mean(dim=1)  # (B, embed_dim)  # NUEVO: Global average pooling
            histone_pooled = histone_attended.mean(dim=1)  # (B, embed_dim)

            # Concatenar
            combined = torch.cat([dna_pooled, histone_pooled], dim=1)  # (B, embed_dim*2)  # NUEVO

            attention_weights = {'dna_to_histone': attn_weights_1,
                                'histone_to_dna': attn_weights_2}  # NUEVO: Guardar ambos
        else:
            # Self-attention sobre todas las features concatenadas
            all_features = torch.cat([dna_embedded, histone_embedded], dim=1)  # (B, L'+4*L', embed_dim)
            attended, attention_weights = self.attn(all_features, all_features, all_features)  # MODIFICADO
            attended = self.layer_norm(attended)  # NUEVO: Layer normalization
            combined = attended.mean(dim=1)  # (B, embed_dim)  # NUEVO: Pooling
            combined = combined.repeat(1, 5)  # (B, embed_dim*5) para mantener dimensión compatible

        if self.save_intermediates:
            self.intermediates['attention_weights'] = attention_weights  # NUEVO: Guardar attention weights

        # Predicción final
        methylation_prediction = self.fc(combined)

        if return_attention:
            return methylation_prediction, attention_weights  # NUEVO: Retornar attention si se solicita

        return methylation_prediction

    def training_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch
        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)

        loss = self.loss_fn(prediction, methylation.unsqueeze(-1).float())

        # NUEVO: Calcular métricas adicionales
        self.train_mae(prediction.squeeze(), methylation.float())
        self.train_r2(prediction.squeeze(), methylation.float())
        self.train_pearson(prediction.squeeze(), methylation.float())

        # Logging
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)  # MODIFICADO: prog_bar=True
        self.log('train_mae', self.train_mae, on_step=False, on_epoch=True)  # NUEVO
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True)  # NUEVO
        self.log('train_pearson', self.train_pearson, on_step=False, on_epoch=True)  # NUEVO

        return loss

    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics.get("train_loss")  # MODIFICADO: Usar .get() para evitar KeyError

        if epoch_loss is not None:
            epoch_loss = epoch_loss.item()

            if self.current_epoch == 0:
                self.first_epoch_loss = epoch_loss

            if self.first_epoch_loss is not None:
                relative = epoch_loss / self.first_epoch_loss * 100
                self.log("train_loss_relative", relative)

        # NUEVO: Visualizar DNA motifs cada N epochs
        if self.current_epoch % 5 == 0:  # Cada 5 epochs
            self.visualize_dna_filters()

    def test_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch

        # NUEVO: Activar guardado de intermediates en el primer batch de test
        if batch_idx == 0:
            self.save_intermediates = True

        prediction, attention_weights = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3,
                                                     return_attention=True)  # NUEVO: Obtener attention weights

        if batch_idx == 0:
            self.save_intermediates = False

        loss = self.loss_fn(prediction, methylation.unsqueeze(-1).float())  # CORREGIDO: self.loss_fn

        # NUEVO: Calcular métricas adicionales
        self.test_mae(prediction.squeeze(), methylation.float())
        self.test_r2(prediction.squeeze(), methylation.float())
        self.test_pearson(prediction.squeeze(), methylation.float())

        # Logging
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)  # MODIFICADO
        self.log('test_mae', self.test_mae, on_step=False, on_epoch=True)  # NUEVO
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)  # NUEVO
        self.log('test_pearson', self.test_pearson, on_step=False, on_epoch=True)  # NUEVO

        # NUEVO: Visualizar attention weights del primer batch
        if batch_idx == 0:
            self.visualize_attention_weights(attention_weights)

        return loss

    # NUEVO: Método para visualizar filtros de DNA
    def visualize_dna_filters(self):
        """Visualiza los filtros convolucionales de DNA como sequence logos"""
        try:
            # Obtener pesos del primer layer de DNA
            first_conv = self.dna_module[0]
            filters = first_conv.weight.data  # (out_channels, 4, kernel_size)

            # Crear directorio si no existe
            os.makedirs('dna_motifs', exist_ok=True)

            # Visualizar los primeros 4 filtros
            num_filters = min(4, filters.size(0))
            for i in range(num_filters):
                filter_weight = filters[i]  # (4, kernel_size)
                pwm = DNAMotifVisualizer.filter_to_pwm(filter_weight.unsqueeze(0))[0]  # (4, kernel_size)

                save_path = f'dna_motifs/filter_{i}_epoch_{self.current_epoch}.png'
                fig = DNAMotifVisualizer.plot_sequence_logo(pwm,
                                                            title=f"DNA Filter {i} - Epoch {self.current_epoch}",
                                                            save_path=save_path)

                # Enviar a WandB
                if self.logger:
                    self.logger.experiment.log({f"dna_motif/filter_{i}": wandb.Image(save_path)})

            print(f"DNA filters visualized and saved to dna_motifs/")
        except Exception as e:
            print(f"Error visualizing DNA filters: {e}")

    # NUEVO: Método para visualizar attention weights
    def visualize_attention_weights(self, attention_weights):
        """Visualiza los attention weights como heatmap"""
        try:
            os.makedirs('attention_maps', exist_ok=True)

            if self.use_cross_attention:
                # Visualizar ambas direcciones de cross-attention
                for attn_name, attn_tensor in attention_weights.items():
                    # attn_tensor: (B, num_heads, L_query, L_key)
                    # Promediar sobre batch y heads
                    attn_avg = attn_tensor[0].mean(dim=0).detach().cpu().numpy()  # (L_query, L_key)

                    plt.figure(figsize=(12, 8))
                    sns.heatmap(attn_avg, cmap='viridis', cbar=True)
                    plt.title(f'Attention Weights: {attn_name} - Epoch {self.current_epoch}')
                    plt.xlabel('Key positions')
                    plt.ylabel('Query positions')

                    save_path = f'attention_maps/{attn_name}_epoch_{self.current_epoch}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    # Enviar a WandB
                    if self.logger:
                        self.logger.experiment.log({f"attention/{attn_name}": wandb.Image(save_path)})
            else:
                # Self-attention
                attn_avg = attention_weights[0].mean(dim=0).detach().cpu().numpy()

                plt.figure(figsize=(12, 10))
                sns.heatmap(attn_avg, cmap='viridis', cbar=True)
                plt.title(f'Self-Attention Weights - Epoch {self.current_epoch}')
                plt.xlabel('Key positions')
                plt.ylabel('Query positions')

                save_path = f'attention_maps/self_attention_epoch_{self.current_epoch}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                if self.logger:
                    self.logger.experiment.log({"attention/self_attention": wandb.Image(save_path)})

            print(f"Attention weights visualized and saved to attention_maps/")
        except Exception as e:
            print(f"Error visualizing attention weights: {e}")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

        # NUEVO: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',  # Métrica a monitorear
                'interval': 'epoch',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    # NUEVO: Configuración de callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='methylation-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Inicializar data module
    data_module = MethDataModule(npz_path='chr19.npz', train_split=0.8, batch_size=32, num_workers=8)  # MODIFICADO: num_workers reducido

    # NUEVO: Inicializar modelo mejorado
    model = ImprovedModel(
        DNA_kernel_sizes=(10, 10, 5),
        DNA_strides=(2, 3, 3),
        DNA_conv_channels=32,
        embed_dim=128,
        num_heads=8,
        dropout=0.3,
        learning_rate=1e-3,
        use_cross_attention=True  # Usar cross-attention
    )

    # NUEVO: Trainer con callbacks
    trainer = pl.Trainer(
        max_epochs=300,  # MODIFICADO: Más epochs
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],  # NUEVO: Agregar callbacks
        gradient_clip_val=1.0,  # NUEVO: Gradient clipping para estabilidad
        log_every_n_steps=10,  # NUEVO: Log más frecuente
        enable_progress_bar=True  # NUEVO: Mostrar progress bar
    )

    # NUEVO: Compilar modelo (requiere PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Could not compile model: {e}")
        print("Continuing without compilation...")

    # Training loop
    trainer.fit(model=model, train_dataloaders=data_module)

    # Test loop
    trainer.test(model=model, dataloaders=data_module)

    # NUEVO: Cerrar wandb
    wandb.finish()

    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Checkpoints saved in: checkpoints/")
    print(f"DNA motifs saved in: dna_motifs/")
    print(f"Attention maps saved in: attention_maps/")
    print(f"WandB logs: {wandb_logger.experiment.url}")
