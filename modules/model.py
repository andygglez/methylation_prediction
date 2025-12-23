import torch
import torch.nn as nn
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, DNA_kernel_sizes, DNA_strides, DNA_conv_channels, dropout=0.3, loss_fn=nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam, learning_rate=1e-3):
        super().__init__()
        # Module parameters
        self.DNA_layer1_kernel_size, self.DNA_layer2_kernel_size, self.DNA_layer3_kernel_size = DNA_kernel_sizes
        self.DNA_conv_channels = DNA_conv_channels
        self.DNA_layer1_stride, self.DNA_layer2_stride, self.DNA_layer3_stride = DNA_strides

        self.loss_fn = loss_fn()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.first_epoch_loss = None
        self.first_test_loss = None

        
        ############## Modules and architecture
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

        self.H3K4me3_module = nn.Sequential(
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


        self.H3K36me2_module = nn.Sequential(
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


        self.H3K27me3_module = nn.Sequential(
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


        self.H3K9me3_module = nn.Sequential(
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


        #### Cross-Attention
        self.attn = nn.MultiheadAttention(embed_dim=25, num_heads=5, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(125, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3):
        sequence = sequence.to(torch.float32).permute(0, 2, 1) ### Changed to (B,C=4,L=500) to use Conv1D
        # histone_stack = torch.stack([H3K4me3, H3K36me2, H3K27me3, H3K9me3]).permute(1,0,2) ### (B, 4, 500)
        
        dna_module_output = self.dna_module(sequence)
        # histone_stack_output = self.histone_encoder(histone_stack)

        H3K4me3_features = self.H3K4me3_module(H3K4me3.unsqueeze(1))
        H3K36me2_features = self.H3K36me2_module(H3K36me2.unsqueeze(1))
        H3K27me3_features = self.H3K27me3_module(H3K27me3.unsqueeze(1))
        H3K9me3_features = self.H3K9me3_module(H3K9me3.unsqueeze(1))
        
        
        stack = torch.stack([H3K4me3_features, H3K36me2_features, 
                             H3K27me3_features, H3K9me3_features,
                             dna_module_output]).permute(1,0,2,3).squeeze(dim=2) # Not sure if this is ok

        ### Attention
        attention_output, attention_weights = self.attn(stack, stack, stack, need_weights=True)
        attention_flattened = attention_output.reshape(attention_output.size(0), -1)

        ### FC
        methylation_prediction = self.fc(attention_flattened)

        return methylation_prediction
    
    def training_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch
        methylation = torch.round(methylation)
        
        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        loss = self.loss_fn(prediction, methylation.unsqueeze(-1).float())
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics["train_loss"].item()

        if self.current_epoch == 0:
            self.first_epoch_loss = epoch_loss

        if self.first_epoch_loss is not None:
            relative = epoch_loss / self.first_epoch_loss * 100
            print(f"Epoch: {self.current_epoch}: train_loss_relative", relative)
            self.log("train_loss_relative", relative)
    

    ############################# NOT USING THIS ######################################
    def validation_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch
        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        loss = loss_fn(prediction, methylation.unsqueeze(-1).float())
        self.log('val_loss', loss, on_epoch=True)
        return loss
    ############################# NOT USING THIS ######################################
    
    def test_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch
        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        loss = self.loss_fn(prediction, methylation.unsqueeze(-1).float())
        self.log('test_loss', loss.detach() / self.first_epoch_loss * 100)
        return loss
    
    # def on_test_epoch_end(self):
    #     epoch_loss = self.trainer.callback_metrics["test_loss"].item()
    #     if not hasattr(self, "first_test_loss") or self.first_test_loss is None:
    #         self.first_test_loss = epoch_loss
    #     relative = epoch_loss / self.first_test_loss * 100
    #     print("test_loss_relative:", relative)
    #     self.log("test_loss_relative", relative)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)