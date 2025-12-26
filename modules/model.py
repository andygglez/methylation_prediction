import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class Model(pl.LightningModule):
    def __init__(self, DNA_kernel_sizes, DNA_strides, DNA_conv_channels, linear_layers, 
                dropout=0.3, loss_fn=nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam, 
                learning_rate=1e-3):
        super().__init__()
        # Module parameters
        self.DNA_layer1_kernel_size, self.DNA_layer2_kernel_size, self.DNA_layer3_kernel_size = DNA_kernel_sizes
        self.DNA_conv_channels = DNA_conv_channels
        self.DNA_layer1_stride, self.DNA_layer2_stride, self.DNA_layer3_stride = DNA_strides
        self.dropout = dropout

        self.linear_layer2, self.linear_layer3, self.linear_layer4 = linear_layers

        # self.linear_layer2 = 250
        # self.linear_layer3 = 100
        # self.linear_layer4 = 10

        self.loss_fn = loss_fn()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.first_epoch_loss = None
        self.first_test_loss = None

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(task="binary"),
                "F1": torchmetrics.classification.F1Score(task="binary"),
                "AUROC": torchmetrics.classification.AUROC(task="binary"),
                "precision": torchmetrics.classification.Precision(task="binary"),
                "recall": torchmetrics.classification.Recall(task="binary")
            },
            prefix="train_",
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(task="binary"),
                "F1": torchmetrics.classification.F1Score(task="binary"),
                "AUROC": torchmetrics.classification.AUROC(task="binary"),
                "precision": torchmetrics.classification.Precision(task="binary"),
                "recall": torchmetrics.classification.Recall(task="binary")
            },
            prefix="test_",
        )

        
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

        #### Self-Attention
        def get_shape_Conv1D(length, kernel_size, padding, stride, dilation=1):
            return np.floor((length+2*padding-dilation*(kernel_size-1)-1)/stride+1).astype(int)
        def get_num_heads(embedding_dim):
            return max([d if embedding_dim % d == 0 else 1 for d in range(10, 1, -1)])
        
        out_dim1 = get_shape_Conv1D(500, self.DNA_layer1_kernel_size, 0, self.DNA_layer1_stride)
        out_dim2 = get_shape_Conv1D(out_dim1, self.DNA_layer2_kernel_size, 0, self.DNA_layer2_stride)
        embed_dimmension = get_shape_Conv1D(out_dim2, self.DNA_layer3_kernel_size, 0, self.DNA_layer3_stride)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dimmension, num_heads=5, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(embed_dimmension*5, self.linear_layer2),
            nn.ReLU(),
            nn.Linear(self.linear_layer2, self.linear_layer3),
            nn.ReLU(),
            nn.Linear(self.linear_layer3, self.linear_layer4),
            nn.ReLU(),
            nn.Linear(self.linear_layer4, 1)
        )

        # Original parameters
        # self.linear_layer2 = 250
        # self.linear_layer3 = 100
        # self.linear_layer4 = 10

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
        methylation = torch.round(methylation).int()
        
        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        loss = self.loss_fn(prediction, methylation.unsqueeze(-1).float())
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)

        batch_metrics = self.train_metrics(torch.sigmoid(prediction.squeeze()), methylation)
        self.log_dict(batch_metrics, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics["train_loss"].item()

        if self.current_epoch == 0:
            self.first_epoch_loss = epoch_loss

        if self.first_epoch_loss is not None:
            relative = epoch_loss / self.first_epoch_loss * 100
            print(f"Epoch: {self.current_epoch}: train_loss_relative", relative)
            self.log("train_loss_relative", relative, sync_dist=True)
        
        # self.train_metrics.reset()

    ############################# NOT USING THIS ######################################
    def validation_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch
        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        loss = loss_fn(prediction, methylation.unsqueeze(-1).float())
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss
    ############################# NOT USING THIS ######################################
    
    def test_step(self, batch, batch_idx):
        sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates = batch
        methylation = torch.round(methylation).int()

        prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        loss = self.loss_fn(prediction, methylation.unsqueeze(-1).float())
        self.log('test_loss', loss.detach() / self.first_epoch_loss * 100, on_step=True, on_epoch=True, sync_dist=True)

        test_metrics = self.test_metrics(torch.sigmoid(prediction.squeeze()), methylation)
        self.log_dict(test_metrics, on_step=True, on_epoch=True, sync_dist=True)

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