import os
import time

import numpy as np

import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Optional

from sklearn.metrics import f1_score, accuracy_score



class ProcessLSTM(nn.Module):
    
    """
    Process-aware LSTM for next-event prediction.

    Architecture:
    - Dynamic categorical features  -> embeddings
    - Dynamic continuous features   -> concatenated to LSTM input
    - Static categorical features   -> embeddings -> MLP
    - Static continuous features    -> MLP
    - Fusion head                   -> final prediction
    """

    def __init__(
        self,
        
        dynamic_categorical_info: List[Tuple[int, int]],   # (num_categories, emb_dim)
        static_categorical_info:  List[Tuple[int, int]],   # (num_categories, emb_dim)
        
        n_dynamic_continuous:     int,
        n_static_continuous:      int,
        n_classes:                int,
        
        lstm_hidden_size:         int = 128,
        lstm_layers:              int = 1,
        static_hidden:            int = 64,
        fusion_hidden:            int = 128,
        dropout:                  float = 0.1,
    ):
        
        super().__init__()

        # --- Saved for reconstruct ---
        self.dynamic_categorical_info = dynamic_categorical_info
        self.static_categorical_info = static_categorical_info
        self.n_dynamic_continuous = n_dynamic_continuous
        self.n_static_continuous = n_static_continuous
        self.n_classes = n_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.static_hidden = static_hidden
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout

        # --- Flags ---
        self.use_dyn_cat = len(dynamic_categorical_info) > 0
        self.use_static_cat = len(static_categorical_info) > 0
        self.use_dyn_cont = n_dynamic_continuous > 0
        self.use_static_cont = n_static_continuous > 0

        # --- Embedding layers (dynamic categorical features) ---
        self.dynamic_embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in dynamic_categorical_info
        ]) if self.use_dyn_cat else None

        dyn_cat_dim = sum(emb_dim for _, emb_dim in dynamic_categorical_info) if self.use_dyn_cat else 0

        # --- LSTM (dynamic sequence encoder) ---
        lstm_input_dim = dyn_cat_dim + (n_dynamic_continuous if self.use_dyn_cont else 0)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # --- Embedding layers (static categorical features) ---
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in static_categorical_info
        ]) if self.use_static_cat else None

        static_cat_dim = sum(emb_dim for _, emb_dim in static_categorical_info) if self.use_static_cat else 0

        # --- MLP (static continuous) ---
        if self.use_static_cont:
            self.static_continuous = nn.Sequential(
                nn.Linear(n_static_continuous, static_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            static_cont_dim = static_hidden
        else:
            self.static_continuous = None
            static_cont_dim = 0

        static_total_dim = static_cat_dim + static_cont_dim

        # --- MLP (static projection) ---
        if static_total_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_total_dim, static_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fusion_static_dim = static_hidden
        else:
            self.static_proj = None
            fusion_static_dim = 0

        # --- Fusion head ---
        fusion_dim = lstm_hidden_size + fusion_static_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Final output layer ---
        self.output_layer = nn.Sequential(
            nn.LayerNorm(fusion_hidden), 
            nn.Linear(fusion_hidden, fusion_hidden), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(fusion_hidden, n_classes)
        )


    # --- Save ---
    def save(self, path:  str = "./pretrained_models/") -> None:
            
        """Save model configuration and weights to disk."""
            
        os.makedirs(path, exist_ok=True)
            
        config = {
            "dynamic_categorical_info": getattr(self, "dynamic_categorical_info", []),
            "static_categorical_info": getattr(self, "static_categorical_info", []),
            "n_dynamic_continuous": getattr(self, "n_dynamic_continuous", 0),
            "n_static_continuous": getattr(self, "n_static_continuous", 0),
            "n_classes": getattr(self, "n_classes", 0),
            "lstm_hidden_size": getattr(self, "lstm_hidden_size", 128),
            "lstm_layers": getattr(self, "lstm_layers", 1),
            "static_hidden": getattr(self, "static_hidden", 32),
            "fusion_hidden": getattr(self, "fusion_hidden", 128),
            "dropout": getattr(self, "dropout", 0.2)
        }
            
        torch.save(config, f"{path}/next_event_model_config.pt")
        torch.save(self.state_dict(), f"{path}/next_event_model_weights.pt")


    # --- Load ---
    @classmethod
    def load(cls, path:  str = "./pretrained_models/") -> "ProcessLSTM":
            
        """Load model configuration and weights from disk."""
            
        config = torch.load(f"{path}/next_event_model_config.pt")
        model = cls(**config)
            
        state_dict = torch.load(f"{path}/next_event_model_weights.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model

        
    # --- Forward pass ---
    def forward(
        self,
        dynamic_cat:  List[torch.Tensor],             # list of (batch, seq_len)
        dynamic_cont: torch.Tensor,                   # (batch, seq_len, n_dynamic_continuous)
        static_cat:   List[torch.Tensor],             # list of (batch,)
        static_cont:  torch.Tensor,                   # (batch, n_static_continuous)
        mask:         Optional[torch.Tensor] = None,  # (batch, seq_len) 1=valid, 0=padding
    ):
        
        # --- Current hardware device where the model lives ---
        device = next(self.parameters()).device
        
        # --- 1. Dynamic Features Processing ---
        dynamic_components = []

        if self.use_dyn_cat and dynamic_cat is not None:
            dyn_emb = [
                emb(x.to(device)) for emb, x in zip(self.dynamic_embeddings, dynamic_cat)
            ]
            x_dyn_cat = torch.cat(dyn_emb, dim=-1)
            dynamic_components.append(x_dyn_cat)

        if self.use_dyn_cont and dynamic_cont is not None:
            dynamic_components.append(dynamic_cont.to(device))

        if not dynamic_components:
            raise ValueError("Sequence model received zero active dynamic features. Check your inputs.")
            
        x_dyn = torch.cat(dynamic_components, dim=-1)

        # --- 2. LSTM Forward Pass (with optional packing) ---
        if mask is not None:
            lengths = mask.bool().sum(dim=1).to(torch.long).cpu()
            
            packed = nn.utils.rnn.pack_padded_sequence(
                x_dyn,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x_dyn.size(1)
            )
        else:
            lstm_out, _ = self.lstm(x_dyn)

        # --- 3. Static Features Processing ---
        static_components = []

        if self.use_static_cat and static_cat is not None:
            stat_emb = [
                emb(x.to(device)) for emb, x in zip(self.static_embeddings, static_cat)
            ]
            x_stat_cat = torch.cat(stat_emb, dim=-1)
            static_components.append(x_stat_cat)

        if self.use_static_cont and static_cont is not None:
            x_stat_cont = self.static_continuous(static_cont.to(device))
            static_components.append(x_stat_cont)

        # --- 4. Static Fusion and Broadcasting ---
        if static_components:
            x_stat = torch.cat(static_components, dim=-1)
            static_repr = self.static_proj(x_stat)
            static_repr = static_repr.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
            
            combined = torch.cat([lstm_out, static_repr], dim=-1)
        else:
            combined = lstm_out
    
        # --- 5. Output Prediction Head ---
        fused = self.fusion_head(combined)
        logits = self.output_layer(fused)
        
        return logits   # (batch, seq_len, n_classes)


    # --- Forward pass differentiable ---
    def forward_differentiable(
        self,
        dynamic_cat:     List[torch.Tensor],                # list of (B, T, K_i)
        dynamic_cont:    torch.Tensor,                      # (B, T, n_dynamic_continuous)
        static_cat:      List[torch.Tensor],                # list of (B, K_i)
        static_cont:     torch.Tensor,                      # (B, n_static_continuous)
        mask:            Optional[torch.Tensor] = None,     # (B, T)
    ):
        
        """
        Differentiable forward pass for DiCE-style counterfactual optimization.
    
        Categorical inputs are expected as soft one-hot / probability tensors,
        not integer label IDs.
    
        This replaces:
            embedding(cat_id)
    
        with:
            cat @ embedding.weight
        """
    
        device = next(self.parameters()).device
    
        # ---1.  Dynamic Features Processing ---
        dynamic_components = []
    
        if self.use_dyn_cat:
            dyn_emb = [
                torch.matmul(cat_prob.to(device).float(), emb.weight)
                for cat_prob, emb in zip(dynamic_cat, self.dynamic_embeddings)
            ]
    
            x_dyn_cat = torch.cat(dyn_emb, dim=-1)
            dynamic_components.append(x_dyn_cat)
    
        if self.use_dyn_cont:
            dynamic_components.append(dynamic_cont.to(device).float())
    
        if not dynamic_components:
            raise ValueError("forward_differentiable received zero dynamic features.")
    
        x_dyn = torch.cat(dynamic_components, dim=-1)

        # --- 2. LSTM Forward Pass ---
        if mask is not None:
            lengths = mask.bool().sum(dim=1).to(torch.long).cpu()
    
            packed = nn.utils.rnn.pack_padded_sequence(
                x_dyn,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
    
            packed_out, _ = self.lstm(packed)
    
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x_dyn.size(1)
            )
        else:
            lstm_out, _ = self.lstm(x_dyn)
    
        # --- 3. Static Features Processing ---
        static_components = []
    
        if self.use_static_cat:
            stat_emb = [
                torch.matmul(cat_prob.to(device).float(), emb.weight)
                for cat_prob, emb in zip(static_cat, self.static_embeddings)
            ]
    
            x_stat_cat = torch.cat(stat_emb, dim=-1)
            static_components.append(x_stat_cat)
    
        if self.use_static_cont:
            x_stat_cont = self.static_continuous(
                static_cont.to(device).float()
            )
            static_components.append(x_stat_cont)
    
        # --- 4. Static Fusion + Broadcasting ---
        if static_components:
            x_stat = torch.cat(static_components, dim=-1)
            static_repr = self.static_proj(x_stat)
    
            static_repr = static_repr.unsqueeze(1).expand(
                -1, lstm_out.size(1), -1
            )
    
            combined = torch.cat([lstm_out, static_repr], dim=-1)
        else:
            combined = lstm_out
    
        # --- 5. Output Prediction Head ---
        fused = self.fusion_head(combined)
        logits = self.output_layer(fused)
    
        return logits



def _train_one_epoch(
    model:           ProcessLSTM,
    train_loader:    torch.utils.data.DataLoader,
    optimizer:       torch.optim.Optimizer,
    criterion:       nn.Module = nn.CrossEntropyLoss(),
) -> float:
    
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:

        dynamic_cat = batch["dynamic_cat"]
        dynamic_cont = batch["dynamic_cont"]
        static_cat = batch["static_cat"]
        static_cont = batch["static_cont"]
        mask = batch["mask"]       # (B, T)            
        labels = batch["labels"]   # (B, T)

        optimizer.zero_grad(set_to_none=True)

        # --- Forward pass ---
        logits = model(
            dynamic_cat=dynamic_cat,
            dynamic_cont=dynamic_cont,
            static_cat=static_cat,
            static_cont=static_cont,
            mask=mask
        )  # (B, T, C)

        # --- Flatten ---
        B, T, C = logits.shape

        logits_flat = logits.reshape(B * T, C)
        labels_flat = labels.reshape(B * T)
        mask_flat = mask.reshape(B * T)

        valid_idx = mask_flat.bool()

        # --- Loss (only valid tokens) ---
        loss = criterion(
            logits_flat[valid_idx],
            labels_flat[valid_idx].to(logits_flat.device)
        ) # mean loss over valid tokens in the batch

        # --- Backprop ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Metrics ---
        total_loss += loss.item() * valid_idx.sum().item()
        total_tokens += valid_idx.sum().item()
        
        # preds = torch.argmax(logits_flat, dim=1)
        # total_correct += (preds[valid_idx] == labels_flat[valid_idx]).sum().item()

    return total_loss / total_tokens  # train_loader level mean loss per token
    # return {
    #     "loss": total_loss / total_tokens
    #     "accuracy": total_correct / total_tokens
    # }



def train_ProcessLSTM(
    model:          ProcessLSTM,
    train_loader:   torch.utils.data.DataLoader,
    learning_rate:  float = 1e-3,
    criterion:      nn.Module = nn.CrossEntropyLoss(),
    num_epochs:     int = 100,
    device:         str = "cpu",
    verbose_freq:   int = 20,
) -> Dict[str, List[float]]:

    start = time.perf_counter()
    
    model.to(device)
    criterion.to(device)
    
    # --- Parameter Grouping for AdamW ---
    # We isolate biases and LayerNorm parameters to exempt them from weight decay.
    # Decaying norms or biases can destabilise the final GELU activation head.
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,  # Standard, safe weight decay for AdamW
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,   # No weight decay for biases and norm scales
        },
    ]
    
    # --- Optimizer (AdamW) ---
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=learning_rate,
        eps=1e-8
    )
    
    # --- Learning Rate Scheduler ---
    # Pairs with AdamW to smoothly decay learning rates across epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=1e-6
    )

    history = {
        "train_loss": [],
    }

    # --- Training ---
    for epoch in range(num_epochs):

        train_loss = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion
        )
        
        scheduler.step()
        history["train_loss"].append(train_loss)

        # --- Logging ---
        if (epoch + 1) % verbose_freq == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1:03d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )

    end = time.perf_counter()
    print(f"Time taken for next event model (training): {end - start:.6f} seconds")

    return history



@torch.no_grad()
def validate_ProcessLSTM(
    model:        ProcessLSTM,
    val_loader:   torch.utils.data.DataLoader,
    criterion:    nn.Module = nn.CrossEntropyLoss(),
    device:       str = "cpu"
) -> Dict[str, float]:

    start = time.perf_counter()

    model.to(device)
    criterion.to(device)
    
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:

        dynamic_cat = batch["dynamic_cat"]
        dynamic_cont = batch["dynamic_cont"]
        static_cat = batch["static_cat"]
        static_cont = batch["static_cont"]
        mask = batch["mask"]
        labels = batch["labels"]

        # --- Forward pass ---
        logits = model(
            dynamic_cat=dynamic_cat,
            dynamic_cont=dynamic_cont,
            static_cat=static_cat,
            static_cont=static_cont,
            mask=mask
        )  # (B, T, C)

        # --- Flatten ---
        B, T, C = logits.shape

        logits_flat = logits.reshape(B * T, C)
        labels_flat = labels.reshape(B * T)
        mask_flat = mask.reshape(B * T)

        valid_idx = mask_flat.bool()

        # --- Loss (only valid tokens) ---
        loss = criterion(
            logits_flat[valid_idx],
            labels_flat[valid_idx].to(logits_flat.device)
        )

        # --- Metrics ---
        total_loss += loss.item() * valid_idx.sum().item()
        total_tokens += valid_idx.sum().item()

        # --- Predictions ---
        preds = torch.argmax(logits_flat, dim=1)

        all_preds.append(preds[valid_idx].cpu().numpy())
        all_labels.append(labels_flat[valid_idx].cpu().numpy())

    # --- Concatenate all batches ---
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    end = time.perf_counter()
    print(f"Time taken for next event model (validation): {end - start:.6f} seconds")

    return {
        "loss": total_loss / total_tokens,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
    }
