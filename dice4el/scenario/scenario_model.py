import os
import time

import numpy as np

import torch
import torch.nn as nn

from typing import List, Tuple, Dict, Optional

from sklearn.metrics import f1_score, accuracy_score



class ScenarioLSTM(nn.Module):
    
    """
    Generalized Scenario LSTM (PyTorch version of LSTMScenarioCfModel).

    Supports multiple sequence channels (e.g., activity, resource, etc.)
    Each channel:
        - embedding
        - stacked LSTM encoder
    Then:
        - fusion of final hidden states
        - MLP prediction head
    """

    def __init__(
        self,
        categorical_info:   List[Tuple[int, int]],   # (num_categories, emb_dim)

        n_continuous:       int,
        n_classes:          int = 1,
        
        lstm_hidden_size:   int = 128,
        lstm_layers:        int = 1,
        dense_dim:          int = 64,
        dropout:            float = 0.1,
    ):
        
        super().__init__()

        self.categorical_info = categorical_info
        self.n_continuous = n_continuous
        self.n_classes = n_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        # --- Flags ---
        self.use_cat = len(categorical_info) > 0
        self.use_cont = n_continuous > 0

        # --- Embedding layers (categorical features) ---
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in categorical_info
        ]) if self.use_cat else None

        cat_dim = sum(emb_dim for _, emb_dim in categorical_info) if self.use_cat else 0

        # --- LSTM (sequence encoder) ---
        lstm_input_dim = cat_dim + (n_continuous if self.use_cont else 0)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

         # --- Final output layer (MLP) ---
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, dense_dim),
            nn.LayerNorm(dense_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, n_classes)
        )


    # --- Save ---
    def save(self, path: str = "./pretrained_models/") -> None:
  
        os.makedirs(path, exist_ok=True)
    
        config = {
            "categorical_info": self.categorical_info,
            "n_continuous": self.n_continuous,
            "n_classes": self.n_classes,
            "lstm_hidden_size": getattr(self, "lstm_hidden_size", 128),
            "lstm_layers": getattr(self, "lstm_layers", 1),
            "dense_dim": getattr(self, "dense_dim", 64),
            "dropout": getattr(self, "dropout", 0.2)
        }
    
        torch.save(config, f"{path}/scenario_model_config.pt")
        torch.save(self.state_dict(), f"{path}/scenario_model_weights.pt")

    
     # --- Load ---
    @classmethod
    def load(cls, path: str = "./pretrained_models/") -> "ScenarioLSTM":
      
        config = torch.load(f"{path}/scenario_model_config.pt", map_location="cpu")
    
        model = cls(**config)
    
        state_dict = torch.load(
            f"{path}/scenario_model_weights.pt",
            map_location="cpu"
        )
    
        model.load_state_dict(state_dict)
    
        return model

    
    # --- Forward pass ---
    def forward(
        self, 
        cat:     List[torch.Tensor],             # list of (batch, seq_len)
        cont:    torch.Tensor,                   # (batch, seq_len, n_dynamic_continuous)
        mask:    Optional[torch.Tensor] = None,  # (batch, seq_len) 1=valid, 0=padding
    ):
        
        device = next(self.parameters()).device
    
        # --- Categorical embeddings ---
        if self.use_cat:
            emb_list = [
                emb(x.to(device))
                for emb, x in zip(self.embeddings, cat)
            ]
            cat_emb = torch.cat(emb_list, dim=-1)
        else:
            cat_emb = None
    
        # --- Combine features ---
        if self.use_cont:
            cont = cont.to(device)
            x = cont if cat_emb is None else torch.cat([cat_emb, cont], dim=-1)
        else:
            x = cat_emb
    
        if x is None:
            raise ValueError("No input provided")
    
        # --- LSTM forward ---
        if mask is not None:
            lengths = mask.sum(dim=1).to(torch.long).cpu()
    
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.size(1)
            )
    
        else:
            lstm_out, _ = self.lstm(x)

        # --- Output ---
        logits = self.output_layer(lstm_out)

        return logits    # (batch, seq_len, 1)


    # --- Forward pass differentiable ---
    def forward_differentiable(
        self,
        cat:     List[torch.Tensor],                 # list of (B, T, K_i)
        cont:    torch.Tensor,                       # (B, T, n_continuous)
        mask:    Optional[torch.Tensor] = None,      # (B, T)
    ):
        
        """
        Differentiable forward pass for DiCE-style optimization.
    
        Normal forward:
            cat index -> nn.Embedding(index)
    
        This forward:
            soft one-hot / probability vector -> probs @ embedding.weight
        """
    
        device = next(self.parameters()).device
    
        # --- 1. Categorical feature processing ---
        if self.use_cat:
            emb_list = [
                torch.matmul(cat_prob.to(device).float(), emb.weight)
                for cat_prob, emb in zip(cat, self.embeddings)
            ]
    
            cat_emb = torch.cat(emb_list, dim=-1)
    
        else:
            cat_emb = None
    
        # --- 2. Combine features ---
        if self.use_cont:
            if cont is None:
                raise ValueError(
                    "cont must be provided when continuous features are used."
                )
    
            cont = cont.to(device).float()
            x = cont if cat_emb is None else torch.cat([cat_emb, cont], dim=-1)
    
        else:
            x = cat_emb
    
        if x is None:
            raise ValueError("No input provided to forward_differentiable().")
    
        # --- 3. LSTM forward ---
        if mask is not None:
            lengths = mask.to(device).bool().sum(dim=1).to(torch.long).cpu()
    
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
    
            packed_out, _ = self.lstm(packed)
    
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.size(1)
            )
    
        else:
            lstm_out, _ = self.lstm(x)
    
        # --- 4. Output prediction head ---
        logits = self.output_layer(lstm_out)
    
        return logits      # (B, T, n_classes)



def _train_one_epoch(
    model:           ScenarioLSTM,
    train_loader:    torch.utils.data.DataLoader,
    optimizer:       torch.optim.Optimizer,
    criterion:       nn.Module = nn.BCEWithLogitsLoss(),
) -> float:
    
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:

        cat = batch["cat"]
        cont = batch["cont"]
        mask = batch["mask"]       # (B, T)            
        labels = batch["labels"]   # (B, T)

        optimizer.zero_grad(set_to_none=True)

        # --- Forward pass ---
        logits = model(
            cat=cat,
            cont=cont,
            mask=mask
        )  # (B, T, 1)

        # --- Flatten ---
        logits_flat = logits.view(-1)                 
        labels_flat = labels.view(-1).to(logits_flat.device)               
        mask_flat = mask.view(-1).to(logits_flat.device).bool() 

        # --- Loss (only valid tokens) ---
        loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])

        # --- Backprop ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Metrics ---
        n_valid_tokens = mask_flat.sum().item()
        total_loss += loss.item() * n_valid_tokens
        total_tokens += n_valid_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0



def train_ScenarioLSTM(
    model:          ScenarioLSTM,
    train_loader:   torch.utils.data.DataLoader,
    learning_rate:  float = 1e-3,
    criterion:      nn.Module = nn.BCEWithLogitsLoss(),
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
        "train_loss": []
    }

    # --- Training loop ---
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

    print(f"Time taken for scenario model (training): {end - start:.6f} seconds")

    return history



@torch.no_grad()
def validate_ScenarioLSTM(
    model:        ScenarioLSTM,
    val_loader:   torch.utils.data.DataLoader,
    criterion:    nn.Module = nn.BCEWithLogitsLoss(),
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

        cat = batch["cat"]
        cont = batch["cont"]
        mask = batch["mask"]       # (B, T)            
        labels = batch["labels"]   # (B, T)

        # --- Forward pass ---
        logits = model(
            cat=cat,
            cont=cont,
            mask=mask
        )  # (B, T, 1)

        # --- Flatten ---
        logits_flat = logits.view(-1)                 
        labels_flat = labels.view(-1).to(logits_flat.device)               
        mask_flat = mask.view(-1).to(logits_flat.device).bool() 

        # --- Loss (only valid tokens) ---
        loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat])

        # --- Metrics ---
        n_valid_tokens = mask_flat.sum().item()
        total_loss += loss.item() * n_valid_tokens
        total_tokens += n_valid_tokens

        # --- Predictions ---
        valid_logits = logits_flat[mask_flat]
        probs = torch.sigmoid(valid_logits)
        preds = (probs > 0.5).long()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels_flat[mask_flat].cpu().numpy())

    if total_tokens == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}
        
    # --- Concatenate all batches ---
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    end = time.perf_counter()
    print(f"Time taken for scenario model (validation): {end - start:.6f} seconds")

    return {
        "loss": total_loss / total_tokens,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
    }
