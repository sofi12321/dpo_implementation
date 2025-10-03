import torch.nn as nn

class PEFTGPT2MLP(nn.Module):
    """
    Module from
    """
    def __init__(self, model, hidden_size):
        super().__init__()
        self.mlp_basic = model
        input_size = model.c_proj.nf

        self.add_fc = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.add_proj = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(model.dropout.p)

    def forward(self, hidden_states):
        hidden_states = self.mlp_basic(hidden_states)

        hidden_states = self.add_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.add_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

def make_adapter_model(model):
    # Add adapter layers
    for i in range(1, 11):
        model.transformer.h[i].mlp = PEFTGPT2MLP(model.transformer.h[i].mlp, 100)
        # Freeze all params in the layer
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False
        # Unfreeze adapter layers
        for part in [model.transformer.h[i].mlp.add_fc, model.transformer.h[i].mlp.act, model.transformer.h[i].mlp.add_proj, model.transformer.h[i].mlp.dropout]:
            for param in part.parameters():
                param.requires_grad = True
    return model
