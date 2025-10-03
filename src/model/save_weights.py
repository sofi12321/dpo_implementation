import torch
import os

def load_best_model(model, device, checkpoint_dir = '/content/checkpoints/', model_name="best_model_small.pt"):
    best_checkpoint = os.path.join(checkpoint_dir, model_name)
    # last_checkpoint_path = os.path.join(checkpoint_dir, "last_model.pt")

    checkpoint = torch.load(best_checkpoint, map_location=device)

    model.load_state_dict(checkpoint)

    # print(f"Загружена лучшая модель, эпоха: {checkpoint['epoch']}, val loss: {checkpoint['val_loss']:.4f}")
    return model