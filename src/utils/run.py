import os
import torch

from src.utils.eval_iter import evaluate_one_epoch
from src.utils.train_iter import train_one_epoch


def run_training(model, ref_model, train_dataloader, test_dataloader, optimizer, scheduler, device, batch_size=5, checkpoint_dir = './checkpoints/', num_epochs=300, early_stop_patience=50, num_exp=0):
    train_losses = []
    val_losses = []
    best_epoch = 0
    epochs_no_improve = 0

    # Validation iteration
    val_l = evaluate_one_epoch(model, ref_model, test_dataloader, device)
    val_losses.append(val_l)
    print(f"Average val loss before training: {val_l:.4f}")
    print()

    for epoch in range(num_epochs):
        # Train iteration
        train_l = train_one_epoch(model, ref_model, train_dataloader, optimizer, scheduler, epoch, device)
        train_losses.append(train_l)
        print(f"Epoch {epoch+1} completed. Average train loss: {train_l:.4f}")

        # Validation iteration
        val_l = evaluate_one_epoch(model, ref_model, test_dataloader, device)
        val_losses.append(val_l)
        print(f"Average val loss: {val_l:.4f}")
        print()


        if val_l < val_losses[best_epoch + 1]:
            best_epoch = epoch
            epochs_no_improve = 0

            # Сохраняем лучшую модель
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{num_exp}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_l,
                'val_loss': val_l
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        # Сохраняем чекпоинт последней эпохи
        last_checkpoint_path = os.path.join(checkpoint_dir, f"last_model_{num_exp}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_l,
            'val_loss': val_l,
            'best_val_loss': val_losses[best_epoch + 1]
        }, last_checkpoint_path)

        # Проверяем early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"Валидационный лосс не уменьшался {early_stop_patience} эпох. Остановка обучения")
            return

