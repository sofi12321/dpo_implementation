import torch
from tqdm import tqdm
from src.utils.loss import get_batch_logps, dpo_loss


def train_one_epoch(model, ref_model, dataloader, optimizer, scheduler, epoch, device):
    """Основной цикл обучения"""

    model.train()

    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Train epoch {epoch+1}")

    for batch_idx, batch in enumerate(progress_bar):
        # Перемещаем данные на устройство
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)

        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)


        # Forward pass для chosen примеров
        policy_chosen_outputs = model(
            chosen_input_ids,
            attention_mask=chosen_attention_mask
        )

        # Forward pass для rejected примеров
        policy_rejected_outputs = model(
            rejected_input_ids,
            attention_mask=rejected_attention_mask
        )

        # Reference model forward passes
        with torch.no_grad():
            ref_chosen_outputs = ref_model(
                chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            ref_rejected_outputs = ref_model(
                rejected_input_ids,
                attention_mask=rejected_attention_mask
            )

        # Вычисляем log probabilities
        policy_chosen_logps = get_batch_logps(
            policy_chosen_outputs.logits,
            chosen_labels,
            chosen_attention_mask
        )
        policy_rejected_logps = get_batch_logps(
            policy_rejected_outputs.logits,
            rejected_labels,
            rejected_attention_mask
        )
        ref_chosen_logps = get_batch_logps(
            ref_chosen_outputs.logits,
            chosen_labels,
            chosen_attention_mask
        )
        ref_rejected_logps = get_batch_logps(
            ref_rejected_outputs.logits,
            rejected_labels,
            rejected_attention_mask
        )

        # Вычисляем DPO loss
        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "chosen_reward": f"{chosen_rewards.item():.4f}",
            "rejected_reward": f"{rejected_rewards.item():.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    return avg_loss

