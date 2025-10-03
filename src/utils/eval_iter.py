import torch

from src.utils.loss import get_batch_logps, dpo_loss


def evaluate_one_epoch(model, ref_model, dataloader, device):
    """Основной цикл обучения"""

    model.eval()
    ref_model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
