import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta = 0.1):
    """Вычисление DPO loss"""

    # Вычисляем логарифмы отношений вероятностей
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # Разность между политикой и reference моделью
    logits = policy_logratios - ref_logratios

    # DPO loss
    losses = -F.logsigmoid(beta * logits)

    # Неявные rewards для мониторинга
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def get_batch_logps(logits, labels, attention_mask=None):
    """Вычисление логарифмических вероятностей для батча"""
    if attention_mask is None:
        attention_mask = torch.ones_like(labels)

    # Сдвигаем лейблы и логиты для вычисления loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()

    # Вычисляем per-token log probabilities
    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1),
        dim=2,
        index=shift_labels.unsqueeze(2)
    ).squeeze(2)

    # Усредняем по последовательности с учетом mask
    return (per_token_logps * shift_attention_mask).sum(-1) / shift_attention_mask.sum(-1)
