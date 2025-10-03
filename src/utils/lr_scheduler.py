from torch.optim.lr_scheduler import LambdaLR

def create_warmup_scheduler(optimizer, warmup_steps, target_lr):
    """
    Create a linear warmup scheduler using PyTorch's LambdaLR
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    # We need to scale the base learning rate to our target
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr

    return scheduler