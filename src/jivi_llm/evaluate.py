import torch
from jivi_llm.loss import calc_loss_batch


def calc_loss_loader(loader, model, device, num_batches=None):
    # average loss over num_batches from a dataloader
    # if num_batches is None, use the whole loader
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for inp, tgt in loader:
            if num_batches is not None and count >= num_batches:
                break

            loss = calc_loss_batch(inp, tgt, model, device)
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / count if count > 0 else float("nan")


def evaluate_model(model, train_loader, val_loader, device, eval_batches=5):

    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_batches)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_batches)
    return train_loss, val_loss