import torch 
import torch.nn.functional as F 

def calc_loss_batch(input_batch, target_batch, model, device):

    # moving data to correct device 
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # forword pass: get lotits (B, T, vocab_size)
    logits = model(input_batch)

    # reshape for cross-entropy: (B, T, V) => (B * T, V) and (B, T) -> (B * T)
    # cross_entropy expects : (number_of_predictions, number_of_classes)
    # In our case, number_of_predictions is (B * T) and number_of_classes is (vocab_size)
    # So we reshape the logits to (B * T, V)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), # It simply flattens the batch and sequence dimensions into one dimension.
        target_batch.view(-1) # and we do the same for the target batch 
    )

    return loss 
