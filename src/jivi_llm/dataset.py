import torch 
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):

    # sliding window dataset for next word prediction 

    def __init__(self, token_ids, context_length, stride):
        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + context_length + 1]

            self.inputs.append(torch.tensor(input_chunk))
            self.targets.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]    


def create_dataloader(token_ids, context_length, stride, batch_size, shuffle=True):
    # wraping a token sequence in a dataloader ready for traning
    dataset = GPTDataset(token_ids, context_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 


            