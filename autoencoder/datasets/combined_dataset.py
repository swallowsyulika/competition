from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, dirty, clean) -> None:
        super().__init__()
        self.dirty_datasets = dirty
        self.clean_dataset = clean

        for dirty_dataset in self.dirty_datasets:
            if len(dirty_dataset) != len(self.clean_dataset):
                print("[E] Datasets should have same length.")

    def __len__(self):
        return len(self.clean_dataset) * len(self.dirty_datasets)

    def __getitem__(self, index):
        ds_idx = 0
        while index > len(self.dirty_datasets[ds_idx]) - 1:
            ds_idx += 1
            index -= len(self.dirty_datasets[ds_idx])

        return self.dirty_datasets[ds_idx][index], self.clean_dataset[index]