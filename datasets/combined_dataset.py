from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, dirty, clean, transform = None) -> None:
        super().__init__()
        self.transform = transform
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
        
        dirty_item = self.dirty_datasets[ds_idx][index]
        clean_item = self.clean_dataset[index]

        if self.transform is not None:
            # convert CHW to HWC
            dirty_item = dirty_item.permute(1, 2, 0)
            clean_item = clean_item.permute(1, 2, 0)
            transformed = self.transform(image=dirty_item.numpy(), image0=clean_item.numpy())
            dirty_item = transformed['image']
            clean_item = transformed['image0']

        return dirty_item, clean_item