# reference: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

        print("= concatenated dataset =")
        print(f"number of datasets: {len(self.datasets)}")
        
        for idx, ds in enumerate(self.datasets):
            print(f"{idx} {ds} {len(ds)}")

        print("= end of concatenated dataset =")
        

    def __getitem__(self, i):
        idx = 0
        while i > (len(self.datasets[idx]) - 1):
            i -= len(self.datasets[idx])
            idx += 1
        return self.datasets[idx][i]

    def __len__(self):
        return sum([len(d) for d in self.datasets])

if __name__ == "__main__":
    def print_loader(loader):
        for num in loader:
            print(num)
        print()
    
    from torch.utils.data import DataLoader
    class NumberDataset(Dataset):
        def __init__(self, num) -> None:
            super().__init__()
            self.data = []

            for i in range(num):
                self.data.append(i)
        
        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]
    
    ds1 = NumberDataset(10)
    ds2 = NumberDataset(5)
    ds3 = NumberDataset(3)

    loader1 = DataLoader(ds1, shuffle=False, batch_size=100)
    loader2 = DataLoader(ds2, shuffle=False, batch_size=100)
    loader3 = DataLoader(ds3, shuffle=False, batch_size=100)

    print_loader(loader1)
    print_loader(loader2)
    print_loader(loader3)
    

    ds = ConcatDataset(ds1, ds2, ds3)
    loader = DataLoader(ds, shuffle=False, batch_size=100)

    print_loader(loader)

    
