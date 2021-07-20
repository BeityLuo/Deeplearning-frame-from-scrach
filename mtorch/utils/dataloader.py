from mtorch.utils.dataset import Dataset


class DataLoader():
    def __init__(self, dataset, batch_size = 1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle=shuffle

    def __iter__(self):
        self.idx = 0
        self.dataset.set_idx(0)
        if self.shuffle:
            self.dataset.shuffle()
        return self

    def __next__(self):
        if self.idx < len(self.dataset):
            imgs = self.dataset.get_next_imgs(num=self.batch_size)
            targets = self.dataset.get_next_labels(num=self.batch_size)
            self.idx += self.batch_size
            return (imgs, targets)
        else:
            raise StopIteration


        
