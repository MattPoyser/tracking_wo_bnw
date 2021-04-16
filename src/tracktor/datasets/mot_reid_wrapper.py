from torch.utils.data import Dataset, ConcatDataset

from .mot_reid import MOTreID


class MOTreIDWrapper(Dataset):
    """A Wrapper class for MOTSiamese.

    Wrapper class for combining different sequences into one dataset for the MOTreID
    Dataset.
    """

    def __init__(self, split, kwargs):
        # train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
        #                    'MOT17-10', 'MOT17-11', 'MOT17-13']
        train_sequences = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                           '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                           '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
                           '48', '49', '50', '51', '52', '53']

        if split == "train":
            sequences = train_sequences
        elif f"MOT17-{split}" in train_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT split not available.")

        dataset = []
        for seq in sequences:
            # dataset.append(MOTreID(seq, split=split, **kwargs))
            dataset.append(MOTreID(seq, **kwargs))

        self.split = ConcatDataset(dataset)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        return self.split[idx]
