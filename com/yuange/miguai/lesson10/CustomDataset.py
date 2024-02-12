from torch.utils.data import Dataset

class NumbersDataset(Dataset):
    def __init__(self, tranining= True):
        if tranining:
            self.samples = list(range(1, 1001))
        else:
            self.samples = list(range(1001, 1501))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# for epoch in range(epochs):
#     train(train_db)
#
#     if epoch % 10 == 0:
#         val_acc = evaluate(val_db)
#         if val_ass is the best:
#             save_ckpt()
#         if out_of_patience():
#             break
#
# load_ckpt()
# test_acc = evaluate(test_db)