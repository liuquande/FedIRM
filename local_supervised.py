import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
from options import args_parser
import copy
from utils import losses
from confuse_matrix import get_confuse_matrix

args = args_parser()


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, image, label = self.dataset[self.idxs[item]]
        return items, index, image, label


class SupervisedLocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        self.epoch = 0
        self.iter_num = 0
        self.confuse_matrix = torch.zeros((args.num_classes, args.num_classes)).cuda()
        self.base_lr = args.base_lr

    def train(self, args, net, op_dict):
        net.train()
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4
        )
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr

        loss_fn = losses.LabelSmoothingCrossEntropy()
        self.confuse_matrix = torch.zeros((args.num_classes, args.num_classes)).cuda()
        # train and update
        epoch_loss = []
        print("begin training")
        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)
            print(iter_max)
            for i, (_, _, (image_batch, ema_image_batch), label_batch) in enumerate(
                self.ldr_train
            ):
                image_batch, ema_image_batch, label_batch = (
                    image_batch.cuda(),
                    ema_image_batch.cuda(),
                    label_batch.cuda(),
                )
                ema_inputs = ema_image_batch
                inputs = image_batch
                activations, outputs = net(inputs)
                _, aug_outputs = net(ema_inputs)
                with torch.no_grad():
                    self.confuse_matrix = self.confuse_matrix + get_confuse_matrix(
                        outputs, label_batch
                    )
                loss_classification = loss_fn(outputs, label_batch.long()) + loss_fn(
                    aug_outputs, label_batch.long()
                )
                loss = loss_classification
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                self.iter_num = self.iter_num + 1

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            print(epoch_loss)
        with torch.no_grad():
            self.confuse_matrix = self.confuse_matrix / (1.0 * args.local_ep * iter_max)
        return (
            net.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(self.optimizer.state_dict()),
        )
