from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import DenseNet121
from utils import losses, ramps
from utils.util import get_timestamp
from confuse_matrix import get_confuse_matrix, kd_loss

args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, image, label = self.dataset[self.idxs[item]]
        return items, index, image, label


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        net = DenseNet121(
            out_size=args.num_classes,
            mode=args.label_uncertainty,
            drop_rate=args.drop_rate,
        )
        if len(args.gpu.split(",")) > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])
        self.ema_model = net.cuda()
        for param in self.ema_model.parameters():
            param.detach_()
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 2e-4

    def train(self, args, net, op_dict, epoch, target_matrix):
        net.train()
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4
        )
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr

        self.epoch = epoch
        if self.flag:
            self.ema_model.load_state_dict(net.state_dict())
            self.flag = False
            print("done")

        epoch_loss = []
        print("begin training")
        for epoch in range(args.local_ep):
            batch_loss = []
            iter_max = len(self.ldr_train)

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

                _, outputs = net(inputs)

                with torch.no_grad():
                    ema_activations, ema_output = self.ema_model(ema_inputs)
                T = 10

                with torch.no_grad():
                    _, logits_sum = net(inputs)
                    for i in range(T):
                        _, logits = net(inputs)
                        logits_sum = logits_sum + logits
                    logits = logits_sum / (T + 1)
                    preds = F.softmax(logits, dim=1)
                    uncertainty = -1.0 * torch.sum(
                        preds * torch.log(preds + 1e-6), dim=1
                    )
                    uncertainty_mask = uncertainty < 2.0

                with torch.no_grad():
                    activations = F.softmax(outputs, dim=1)
                    confidence, _ = torch.max(activations, dim=1)
                    confidence_mask = confidence >= 0.3
                mask = confidence_mask * uncertainty_mask

                pseudo_labels = torch.argmax(activations[mask], dim=1)
                pseudo_labels = F.one_hot(pseudo_labels, num_classes=args.num_classes)
                source_matrix = get_confuse_matrix(outputs[mask], pseudo_labels)

                consistency_weight = get_current_consistency_weight(self.epoch)
                consistency_dist = (
                    torch.sum(losses.softmax_mse_loss(outputs, ema_output))
                    / args.batch_size
                )
                consistency_loss = consistency_dist

                loss = (
                    15 * consistency_weight * consistency_loss
                    + 15
                    * consistency_weight
                    * torch.sum(kd_loss(source_matrix, target_matrix))
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            timestamp = get_timestamp()

            self.epoch = self.epoch + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(epoch_loss)

        return (
            net.state_dict(),
            sum(epoch_loss) / len(epoch_loss),
            copy.deepcopy(self.optimizer.state_dict()),
        )
