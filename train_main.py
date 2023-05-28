from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
from FedAvg import FedAvg
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import DenseNet121
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from torch.utils.data import DataLoader

args = args_parser()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def split(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def test(epoch, save_mode_path):
    checkpoint_path = save_mode_path

    checkpoint = torch.load(checkpoint_path)
    net = DenseNet121(
        out_size=args.num_classes, mode=args.label_uncertainty, drop_rate=args.drop_rate
    )
    model = net.cuda()
    model.load_state_dict(checkpoint["state_dict"])
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_dataset = dataset.CheXpertDataset(
        root_dir=args.root_path,
        csv_file=args.csv_file_test,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    AUROCs, Accus, Senss, Specs, _, _ = epochVal_metrics_test(
        model, test_dataloader, thresh=0.4
    )
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()

    return AUROC_avg, Accus_avg, Senss_avg, Specs_avg


AUROCs = []
Accus = []
Senss = []
Specs = []

match args.mode:
    case "sup":
        supervised_user_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        unsupervised_user_id = []
    case "ssfl":
        supervised_user_id = [0, 1]
        unsupervised_user_id = [2, 3, 4, 5, 6, 7, 8, 9]
    case "unsup":
        supervised_user_id = []
        unsupervised_user_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

flag_create = False
print("done")

if __name__ == "__main__":
    snapshot_folder = f"./model/{args.dataset}_{args.mode}"
    os.makedirs(snapshot_folder, exist_ok=True)

    # create logs folder if not exist
    os.makedirs("./logs", exist_ok=True)

    # add a line end of the log file
    with open(f"./logs/{args.dataset}_log.txt", "a") as f:
        f.write(f"\n{100*'-'}\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.basicConfig(
        filename=f"./logs/{args.dataset}_log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_dataset = dataset.CheXpertDataset(
        root_dir=args.root_path,
        csv_file=args.csv_file_train,
        transform=dataset.TransformTwice(
            transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        ),
    )
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [0.875, 0.125])
    dict_users = split(train_dataset, args.num_users)
    net_glob = DenseNet121(
        out_size=args.num_classes,
        mode=args.label_uncertainty,
        drop_rate=args.drop_rate,
    )

    if len(args.gpu.split(",")) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[0, 1])
    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    optim_locals = []

    for i in supervised_user_id:
        trainer_locals.append(SupervisedLocalUpdate(args, train_dataset, dict_users[i]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())
        optimizer = torch.optim.Adam(
            net_locals[i].parameters(),
            lr=args.base_lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    for i in unsupervised_user_id:
        trainer_locals.append(
            UnsupervisedLocalUpdate(args, train_dataset, dict_users[i])
        )

    for com_round in range(args.rounds):
        print("begin")
        loss_locals = []

        if com_round * args.local_ep < 200:
            for idx in supervised_user_id:
                if com_round * args.local_ep > 20:
                    trainer_locals[idx].base_lr = 3e-4
                local = trainer_locals[idx]

                optimizer = optim_locals[idx]
                w, loss, op = local.train(args, net_locals[idx], optimizer)
                w_locals[idx] = copy.deepcopy(w)
                optim_locals[idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))

        if com_round * args.local_ep > 20:
            if not flag_create:
                print("begin unsup")
                for i in unsupervised_user_id:
                    w_locals.append(copy.deepcopy(w_glob))
                    net_locals.append(copy.deepcopy(net_glob).cuda())
                    optimizer = torch.optim.Adam(
                        net_locals[i].parameters(),
                        lr=args.base_lr,
                        betas=(0.9, 0.999),
                        weight_decay=5e-4,
                    )
                    optim_locals.append(copy.deepcopy(optimizer.state_dict()))
                flag_create = True
            for idx in unsupervised_user_id:
                local = trainer_locals[idx]
                optimizer = optim_locals[idx]
                w, loss, op = local.train(
                    args,
                    net_locals[idx],
                    optimizer,
                    com_round * args.local_ep,
                    avg_matrix,
                )
                w_locals[idx] = copy.deepcopy(w)
                optim_locals[idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            avg_matrix = trainer_locals[0].confuse_matrix
            for idx in supervised_user_id[1:]:
                avg_matrix = avg_matrix + trainer_locals[idx].confuse_matrix
            avg_matrix = avg_matrix / len(supervised_user_id)

        with torch.no_grad():
            print("begin fedavg")
            w_glob = FedAvg(w_locals)
            print(w_glob.keys())

        net_glob.load_state_dict(w_glob)
        for i in supervised_user_id:
            net_locals[i].load_state_dict(w_glob)
        if com_round * args.local_ep > 20:
            for i in unsupervised_user_id:
                net_locals[i].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print(loss_avg, com_round)
        logging.info(
            "Loss Avg {} Round {} LR {} ".format(loss_avg, com_round, args.base_lr)
        )
        if com_round % 10 == 0:
            save_mode_path = os.path.join(
                snapshot_folder, "epoch_" + str(com_round) + ".pth"
            )
            torch.save(
                {
                    "state_dict": net_glob.state_dict(),
                },
                save_mode_path,
            )
            AUROC_avg, Accus_avg, Senss_avg, Specs_avg = test(com_round, save_mode_path)
            logging.info("\nTEST Student: Epoch: {}".format(com_round))
            logging.info(
                "\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}".format(
                    AUROC_avg, Accus_avg, Senss_avg, Specs_avg
                )
            )
