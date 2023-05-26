import argparse
from networks.models import DenseNet121


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ssfl",
        choices=["ssfl", "sup", "unsup"],
        help="mode of training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="isic2019",
        choices=["ham10000", "brain", "isic2019"],
        help="name of dataset",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/media/undergrad/Data/tubitak-data/isic2019/all_images",
        help="dataset root dir",
    )
    parser.add_argument(
        "--csv_file_train",
        type=str,
        default="/media/undergrad/Data/tubitak-data/isic2019/train.csv",
        help="training set csv file",
    )
    # parser.add_argument('--csv_file_val', type=str, default='/research/pheng4/qdliu/Semi/dataset/skin/validation.csv', help='validation set csv file')
    parser.add_argument(
        "--csv_file_test",
        type=str,
        default="/media/undergrad/Data/tubitak-data/isic2019/test.csv",
        help="testing set csv file",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="number of classes",
    )
    parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
    parser.add_argument("--drop_rate", type=int, default=0.2, help="dropout rate")
    parser.add_argument(
        "--ema_consistency", type=int, default=1, help="whether train baseline model"
    )
    parser.add_argument(
        "--base_lr", type=float, default=2e-4, help="learning rate for baseline model"
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--local_ep", type=int, default=1, help="local epoch")
    parser.add_argument("--num_users", type=int, default=10, help="local epoch")
    parser.add_argument("--rounds", type=int, default=101, help="local epoch")

    ### tune
    parser.add_argument("--resume", type=str, default=None, help="model to resume")
    parser.add_argument("--start_epoch", type=int, default=0, help="start_epoch")
    parser.add_argument("--global_step", type=int, default=0, help="global_step")
    ### costs
    parser.add_argument(
        "--label_uncertainty", type=str, default="U-Ones", help="label type"
    )
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
    parser.add_argument("--consistency", type=float, default=1, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=30, help="consistency_rampup"
    )
    args = parser.parse_args()

    match args.dataset:
        case "ham10000":
            args.root_path = "/media/undergrad/Data/tubitak-data/ham10000/all_images"
            args.csv_file_train = (
                "/media/undergrad/Data/tubitak-data/ham10000/train.csv"
            )
            args.csv_file_test = "/media/undergrad/Data/tubitak-data/ham10000/test.csv"
            args.num_classes = 7
        case "isic2019":
            args.root_path = "/media/undergrad/Data/tubitak-data/isic2019/all_images"
            args.csv_file_train = (
                "/media/undergrad/Data/tubitak-data/isic2019/train.csv"
            )
            args.csv_file_test = "/media/undergrad/Data/tubitak-data/isic2019/test.csv"
            args.num_classes = 8
        case "brain":
            args.root_path = "/media/undergrad/Data/tubitak-data/brain/RSNA-ICH/organized/stage_2_train"
            args.csv_file_train = (
                "/media/undergrad/Data/tubitak-data/brain/RSNA-ICH/training.csv"
            )
            args.csv_file_test = (
                "/media/undergrad/Data/tubitak-data/brain/RSNA-ICH/testing.csv"
            )
            args.num_classes = 5

    return args
