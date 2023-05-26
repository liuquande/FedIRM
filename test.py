import torch
from networks.models import DenseNet121
from options import args_parser
import numpy as np
from validation import epochVal_metrics_test
from torchvision import transforms
import os
from dataloaders import dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pandas as pd

args = args_parser()

checkpoint_path = os.path.join(f"./model/{args.dataset}_{args.mode}", "epoch_100.pth")

if __name__ == "__main__":
    checkpoint = torch.load(checkpoint_path)
    net = DenseNet121(
        out_size=args.num_classes, mode=args.label_uncertainty, drop_rate=args.drop_rate
    )
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
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
        num_workers=8,
        pin_memory=True,
    )

    all_preds = pd.DataFrame(columns=["thresh", "AUROC", "Accu", "Sens", "Spec", "f1"])

    thre = np.arange(0.4, 0.93, 0.01)
    thre = list(thre)
    for thresh in thre:
        print("begin", thresh)

        AUROCs, Accus, Senss, Specs, pre, f1 = epochVal_metrics_test(
            model, test_dataloader, thresh=thresh
        )

        print(f"AUROCs: {AUROCs}")
        print(f"Accus: {Accus}")
        print(f"Senss: {Senss}")
        print(f"Specs: {Specs}")
        print(f"f1: {f1}")

        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        f1_avg = np.array(f1).mean()

        print(f"AUROC_avg: {AUROC_avg}")
        print(f"Accus_avg: {Accus_avg}")
        print(f"Senss_avg: {Senss_avg}")
        print(f"Specs_avg: {Specs_avg}")
        print(f"f1_avg: {f1_avg}")
        print(f"{100 * '='}")

        # concat the results
        all_preds = pd.concat(
            [
                all_preds,
                pd.DataFrame(
                    {
                        "thresh": [thresh],
                        "AUROC": [AUROC_avg],
                        "Accu": [Accus_avg],
                        "Sens": [Senss_avg],
                        "Spec": [Specs_avg],
                        "f1": [f1_avg],
                    }
                ),
            ]
        )

    os.makedirs("./test_csv", exist_ok=True)
    all_preds.to_csv(f"./test_csv/test_for_{args.dataset}_{args.mode}.csv", index=False)
