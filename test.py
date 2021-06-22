import torch
from networks.models import DenseNet121
from options import args_parser
import numpy as np
from validation import epochVal_metrics_test
from torchvision import transforms
import os
from dataloaders import dataset
from  torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
import random

args = args_parser()
checkpoint_path = os.path.join('model/'  , 'epoch_0.pth') 

if __name__ == "__main__":
     checkpoint = torch.load(checkpoint_path)
     net = DenseNet121(out_size=5, mode=args.label_uncertainty, drop_rate=args.drop_rate)
     cudnn.benchmark = False
     cudnn.deterministic = True
     random.seed(args.seed)
     np.random.seed(args.seed)
     torch.manual_seed(args.seed)
     torch.cuda.manual_seed(args.seed)
     model = net.cuda()
     model.load_state_dict(checkpoint['state_dict'])
     normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
     test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
     
     test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)
     thre = np.arange(0.4,0.93,0.01)
     thre = list(thre)
     for thresh in thre:
          print('begin',thresh)
	
    
          AUROCs, Accus, Senss, Specs,pre,f1= epochVal_metrics_test(model, test_dataloader,thresh=thresh)  
          
          print(AUROCs)
          print(Accus)
          print(Senss)
          print(Specs)
          
          AUROC_avg = np.array(AUROCs).mean()
          Accus_avg = np.array(Accus).mean()
          Senss_avg = np.array(Senss).mean()
          Specs_avg = np.array(Specs).mean()
          f1_avg = np.array(f1).mean()
           
          print(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, f1_avg)
