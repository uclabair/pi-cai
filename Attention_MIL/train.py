from models import Attention
import os
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from dataset import PicaiDataset, make_loader
import re
from sklearn.metrics import recall_score

FIND_DEGREES = re.compile(r'PIRADS_(\d+)')
from picai_baseline.splits.picai import train_splits, valid_splits

torch.backends.cudnn.deterministic = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# print(torch.cuda.device_count())


class Trainer:
    def __init__(self):

        "=========================================== initialize ======================================================="
        self.random_seed = 1
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1" 
        os.environ["OMP_NUM_THREADS"] = "1" 


        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        # print(torch.cuda.device_count())
 
        "=========================================== create datasets ================================================"
        # for fold, ds_config in train_splits.items():
        #         print(f"Training fold {fold} has cases: {ds_config['subject_list']}")

        train_filenames = [] #list(train_splits.items())[0][1]['subject_list']
        for item in list(train_splits.items())[0][1]['subject_list']:
            if os.path.exists('/raid/mpleasure/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/' + item + '.nii.gz'):
                train_filenames.append(item)
            else:
                print('train not exist')

        val_filenames = [] #list(valid_splits.items())[0][1]['subject_list']
        for item in list(valid_splits.items())[0][1]['subject_list']:
            if os.path.exists('/raid/mpleasure/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/' + item + '.nii.gz'):
                val_filenames.append(item)
            else:
                print('val not exist')

        # path to anatomical annotations
        whole_prostate_path = '/raid/mpleasure/picai_workdir/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b'
        # path to images
        path_to_images = '/raid/mpleasure/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/'
        # path to labels
        path_to_labels = '/raid/mpleasure/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/'

        self.train_loader = make_loader(
                                    train_filenames,
                                    path_to_images,
                                    path_to_labels,
                                    whole_prostate_path,
                                    PicaiDataset,
                                    get_annotation_masks = False,
                                    shuffle_= True,
                                    batch_size = 1,
                                    phase = 'train',
                                    patch = False)

        self.val_loader = make_loader(
                                train_filenames,
                                path_to_images,
                                path_to_labels,
                                whole_prostate_path,
                                PicaiDataset,
                                get_annotation_masks = False,
                                shuffle_= False,
                                batch_size = 1,
                                phase = 'val',
                                patch = False
                            )

        # self.test_loader = make_loader(images_test, masks_test, pirads_test,  PicaiDataset,
        #                         shuffle_=False, batch_size=1, mode='val')   

        "============================================= create networks ================================================"
        
        self.network =  Attention()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = nn.DataParallel(self.network)
        self.network.to('cuda')

        "=========================================== create reconstruction losses ====================================="
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([5.9]).cuda())

        "=========================================== create optimizers ================================================"
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4) 
        "=========================================== tensorboard ==================================================="

        model_path = '/raid/eredekop/picai/saved_models/' + 'model_att1_{0}.pt' #'1114_test_short_attWS2_tiomore_corrGG_f0.pt'
        self.save = lambda ep: torch.save({
        'model': self.network.state_dict(),
        'epoch': ep,
        },model_path.format(ep))

    def train(self, epoch):
        "========================================== updating params ==============================================="
        losses = []
        Y_true = []
        Preds = []

        self.network.train(True)

        training_process = tqdm(self.train_loader, position=0, leave=True)
        print('Epoch: ', epoch)
        for i, (data) in enumerate(training_process):
            if i > 0:
                training_process.set_description("Loss: %.4f" %
                                                (np.mean(losses)))
            

            image = data['input_mris'].type('torch.FloatTensor').cuda()[0]
            label = data['cspca_label'].type('torch.FloatTensor').cuda()#.reshape((pirad.shape[0]*pirad.shape[1]))#[0]



            step_size = 0.5
            patch_size = [8,8,8]
            image_size = [image.shape[1], image.shape[2], image.shape[3]]

            target_step_sizes_in_voxels = [i * step_size for i in patch_size]

            num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

            steps = []
            for dim in range(len(patch_size)):
                # the highest step value for this dimension is
                max_step_value = image_size[dim] - patch_size[dim]
                if num_steps[dim] > 1:
                    actual_step_size = max_step_value / (num_steps[dim] - 1)
                else:
                    actual_step_size = 99999999999  # does not matter because there is only one step at 0

                steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
                steps.append(steps_here)

            Patches = []
            for l in steps[0]:
                for j in steps[1]:
                    for k   in steps[2]:
                        imgs_patch = image[:, l:l + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]].cpu().detach().numpy().astype(np.float32)
                        Patches.append(imgs_patch.astype(np.float32))

            imgs_patch = torch.tensor(np.array(Patches).astype(np.float32)).type('torch.FloatTensor').cuda()
            # print(imgs_patch.shape, label)
            logits, A = self.network(imgs_patch)
            logits = logits[:, 0].cuda()
           

            Y_true.append(label.cpu().detach().numpy().astype(np.uint8))
            Preds.append(torch.sigmoid(logits).cpu().detach().numpy())
            _loss = self.loss(logits.cuda(), label.cuda())


            losses.append(_loss.item())

            
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
        auc = roc_auc_score(np.array(Y_true).astype(np.uint8), np.array(Preds)) 
        acc = accuracy_score(np.array(Y_true).astype(np.uint8), np.array(Preds) >=0.5) 
        print('Train auc={0}, train acc={1}'.format(auc, acc))
        # if auc > 0.7:
            # self.save(epoch)
        # self.writer.add_scalar('Loss/train', np.mean(losses), global_step=epoch)
        # self.writer.add_scalar('Accuracy/train', acc, global_step=epoch)    
        # self.writer.add_scalar('AUC/train', auc, global_step=epoch)
           


    def val(self, epoch, best_acc):
        accs = []
        val_process = tqdm(self.val_loader, position=0, leave=True)
        self.network.train(False)
        Preds = []
        Y_true = []
        with torch.no_grad():
            for i_, (data) in enumerate(val_process):
                if i_ > 0:
                    val_process.set_description("Acc: %.4f" %
                                                    (np.mean(accs)))

                image = data['input_mris'].type('torch.FloatTensor').cuda()[0]
                label = data['cspca_label'].type('torch.FloatTensor').cuda()#.reshape((pirad.shape[0]*pirad.shape[1]))#[0]


                step_size = 0.5
                patch_size = [8, 8, 8]
                image_size = [image.shape[1], image.shape[2], image.shape[3]]

                target_step_sizes_in_voxels = [i * step_size for i in patch_size]

                num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

                steps = []
                for dim in range(len(patch_size)):
                    # the highest step value for this dimension is
                    max_step_value = image_size[dim] - patch_size[dim]
                    if num_steps[dim] > 1:
                        actual_step_size = max_step_value / (num_steps[dim] - 1)
                    else:
                        actual_step_size = 99999999999  # does not matter because there is only one step at 0

                    steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
                    steps.append(steps_here)


                Patches = []
                for l in steps[0]:
                    for j in steps[1]:
                        for k   in steps[2]:
                            imgs_patch = image[:, l:l + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]].cpu().detach().numpy()
                            Patches.append(imgs_patch.astype(np.float32))

                imgs_patch = torch.tensor(np.array(Patches).astype(np.float32)).type('torch.FloatTensor').cuda()

                logits, A = self.network(imgs_patch)#[:, :1], imgs_patch[:, 1:2], imgs_patch[:, 2:]) #[:, 2:]

                logits = logits[:, 0].cuda()

                Y_true.append(label.cpu().detach().numpy().astype(np.uint8))
                Preds.append(torch.sigmoid(logits).cpu().detach().numpy())


            auc = roc_auc_score(np.array(Y_true).astype(np.uint8), np.array(Preds)) 
            acc = accuracy_score(np.array(Y_true).astype(np.uint8), np.array(Preds) >=0.5) 
            self.scheduler.step(auc)
            print('Val auc={0}, val acc={1}'.format(auc, acc))
            if auc > best_acc:
                self.save(epoch)
                best_acc = auc
                print('saved')

            # self.writer.add_scalar('Accuracy/validation', acc, global_step=epoch)   
            # self.writer.add_scalar('AUC/validation', auc, global_step=epoch)   
            return best_acc


def main():      
    trainer = Trainer()
    best_acc = 0
    for epoch in np.arange(150):
        trainer.train(epoch)
            
        acc = trainer.val(epoch, best_acc)
        best_acc = acc
        # if epoch%2:
        # trainer.test(epoch)


if __name__ == '__main__':
    main()
