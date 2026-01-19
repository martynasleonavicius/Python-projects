"""
THIS CODE WAS ORIGINALLY TAKEN FROM
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
AND LATER MODIFIED
The link works as of 19/01/2026

For the training to work, the user has to have the following scripts in the directory:
    -engine.py
    -transforms.py
    -coco_eval.py
    -coco_utils.py
    -utils.py
    
These scripts were NOT made by me. They are open source and I put them on my GitHub page so that it would be easier to access them.
For more information on how to get them from an alterntive source, please check out the link above.
"""

from torchvision.io import read_image
import torch

import os

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from pycocotools.coco import COCO


class DatasetClass(torch.utils.data.Dataset):
    def __init__(self, root, annotations, transformation):
        self.root = root
        self.transformation = transformation

        
        #create the annotation instance
        self.coco = COCO(annotations)
        
        #Fetches image ids
        self.ids = list(sorted(self.coco.imgs.keys()))
        print(self.ids)

    def __getitem__(self, idx):
        coco = self.coco
        
        #Get image ID
        img_id = self.ids[idx]
        
        #get annotation id from coco
        #This is a list:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        
        #Target coco_annotation file for an image
        #This will be a dict
        coco_annotation = coco.loadAnns(ann_ids)
        
        #Path for input images
        path = coco.loadImgs(img_id)[0]['file_name']
        #open the input image. We use only one channel to reduce the computing power needs
        #since all the pictures are black and white anyway
        img = read_image(os.path.join(self.root, path))[:1, :, :]
        
        #For storing the segmentations
        segmentation = []
        
        #Number of objects in the image
        num_objs = len(coco_annotation)
        
        
        #we need to change the bounding box format from
        #[xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            segmentation.append(coco_annotation[i]['segmentation'])
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img), dtype=torch.float32)
        
        #Here we'll have labels
        labels = torch.ones((num_objs), dtype=torch.int64)
        
        #Change the tensor ids into tensors (Why?)
        # img_id = torch.tensor([img_id])
        
        #Save the areas of the box
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.tensor(areas)
            
        iscrowd = torch.zeros((num_objs), dtype=torch.int64)
        
        #There is a function that can convert the annotations into masks
        masks = []
        for annotation in coco_annotation:
            rle = self.coco.annToMask(annotation)
            masks.append(rle) 
        masks = tv_tensors.Mask(masks)
        
        my_annotation = {
            "boxes" : boxes,
            'labels': labels,
            'image_id':img_id,
            'segmentation':segmentation,
            'masks':masks,
            'area' : areas,
            'iscrowd':iscrowd,
            }
        
        #transform images
        if self.transformation is not None:
            try:
                img, my_annotation = self.transformation(img, my_annotation)
            except TypeError:
                pass

        return img, my_annotation
    

    def __len__(self):
        return len(self.ids)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=5)
    # trainable_backbone_layers = 5
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

from torchvision.transforms import v2 as T
from torchvision.transforms import AutoAugment
import random

#%%
def get_transform(train):
    transforms = []
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        # if random.random() >= 0.5:
        #     transforms.append(T.CutMix(alpha=1.0))
        # if random.random() < 0.1:
        #     transforms.append(T.MixUp(alpha=1.0))
        if random.random() > 0.5:
            transforms.append(T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)))
        if random.random() > 0.5: 
            transforms.append(
                    T.ColorJitter(brightness=0.2, contrast=random.random(), saturation = random.random(), hue = random.randint(0,5)/10)
                )
        transforms.append(T.RandomGrayscale(p=0.5))
        
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.Normalize(mean=[0.485], std=[0.229]))
    #mean val ,0.456, 0.406
    #Corresponding std val , 0.224, 0.225
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


from engine import train_one_epoch, evaluate

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

train_data_dir = r'...\\training image location'.replace('\\', '/')
train_coco_annotation = r'...\\annotation file location'.replace('\\', '/')

dataset = DatasetClass(train_data_dir,
                           train_coco_annotation, 
                           get_transform(train=True)
                           )
dataset_test = DatasetClass(train_data_dir,
                           train_coco_annotation, 
                           get_transform(train=False)
                           )

#Here we split the image dataset into training and validation datasets
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-1])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-1:])

import utils
import logging
from torch_snippets.torch_loader import Report

#%%
def training(num_epochs, batch_size, learning_rate, momentum, weight_decay, step_size, gamma, variation_type, iteration, log_folder):
    tot_time = time.time()
    #Define logging file
    torch.manual_seed(42)
    os.makedirs(log_folder, exist_ok=True)
    file_loc = os.path.join(log_folder, f'{variation_type}_{iteration}.log')


    logging.basicConfig(
        filename=file_loc,
        filemode='w',           # 'w' overwrites file on each run; use 'a' to append
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        #If this doesn't exist, no new logging files will be created
        force=True
    )
    
    #Empty cache before repeating training to avoid running out of memory
    torch.cuda.empty_cache()
    
    start = time.time()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        #CHANGE
        # sampler=sampler,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=utils.collate_fn,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )
    
    #WE need to convert the exact dataset_test to the coco format to avoid AssertionError: Results do not correspond to current coco set
    # coco_test = coco_utils.convert_to_coco_api(dataset_test)
    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    #Creates the list of parameters that require gradient computation to save computing resources
    params = [p for p in model.parameters() if p.requires_grad]
    #initializes the optimizer (stochastic gradient descent)
    optimizer = torch.optim.SGD(
        #Uses the list of gradients that require gradient computations
        params,
        #CHANGE
        #Learning rate
        # lr=round((0.001-w/100000), 4),
        #Governs how fast the weights are updated. Lower rates require more epochs
        #but results in more accurate results
        lr=learning_rate,
        #Needed to stabalize the optimization process. (WHY???)
        momentum=momentum,
        #This sets up L2 regularization. Helps to prevent overfitting.
        #L2 regularization pushes weights to converge towards 0.
        #increases penalty in the loss function.
        weight_decay=weight_decay
    )
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    
    
    logging.info("starting training...")
    logging.info("This is a basic training of the combined annotations (CA). All hyperparameters will be varied.")
    logging.info(f"hyperparameters: num_epochs: {num_epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}, momentum: {momentum}, weight_decay: {weight_decay}, step_size: {step_size}, gamma: {gamma}")
    logging.info(f"Dataset size:{len(dataset)}")
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_metrics =train_one_epoch(
                model, 
                optimizer, 
                data_loader, 
                device, 
                epoch, 
                print_freq=1
                )
            
        # Extract MetricLogger object to a dictionary
        train_metrics_dict = {name: meter.global_avg for name, meter in train_metrics.meters.items()}
        loss_info = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics_dict.items()])
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Training - {loss_info}")
        
        # update the learning rate
        lr_scheduler.step()
            
        # evaluate on the test dataset
        val_metrics = evaluate(
            model, 
            data_loader_test, 
            # data_loader,
            device=device
            )
        val_info = []
        for iou_type, coco_eval in val_metrics.coco_eval.items():
            stats = coco_eval.stats  # This is a 1D array of floats (typical shape: 12)
            
            # Format whichever metrics you care about:
            ap_50_95     = stats[0]
            ap_50        = stats[1]
            ap_75        = stats[2]
            ap_small     = stats[3]
            ap_medium    = stats[4]
            ap_large     = stats[5]
            ar_iou_50_95 = stats[6]
            ar_iou_50    = stats[7]
            ar_iou_75    = stats[8]
            ar_small     = stats[9]
            ar_medium    = stats[10]
            ar_large     = stats[11]
        
            # Build a nicely formatted string:
            iou_info = (
                f"IoU Type: {iou_type}, "
                f"AP(0.5:0.95): {ap_50_95:.4f}, "
                f"AP(0.5): {ap_50:.4f}, "
                f"AP(0.75): {ap_75:.4f}, "
                f"AP(small): {ap_small:.4f}, "
                f"AP(medium): {ap_medium:.4f}, "
                f"AP(large): {ap_large:.4f}, "
                f"AR(0.5:0.95): {ar_iou_50_95:.4f},"
                f"AR(0.5): {ar_iou_50:.4f},"
                f"AR(0.75): {ar_iou_75:.4f},"
                f"AR(small): {ar_small:.4f},"
                f"AR(medium): {ar_medium:.4f},"
                f"AR(large): {ar_large:.4f}"
            )
        
            val_info.append(iou_info)
        
        val_info = " | ".join(val_info)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Validation - {val_info}")
        logging.info('\n')
    
    
    end = time.time()
    delta = round((end - start)/60, 2)
    print("That's it!")
    print(f"Time taken: {(time.time() - tot_time)/60}")    
    logging.info(f"Time taken: {(time.time() - tot_time)/60}")
    logging.shutdown()
    
    return model

#%%
# Define the hyperparameters
import time
import os
import utils
import logging
from torch_snippets.torch_loader import Report
from engine import train_one_epoch, evaluate


num_epochs = 1
batch_size = 5
learning_rate = 0.002
momentum = 0.9
weight_decay=0.0005
step_size=3
gamma=0.7

def err_message(x):
    print(f"ERR.: {x}")
    
start_time=time.time() 
#Finally we train the models
model = training(num_epochs, batch_size, learning_rate, momentum, weight_decay, step_size, gamma, "test", 0, 'log_folder')
torch.save(model.state_dict(), 'test.pth')


# THE CODE BELLOW USED TO PRODUCE VARYING MODELS WHICH WERE TRAINED ON DIFFERENT HYPERPARAMETERS. THIS WAS USED TO EVALUATE WHICH SET OF HYPERPARAMETERS PRODUCED THE BEST MODEL ON THE GIVEN DATA
#Run the same num_epochs multiple times -> see how it varies.
# for i in range(0, 5):
#     print(i)
#     variation_type = "running_the_same_batch_size"
#     try:
#         #model = training(30, batch_size, learning_rate, momentum, weight_decay, step_size, gamma, variation_type, i)
#         pass
#     except Exception:
#          err_message(variation_type)

#model = training(10, batch_size, learning_rate, momentum, weight_decay, step_size, gamma, variation_type, i)
#torch.save(model.state_dict(), '10_epoch.pth')

#Varying batch_size (1, 2, 3, 4, 5, 6)
# for i in range(1, 7):
#     print(i)
#     variation_type = "varying_batch_size"
#     try:
#         model = training(10, i, learning_rate, momentum, weight_decay, step_size, gamma, variation_type, i)
#         pass
#     except Exception:
#         print('ERR.: varying_learning_rate')
        
    
# #Varying momentum (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
# for i in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0]:
#     print(i)
#     variation_type = "varying_momentum"
#     try:
#         model = training(10, batch_size, learning_rate, i, weight_decay, step_size, gamma, variation_type, i)
#         pass
#     except Exception:
#         err_message(variation_type)
    
    
# #Varying weight_decay (0.0, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009)
# for i in [0.0, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
#     print(i)
#     variation_type = "varying_weight_decay"
#     try:
#         model = training(10, batch_size, learning_rate, momentum, i, step_size, gamma, variation_type, i)
#         pass
#     except Exception:
#         err_message(variation_type)


# #Varying gamma (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
# for i in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
#     print(i)
#     variation_type = "varying_gamma"
#     try:
#         #model = training(10, batch_size, learning_rate, momentum, weight_decay, step_size, i, variation_type, i)
#         pass
#     except Exception:
#         err_message(variation_type)
        
        
# #Varying learning_rate (0.001, 0.002, 0.003, 0.004, 0.005)
# for i in [0.001, 0.002, 0.003, 0.004, 0.005]:
#     print(i)
#     variation_type = "varying_learning_rate"
#     try:
#         model = training(10, batch_size, i, momentum, weight_decay, step_size, gamma, variation_type, i)
#         pass
#     except Exception:
#         err_message(variation_type)
print("Time taken", (time.time()-start_time)/3600)




































