# -*- coding: utf-8 -*-
"""
This is the code for using trained models on CUDA or my personal computer
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os

#load the model architecture
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
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

#%%define the transformation function
from torchvision.transforms import v2 as T

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    #introduce normalization
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

import torch
device = torch.device('cpu')
num_classes = 2
#Here we load the architecture
model=get_model_instance_segmentation(num_classes)
#We load the the trained model
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
#And move it to CPU
model.to(device)

import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image

directory = r"...\images_for_illiustration".replace('\\', '/')

pictures = []

for root, dirs, files in os.walk(directory):
    pictures = files

directory = directory + '//'


#%%Here we define the function that removes ALL overlapped masks
#Here we calculate the Intersection over Union of any two given
def mask_iou(m1: torch.Tensor, m2: torch.Tensor) -> float:
    """
    Compute the IoU between two binary masks m1, m2.
    Each mask should be shape [H, W] or can be cast to bool.
    """
    m1 = m1.bool()
    m2 = m2.bool()
    
    intersection = (m1 & m2).sum().float()
    union = (m1 | m2).sum().float()
    
    return (intersection / union).item() if union > 0 else 0.0

#%%Now we remove all the masks that even touch one another
def remove_all_overlapping_masks(pred, iou_threshold=0):
    """
    Remove ALL masks that overlap above `iou_threshold`.
    We set it to 0 to avoid ANY kind of overlapping flakes.
    
    Any time two masks have IoU > iou_threshold, BOTH masks get removed.
    
    Args:
      pred (dict): 
        Must contain:
          - "masks": [N, 1, H, W] or [N, H, W]
          - "scores": [N] (optional)
          - "labels": [N] (optional)
          - "boxes": [N, 4] (optional)
      iou_threshold (float): Overlap threshold above which masks are discarded.
    
    Returns:
      dict: A filtered version of `pred` with all overlapping masks removed.
    """
    masks = pred["masks"]
    
    # If masks have shape [N, 1, H, W], squeeze the channel dimension
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]  # shape: [N, H, W]
    
    N = masks.shape[0]
    
    # Keep track of which masks are "overlapped" and should be removed
    overlapped = torch.zeros(N, dtype=torch.bool)
    
    # Compare every mask with every other (upper triangular to avoid double-count)
    for i in range(N):
        # if overlapped[i]:
        #     # Already flagged for removal, skip
        #     continue
        for j in range(i + 1, N):
            # if overlapped[j]:
            #     # Already flagged for removal, skip
            #     continue
            iou_val = mask_iou(masks[i], masks[j])
            if iou_val > iou_threshold:
                # Mark both i and j as overlapped (remove them)
                overlapped[i] = True
                overlapped[j] = True
    
    # Indices of masks we want to keep (those NOT overlapped)
    keep_indices = torch.nonzero(~overlapped).squeeze(1)
    
    # Build the filtered dict
    pred_filtered = {}
    pred_filtered["masks"] = pred["masks"][keep_indices]
    for key in ["scores", "labels", "boxes"]:
        if key in pred:
            pred_filtered[key] = pred[key][keep_indices]
    
    return pred_filtered

#%% Alternatively, we can have bbox based rejection
#Torch vision already has a function called non-maximum suppression (NMS)
def box_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    box1, box2: Tensors or lists in the format [x1, y1, x2, y2].
    (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner.

    Returns:
        float: IoU value between 0 and 1.
    """
    # box is [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute the area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute the intersection coordinates
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # Compute the area of intersection
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Compute union
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0  # Avoid division by zero

    return float(intersection) / float(union)

#A helper function that filters out any overlapping bounding boxes
def apply_nms(pred, iou_threshold=0):
    """
    Performs Non-Maximum Suppression (NMS) on the predicted boxes.
    
    Arguments:
        pred (dict): A dictionary containing keys 'boxes', 'scores', 'labels', and (optionally) 'masks'.
        iou_threshold (float): IoU threshold for deciding whether boxes overlap.
        
    Returns:
        dict: The same prediction dictionary but filtered so that only the kept
              boxes (and associated data) remain.
    """
    boxes = pred["boxes"]       # [N, 4]
    N = boxes.shape[0]
    
    # We use a custom box_iou to calculate the IoU of any two bounding boxes
    # and subsequently remove them from the considerations
    
    # Keep track of which boxes are overlapped
    overlapped = torch.zeros(N, dtype=torch.bool)
    
    # Compare every box with every other box
    for i in range(N):
        # if overlapped[i]:
        #     # Already marked as overlapping, skip
        #     continue
        for j in range(i + 1, N):
            # if overlapped[j]:
            #     # Already marked as overlapping, skip
            #     continue
            iou_val = box_iou(boxes[i], boxes[j])
            if iou_val > iou_threshold:
                # Mark both i and j as overlapped
                overlapped[i] = True
                overlapped[j] = True
                
    # Indices we keep are those that were never marked overlapped
    keep_indices = torch.nonzero(~overlapped).squeeze(1)
    
    # Create a filtered prediction dict
    pred_filtered = {}
    for key in pred:
        pred_filtered[key] = pred[key][keep_indices]

    return pred_filtered

#%% Here we define the function that omits the bounding boxes of the objects too close to the boundry
def fit(boxes, width, height, tolerance_level):
    """

    Parameters
    ----------
    box : bounding box.
    width : width if the picture.
    height : height of picture.
    tolerance_level : the minimum # of pixels to the edge.

    Returns
    -------
    fits : boolean value showing if the boxes are too close to the edges.

    """
    o = []
    fits = False
    for i in boxes:
        #For each bounding box i has the structure [x_min, y_min, x_max, y_max]
        #First check if it is close to the left side
        if i[0] < tolerance_level:
            fits = True
        #check if it is close to the right side
        elif i[2] > width - tolerance_level:
            fits = True
        #check if it is close to the bottom 
        elif i[3] > height - tolerance_level:
            fits = True
        #check if it is close to the top 
        elif i[1] < tolerance_level:
            fits = True
        o.append(fits)
    return fits


#Here we remove the bboxes
def ovrl(pred, tolerance_level):
    
    #pred["boxes"] shape is [N, 4]
    #we now convert pred["boxes"] tensor into a list of bounding boxes, which are of the form [x_min, y_min, x_max, y_max]
    #This is done to make analysis easier
    boxes = []
    for box in pred["boxes"]:
        boxes.append(box.long().numpy().tolist())
    
    #Here we extract the dimensions of the image. They are stored
    #in a tensor of a shape (N, height, width)
    #N is the number of predictions. There is the same number of masks as bounding boxes
    N, height, width = pred['masks'].shape
    
    # Keep track of boxes that need to be removed
    removal = torch.zeros(N, dtype=torch.bool)
    
    # Check every box
    for i in range(N):
        # Mark the box as too close to the edge
        removal[i] = fit([boxes[i]], width, height, tolerance_level)
                
    # Indices we keep are those that were never marked overlapped
    keep_indices = torch.nonzero(~removal).squeeze(1)
    
    # Create a filtered prediction dict
    pred_filtered = {}
    for key in pred:
        pred_filtered[key] = pred[key][keep_indices]
        
    return pred_filtered
    

#%%Here nanoplatelet recognition happens
eval_transform = get_transform(train=False)
model.eval()

statistics = []

from statistics_my import output_stats
import time

start_time = time.time()

prediction_labels = []

#define the prediction thresholds
score_threshold = 0.9     #Lowest acceptable bbox prediction

overlap_threshold = 0       #Highest allowed mask or box overlap

mask_threshold = 0.4        #Lowest pixel mask probability prediction
                            #since the masks are probabilistic
                            
tolerance_level = 5         #Describes how many pixels at most the bounding box must be to be
                            #considered too close to the edge

pixel_size = 18.92          #Pixel size
units = 'nm'                #units

empty_masks = 0

scores = []

# pred = 0

#The data should be analyzed as the model finishes analyzing each image.
#There will be too much data for the computer to handel if the data is 
#analyzed after the predictions are made
for ima in pictures:
    #The original images are black and white, hence it is enough to use a single color channel of the picture
    empty = 0
    sts = None
    image = read_image(directory + ima)[:1]
    with torch.no_grad():
        x = eval_transform(image).to(device)
        # convert RGBA -> RGB and move to device
        predictions = model([x, ])
        pred = predictions[0]
    
    #Keep a single channel to make the images grayscale. Reduces computational resources
    image = image[:1, ...]
    
    #Here we filter low score bounding boxe predictions
    if 'scores' in pred:
        keep_score = pred['scores'] > score_threshold
        for key in ['boxes', 'labels', 'masks', 'scores']:
            if key in pred:
                pred[key] = pred[key][keep_score]
                
    #Turn the probabilistic masks into the binary ones
    pred["masks"]  = (pred["masks"] > mask_threshold).squeeze(1)
    
    
    # Now remove all the overlapping masks
    pred_filtered = remove_all_overlapping_masks(
        pred, 
        iou_threshold=overlap_threshold
    )
    pred = pred_filtered
    
    
    # Here we reject bounding boxes that are too close to the edge
    pred = ovrl(pred, tolerance_level)
    
    #Now apply bounding box removal
    # pred = apply_nms(pred, iou_threshold=overlap_threshold)
       
    #This will draw the bounding boxes and will put the score as well as its label.
    pred_labels = [
        f"Score: {score:.3f} \n #: {i} \n  lbl: {label}" for label, score, i  in zip(pred["labels"], pred["scores"], range(len(pred['scores'])))
        ]
    
    pred_boxes = pred["boxes"].long()
    prediction_labels.append(pred_labels)
    
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red", font_size=20)
    
    #This will draw the masks on top of the areas where the model thinks it found a platelet
    masks = pred['masks']
    # masks = (pred["masks"] > mask_threshold).squeeze(1)
    # masks = masks[:len(pred_labels)]
    
    #We only want to draw predictions if there are any predictions to draw
    if pred['masks'].shape[0] != 0:
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="green")
    
    
    
    #The following extracts the bounding box information for statistical analysis.
    #It is converted to a numpy format for easier use.
    boxes = []
    for box in pred["boxes"].to(torch.device('cpu')):
        boxes.append(box.long().numpy().tolist())

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 4), dpi=500)
    fig.suptitle(f'{ima}, {len(pred_labels)}')
    axs[0].imshow(image.permute(1, 2, 0), cmap = 'gray')
    axs[1].imshow(output_image.permute(1, 2, 0), cmap = 'gray')
    axs[0].axis('off')
    axs[1].axis('off')
    fig.subplots_adjust(wspace=0)
    #Let us declare where we save model's visualised predictions
    path = "predictions"
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{ima}", bbox_inches='tight', dpi=400)
    plt.close(fig)
    # plt.show()
    print(ima)
    
    try:
        sts, empty = output_stats(
            image[0], 
            masks, 
            boxes, 
            tolerance_level, 
            pred['masks'].shape[1], 
            pred['masks'].shape[2], 
            pred['scores'], 
            units, 
            ima, 
            pixel_size
            )
    except Exception:
        print(f"WARNING! AT LEAST ONE EMPTY MASK IS EMPTY! SKIPPING {ima}")
    
    empty_masks += empty
    
    scores.append(pred['scores'].float().numpy().tolist())
    statistics.append(sts)
    


#Here we declare the location where the program is going to save statistical analysis graphs
path_to_save_visuals = ".../test/images"
type_t = 'demo'
from statistics_visualisation import dataset_analysis
dataset_analysis(statistics, units, pixel_size, path_to_save_visuals, type_t)
#This function creates a graph of the score distribution
#Score in this context means the model's assigned probability that the recognised object is a nanoplatelet
from score_analysis import score_analysis
score_analysis(scores, path_to_save_visuals, type_t)
    
print("Time taken:", (time.time() - start_time)/60)






































