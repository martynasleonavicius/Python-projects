# -*- coding: utf-8 -*-


import torchvision
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


#NOTE! output bounding boxes are in the XYXY format.

#%%Ellipticity calculations
def calculate_eccentricity(masks):
    
    ellipticity = []
    anisotropy = []
    
    for mask in masks:
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # if len(contours) == 0:
        #     return None  # No object detected
    
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Fit an ellipse
        if len(largest_contour) >= 5:  # Minimum points for ellipse fitting
            ellipse = cv2.fitEllipse(largest_contour)
            (x, y), (major_axis, minor_axis), angle = ellipse
    
            # Compute ellipticity
            a = max(major_axis, minor_axis) / 2  # Semi-major axis
            b = min(major_axis, minor_axis) / 2  # Semi-minor axis
            ellipticity.append(np.sqrt(1 - (b**2 / a**2)))
            #Calculate the anisotropy here as well
            anisotropy.append(np.float64(b)/np.float64(a))
            
    return ellipticity, anisotropy
    
    
#%%Calculate the average background value

def avg_brightness(img, masks):
    """
    Requires a tensor
    
    Return: float
    """
    # Assuming masks is a PyTorch tensor of shape (14, 768, 1024)
    combined_mask = torch.any(masks > 0, dim=0).to(torch.uint8)  # Shape: (768, 1024)
    
    
    # Displays all the masks in the same picture
    # plt.imshow(combined_mask, cmap='gray')
    # plt.show()
    
    
    unmasked_pixels = img[combined_mask == 0]  # Extract pixels where mask == 0
    
    # Compute mean brightness
    average_brightness = unmasked_pixels.float().mean().item() #if unmasked_pixels.numel() > 0 else 0
    # print(f"Average Brightness (excluding masked areas): {average_brightness}")
    return average_brightness
    

#%%Now let's find the smallest value and claim that it's the thinnest layer!

def layer(img, masks):
    """
    

    Parameters
    ----------
    img : tensor
    masks : list of tensors

    Returns
    -------
    discr_flakes : Dictionary of tensors

    """
    #We will create a list populated with flake individual flake images
    flakes = []
    for mask in masks:
        # Create an empty image (same shape as img)
        flake = torch.zeros_like(img)  
        
        # Apply the mask: Set only the masked pixels to their original values
        flake[mask > 0] = img[mask > 0]
        
        flakes.append(flake)
        
        
    #Extract the smallest value
    smallest_value = 0
    for flake in flakes:
        nonzero_pixels = flake[flake > 0]  # Extract only nonzero pixels
        if nonzero_pixels.numel() > 0 and smallest_value < nonzero_pixels.numel():  # Check if there are nonzero pixels
            smallest_value = nonzero_pixels.min().item()  # Find the minimum nonzero pixel
    
    
    discr_flakes = []
    #Discretizing the pixel values
    for flake in flakes:
        flake_discretized = torch.ceil(flake / smallest_value)
        discr_flakes.append(flake_discretized)
    
    return discr_flakes

#%%Smaller functions to calculate some of the statistics
from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops

#Calculates the solidity of the predicted masks
def soliditys(area, masks):
    """

    Parameters
    ----------
    area : array of areas (int)
    discr_flakes : dictionary of tensors

    Returns
    -------
    solidity : array of floats

    """
    solidity = []
    for m in range(len(masks)):
        # img_np = discr_flakes[i].cpu().numpy()
        conv_hull = convex_hull_image(masks[m])
        area_conv_hull = np.count_nonzero(conv_hull)
        solidity.append(area[m]/area_conv_hull)
    return solidity

#Here we calculate the aspect ratio of the masks
def aspect_ratio(masks):
    # This is an old method to find the aspect ratio
    # aspect = []
    # for i in boxes:
    #     #REMEMBER, boxes format is x_min, y_min, x_max, y_max
    #     width = i[2] - i[0]
    #     height = i[3] - i[1]
    #     aspect.append(width/height)
    
    # We will need to use discr_flakes dictionary to remove the detected blank areas
    # Convert to a numpy array and make sure that the numbers are in uint8
    aspect = []
    w_h = []
    R = []
    for mask in masks:
        # mask_np = mask.cpu().numpy().astype(np.float32)
        #Change the mask_np shape into an acceptable format
        mask_np = mask
        
        # Extract the nonzero coordinates (y, x)
        points = np.argwhere(mask_np > 0)  # Shape: (N, 2)
        
        # Swap columns to (x, y) because OpenCV expects (x, y) order
        points = np.flip(points, axis=1)  # Convert (y, x) -> (x, y)
        
        # mask_np = mask_np.reshape(-1, 1, 2)
        rect = cv2.minAreaRect(points)  # (center (x, y), (width, height), angle) 
        R.append(rect)
        _, (w, h), rotation = rect
        if w < h:
            aspect.append(w/h)
        else:
            aspect.append(h/w)
        w_h.append([w, h])
    return aspect, R, w_h

#It just so happens that circularity also calculates the perimeter, area and circuliarity
def circularity(discr_flakes):
    # Calculate the circularity of the objects
    circ = []
    binar_masks = []
    for i in discr_flakes:
        # 1. Suppose `mask_tensor` is your single-channel mask in PyTorch (shape [H, W]) 
        #    with non-zero = object, zero = background. Convert to NumPy:
        # mask_np = i.cpu().numpy()
        
        # 2. Binarize the mask if not already (True/False):
        binary_mask = i > 0
        binar_masks.append(binary_mask)
        
        # 3. Label the (possibly multiple) connected components in the mask
        labels = label(binary_mask)
        
        # 4. Extract region properties
        regions = regionprops(labels)
        
        # For each labeled region, we can compute roundness (circularity)
        for region in regions:
            area = region.area          # number of pixels in the region
            perimeter = region.perimeter  # perimeter length of the region
            
            if perimeter > 0:
                #This is a 'commonly accepted' formula for circularity.
                roundness = 4.0 * np.pi * area / (perimeter * perimeter)
            else:
                roundness = 0  # degenerate case if perimeter is 0
        circ.append([int(area), float(perimeter), float(roundness), len(regions)])
    return circ, binar_masks
    
#%%
def plotas(masks):
    areas = []
    for mask in masks:
        try:
            areas.append(torch.count_nonzero(mask).item())
        except Exception:
            areas.append(np.count_nonzero(mask))
    
    return areas

#%%Produces the histogram and the average thickness of the individual flakes.

def combining_stats(discr_flakes, boxes, masks, height, width, tolerance_level, scores):
    
    # Example usage on all masks
    eccentricity, anisotropy = calculate_eccentricity(masks)
    values = []
    thickens_info = pd.DataFrame()
    
    for flake in discr_flakes:
        values.append(flake[flake > 0].cpu().numpy().flatten())
        # plt.hist(values[-1], edgecolor='black')
        # plt.show()
        
    thickness = [np.mean(i) for i in values]
    thickenss_std = [np.std(i) for i in values]
        
    # fits = fit(boxes, width, height, tolerance_level)
    aspect_rat, R, w_h = aspect_ratio(masks)
    circ, binary = circularity(discr_flakes)
    
    
    #Extract area, perimeter, circularity and how many regions exist
    arrr = []
    per = []
    roudness = []
    region_num = []
    for i in circ:
        arrr.append(i[0])
        per.append(i[1])
        roudness.append(i[2])
        region_num.append(i[3])
    solidity = soliditys(arrr, discr_flakes)
    
    areas = plotas(discr_flakes)
    
    thickens_info = pd.DataFrame(
        {'Average Thickness':thickness,
         'StD': thickenss_std, 
         'Area': arrr, 
         'Area_2_in_px': areas, 'regions': region_num,
         'Eccentricity': eccentricity, 'Solidity': solidity, 
         'Aspect Ratio Width/Height': aspect_rat, 
         'Width/Height': w_h,
         f'Perimeter': per, 'Circularity': roudness,
         'Bbox': boxes, #'Inside': fits
         'Anisotropy': anisotropy,
         'Scores': scores
         }
        )
    
    return thickens_info
    
#%% Remove prediction noise
def remove_small_components(mask, min_size=50):
    """
    Removes small connected components from a binary mask.
    
    :param mask: Binary mask (numpy array HxW, dtype=np.uint8)
    :param min_size: Minimum size (in pixels) to keep a component.
    :return: Cleaned binary mask.
    """
    # Ensure mask is binary (0 or 1)
    # mask = (mask > 0).astype(np.uint8)

    # kernel = np.ones((3,3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)

    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Create a new mask to store the cleaned version
    cleaned_mask = np.zeros_like(mask)

    # Iterate through components and keep only the large ones
    areas = []
    for i in range(1, num_labels):  # Skip background (label 0)
        areas.append(stats[i, cv2.CC_STAT_AREA])
    
    #Find the index of the largest area
    max_area_index = np.argmax(areas)
    
    #Draw the shape of the largest object on a blank mask
    cleaned_mask[labels == (max_area_index + 1)] = 1

    return cleaned_mask
#%% Running the code here

def output_stats(img, masks, boxes, tolerance_level, height, width, scores, units, ima, pixel_size = 18.92):
    
    #Lowest probability value for mask's pizel
    # threshold = 0.2

    #How close the bounding box to the edge for the detected object to be described as not fully fitting in.
    # tolerance_level = 5

    #Width of the picture
    # width = 1024

    #Height of the picture
    # height = 768    
    empty_masks = 0
    # img = read_image(f"{directory}\\_Electron_Image_203.png".replace('\\', '/'))[0]
    # average_brightness = avg_brightness(img, masks)
    
    #We subtract the average brightness from the image.
    #.float() ensures that the integer underflow does not happen.
    #Normalize the flake pixel brightness against the background value
    # img = torch.clamp(img.float()/average_brightness, min=0)
    
    #We need to convert all the masks into binary numpy arrays
    masks = [(mask > 0).cpu().numpy().astype(np.uint8) for mask in masks]
    
    #There might be a problem that an object's mask has disconnected regions (noise)
    #We will keep only the biggest region, because the disconnected areas are only a few pixels big.
    scores = scores.float().numpy().tolist()
    try:
        masks = [remove_small_components(mask.astype(np.uint8)) for mask in masks]
    except Exception:
        print("WARNING! EMPTY MASK DETECTED! \n CHECK empty_masks VARIABLE TO FIND OUT HOW MANY")
        empty_masks += 1
        
    try:    
        masks = [mask.astype(np.float32) for mask in masks if isinstance(mask, np.ndarray)]
    except Exception:
        print(f"WARNING! SOME MASKS WERE NOT CORRECTLY CONVERTED FROM torch.Tensor TO numpy.ndarray TYPE! CHECK statistics.py")
    
    discr_flakes = layer(
        img, 
        masks, 
        #threshold
        )
    
    # for i in range(len(masks)):
    #     plt.imshow(masks[i:(i+1), :, :].permute(1, 2, 0))
    #     plt.show()
    
    thickens_info = combining_stats(discr_flakes, boxes, masks, height, width, tolerance_level, scores)
    
    # min_boxes = [(cv2.boxPoints(rect)).astype(np.int32) for rect in R]    
    
    
    # for i in range(len(masks)):
    #     mask_color = masks[i]
    #     # mask_color = cv2.cvtColor(mask_color * 255, cv2.COLOR_GRAY2BGR)
    #     # Draw the rectangle on the mask
    #     # cv2.drawContours(mask_color, [min_boxes[i]], 0, (0, 255, 0), 2)  # Green box
        
    #     # plt.imshow(flakes[i], cmap='gray')
    #     # plt.imshow(discr_flakes[i], cmap='gray')
    #     plt.imshow(mask_color, cmap='gray')
    #     # plt.imshow(img)
    #     # plt.imshow(img, cmap='gray')
    #     plt.show()
    
    return thickens_info, empty_masks






