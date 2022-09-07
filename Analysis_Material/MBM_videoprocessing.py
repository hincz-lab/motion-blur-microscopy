# -*- coding: utf-8 -*-
"""
Created by Ayesha Gonzales
Case Western Reserve University
Segmentation and post-segmentation processing for video data taken using Motion Blur Microscopy
Version as of Fri Aug 19 2022
"""
import tensorflow as tf
import sys
import cv2
import numpy as np
from scipy.ndimage import label
from skimage import measure
import matplotlib.pyplot as plt
import os, csv
import bisect
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import tkinter as tk
import glob

structure = [[1,1,1],
             [1,1,1],
             [1,1,1]] #2-connectivity

def standard_norm(img):
    height, width, channels = img.shape
    for channel in range(channels):
        img[:,:,channel] = (img[:,:,channel] - np.mean(img[:,:,channel]))/np.std(img[:,:,channel])
    return img

def dist(centroids): #centroids = list of points
  poss_distances = []
  for c in range(len(centroids)-1):
    cent = centroids[c]
    centroids2 = centroids[c+1:] #every point after cent[x for x in centroids if x != cent]
    for cent2 in centroids2:
      poss_distances.append(np.sqrt( (cent[0] - cent2[0])**2 + (cent[1] - cent2[1])**2 ))
  return poss_distances #a list of distances

#iterate through all coords
def shortest_dist(listofarrs): #two arrays, allowed to be any length
  distances = []
  array1 = listofarrs[0].tolist() #a list of points
  array2 = listofarrs[1].tolist()
  for point1 in array1:
    for point2 in array2:
      distances.append(dist([point1,point2])[0])
  return min(distances)

def return_indices(d,N): #d=index of distance list N=total num. of blobs
  #check if d is in a range
  start = 0
  end = (N-1)
  for n in range(2,N+1):
    if d in range(start,end):
      return n-2, n+(d-start)-1 
      break
    else:
      start = end
      end += (N-n)

def eccentricity(hull): #hull=ConvexHull object
  x, y = np.meshgrid(np.arange(min(hull.points[hull.vertices,0]),max(hull.points[hull.vertices,0])+1), np.arange(min(hull.points[hull.vertices,1]),max(hull.points[hull.vertices,1])+1)) # make a canvas with coordinates
  x, y = x.flatten(), y.flatten()
  temp_points = np.vstack((x,y)).T 

  points = []
  # include both cw and ccw to include as many points as possible
  p = Path(hull.points[::-1]) # cw
  grid = p.contains_points(temp_points, radius=0.1)
  grid2 = measure.points_in_poly(temp_points,hull.points[hull.vertices][::-1])
  for g in range(len(grid)):
    if grid[g]==True or grid2[g]==True:
      points.append(np.array(temp_points[g]))
  p = Path(hull.points) # ccw
  grid = p.contains_points(temp_points, radius=0.01)
  grid2 = measure.points_in_poly(temp_points,hull.points[hull.vertices])
  for g in range(len(grid)):
    if grid[g]==True or grid2[g]==True:
      points.append(np.array(temp_points[g]))
  points = np.array(points)

  #take above points and make a mini-image w all points labeled 1, and all other points labeled 0
  y_len  = int(max(hull.points[hull.vertices,0]) - min(hull.points[hull.vertices,0]) +1)
  x_len = int(max(hull.points[hull.vertices,1]) - min(hull.points[hull.vertices,1]) +1)
  temp_image = np.zeros((y_len,x_len))
  for point in points:
    temp_image[int(point[0]-min(hull.points[hull.vertices,0]))][int(point[1]-min(hull.points[hull.vertices,1]))] = 1
  # use measure.regionprops to get eccentricity (double check num of blobs=1)
  outarray,numOfBlobs = label(temp_image,structure)
  properties = measure.regionprops(outarray)
  if numOfBlobs==0:
    print("ERROR: detecting 0 blobs!") #STOP RUNNING CODE IF THIS COMES UP
    ecc = 0
  else:
    eccs = [prop.eccentricity for prop in properties]
    ecc = eccs[0] #should only be 1!
  return ecc

def main(new_model, video_path, celltype, autoconvex,frames_range=(),custom_thresh=()):
    """"
    new_model: path to model for segmentation
    
    video_path: path to video to be analyzed
    
    celltype: 'srbc', 'cart', or 'custom'. if 'custom', custom_thresh must be defined as a 
        length 3 tuple. The first element should be the minimum size threshold for the cell of interest,
        the second the minimum size for adhesion, and the third the number of frames a cell can go undetected.
        'srbc' and 'cart' will ignore anything written in custom_thresh. If multiple cell types present, it is
        best to use the smallest of the size thresholds, and the longest gap between frames where
        a cell is detected.
        
    autoconvex: 'y' or 'n'. whether or not to automatically take convex hull of all cells found.
        can be helpful if segmentation network has trouble identifying a portion of pixels belonging
        to a cell.
        
    frames_range: OPTIONAL. A 2-length tuple that defines minimum and maximum frames that will be analyzed. 
        Helpful if experiment has any gradual increase or decrease in speed near beginning or end of video, 
        so that only the useful portion of the video may be used.
        
    custom_thresh: OPTIONAL. A 3-length tuple. The first element should be the minimum size threshold for the cell of interest,
        the second the minimum size for adhesion, and the third the number of frames a cell can go undetected.
        Will only be used if celltype is 'custom'.
    """
    if autoconvex.lower() == 'y':
        print('Automatically take convex hull of all connected regions')
    elif autoconvex.lower() == 'n':
        pass
    else:
        print('WARNING: autoconvex value invalid. Default N will be used.')
        autoconvex='n'
    
    if celltype.lower() == 'cart':
        thresh = 40
        init_thresh = 40
        ghost_frames=9
    elif celltype.lower() == 'srbc':
        thresh = 45
        init_thresh = 90
        ghost_frames=2
    elif celltype.lower() == 'custom':
        if len(custom_thresh)==3:
            thresh = custom_thresh[0]
            init_thresh = custom_thresh[1]
            ghost_frames= custom_thresh[2]
        else:
            print("ERROR: custom_thresh must be 3 numbers defining the minimum size threshold, the adhesion size threshold, and the number of frames a cell can go undetected.")
            sys.exit(0)
    else:
        print('ERROR: celltype value invalid. Please try again with "srbc", "cart", or "custom"')
        sys.exit(0)
    
    min_frames=0
    max_frames=1000000
    if frames_range:
        if len(frames_range)!=2:
            print("ERROR: frames_range must be two numbers: the beginning frame and end frame to be analyzed.")
            sys.exit(0)
        else:
            limit_frames=True
            min_frames=frames_range[0]
            max_frames=frames_range[1]
    else:
        print('All frames to be analyzed.')
    
    data_filename = video_path.split("/",-1)[-1][:-4].replace(' ', '_') +'_master_static.csv'  
    
    try:
      os.remove(data_filename)
    except OSError:
        pass
    
    with open(data_filename,'a') as fd:
        fd.write('frame' + ',' + 'row' + ',' + 'column' + ',' + 'area' + ',' + 'eccentricity' + ',' + 'rel grey color' + '\n')
    
    # convert video to a sequence of images
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_num +=1
        # if frame is read correctly ret is True
        if not ret:
            print(frame_num-1, "is the max number of frames!")
            max_frames = frame_num-1
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if limit_frames==True:
          if frame_num > max_frames:
              break
          if frame_num < min_frames:
              pass
        #make prediction on "frame", collecting size and location of blobs
        else:
            test_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(np.shape(test_Image)) == 2:
                test_Image = cv2.cvtColor(test_Image, cv2.COLOR_GRAY2RGB)
            image_Height, image_Width, channels = np.shape(test_Image)
            if (image_Height % 150) < 75 and (image_Width % 150) < 75:
                test_Image_Resized = cv2.resize(test_Image,(int(np.floor(image_Width/150)*150), int(np.floor(image_Height/150)*150)), interpolation = cv2.INTER_CUBIC)
                vertical_Tiles = int(np.floor(image_Height/150))
                horizontal_Tiles = int(np.floor(image_Width/150))
            elif (image_Height % 150) >= 75 and (image_Width % 150) >= 75:
                test_Image_Resized = cv2.resize(test_Image,(int((np.floor(image_Width/150) + 1)*150), int((np.floor(image_Height/150) + 1)*150)), interpolation = cv2.INTER_CUBIC)
                vertical_Tiles = int((np.floor(image_Height/150) + 1))
                horizontal_Tiles = int((np.floor(image_Width/150) + 1))
            elif (image_Height % 150) >= 75 and (image_Width % 150) < 75:
                test_Image_Resized = cv2.resize(test_Image,(int(np.floor(image_Width/150)*150), int((np.floor(image_Height/150) + 1)*150)), interpolation = cv2.INTER_CUBIC)
                vertical_Tiles = int((np.floor(image_Height/150) + 1))
                horizontal_Tiles = int(np.floor(image_Width/150))
            else:
                test_Image_Resized = cv2.resize(test_Image,(int((np.floor(image_Width/150) + 1)*150), int(np.floor(image_Height/150)*150)), interpolation = cv2.INTER_CUBIC)
                vertical_Tiles = int(np.floor(image_Height/150))
                horizontal_Tiles = int((np.floor(image_Width/150) + 1))
            image_To_Use = cv2.cvtColor(test_Image_Resized, cv2.COLOR_BGR2GRAY)
            image_Height_Resized, image_Width_Resized, channels = np.shape(test_Image_Resized)
            output_Image = np.zeros((image_Height_Resized,image_Width_Resized))
    
            x_Slider = 0
            y_Slider = 0
            output_Array = np.zeros((128,128))
            for i in range(vertical_Tiles):
                x_Slider = 150*i
                for j in range(horizontal_Tiles):
                    y_Slider = 150*j
                    current_Tile = test_Image_Resized[x_Slider:x_Slider + 150, y_Slider: y_Slider + 150,:]/255
                    current_Tile = cv2.resize(current_Tile, (128,128), interpolation=cv2.INTER_AREA)
                    current_Tile_Normalized = standard_norm(current_Tile.copy())
                    current_Tile_Normalized = current_Tile_Normalized[None,:,:,:]
                    output = new_model.predict(current_Tile_Normalized)
                    for i in range(128):
                        for j in range(128):
                            output_Array[i,j] = np.argmax(output[0,i,j,:])
                    output_Array = cv2.resize(output_Array,(150,150),interpolation = cv2.INTER_AREA)
                    output_Image[x_Slider:x_Slider + 150, y_Slider: y_Slider + 150] = output_Array
                    output_Array = np.zeros((128,128))
            for i in range(image_Height_Resized):
                for j in range(image_Width_Resized):
                    if output_Image[i,j] != 0:
                        output_Image[i,j] = 1
                    else:
                        continue
    
            blobs, _ = label(output_Image, structure=structure) # whole channel image
            properties = measure.regionprops(blobs)
            blob_Sizes = [prop.area for prop in properties if prop.area > thresh]
            centroids = [prop.centroid for prop in properties if prop.area > thresh]
            eccentricities = [prop.eccentricity for prop in properties if prop.area > thresh]
            clusters = [prop.coords for prop in properties if prop.area > thresh]
            something2 = len(centroids)
    
            # Check if any centroids w/in 50 pixels of each other
            indices_to_del = [] #list of indices
            distances = dist(centroids)
            for d in range(len(distances)):
              dis = distances[d]
              if dis<50:
                # get indexes of the two blobs
                idx1,idx2 = return_indices(d,len(centroids))
                indices_to_del.extend([idx1,idx2])
                points = [[prop.coords for prop in properties][idx1],[prop.coords for prop in properties][idx2]]
                if shortest_dist(points)<=5:
                  #compute convex hull for points
                  hull = ConvexHull(list(points[0])+list(points[1]))
                  #append new centroid, area, and eccentricity to end of lists
                  blob_Sizes.append(hull.volume)
                  centroids.append((np.mean(hull.points[hull.vertices,0]),np.mean(hull.points[hull.vertices,1])))
                  eccentricities.append(eccentricity(hull))
                  clusters.append(hull.points)
            if indices_to_del:
              blob_Sizes = [i for j, i in enumerate(blob_Sizes) if j not in indices_to_del]
              centroids = [i for j, i in enumerate(centroids) if j not in indices_to_del]
              eccentricities = [i for j, i in enumerate(eccentricities) if j not in indices_to_del]
              clusters = [i for j, i in enumerate(clusters) if j not in indices_to_del]
              
            if autoconvex.lower()=='y':
              # take convex hull of all other cells
              for cell in range(len(centroids)):
                  points = clusters[cell]
                  hull = ConvexHull(list(points))
                  blob_Sizes[cell] = hull.volume
                  centroids[cell] = (np.mean(hull.points[hull.vertices,0]),np.mean(hull.points[hull.vertices,1]))
                  eccentricities[cell] = eccentricity(hull)
                  clusters[cell] = hull.points
    
            # relative average grayscale value of each cell
            rel_greyscale_vals = []
            for cluster in clusters:
                pixelvals = []
                for cpoint in cluster:
                    pixelvals.append(image_To_Use[int(cpoint[0])][int(cpoint[1])])
                rel_greyscale_vals.append(np.mean(pixelvals) - np.mean(image_To_Use))
    
            # csv w/ each row: frame, , location[0], location[1], area, eccentricity
            # more raw of data lol
            # print(len(centroids))
            for i in range(len(centroids)):
                with open(data_filename,'a') as fd:
                    fd.write(str(frame_num) + ',' + str(centroids[i][0]) + ',' + str(centroids[i][1]) + ',' + str(blob_Sizes[i]) + ',' + str(eccentricities[i]) + ',' + str(rel_greyscale_vals[i]) + "\n")
    
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    
    """"SIZE VARIANCE ANALYSIS: initial attachment follows adhesion size threshold, in following frames looking for same centroid at least the minimum size for cell of interest"""
    # if a cell moves, it is *not adhered*. want location to stay the same -- record new location instances in each frame
    # csv: frame, row, column, area, eccentricity
    
    adhesion_data = []
    other_data = [] #elements:[frame_num, location, eccentricity, area, rolling?, color]
    cell_num = 0
    count = 0
    with open(data_filename, 'r') as csvfile:
      for row in list(csv.reader(csvfile))[1:]:
        frame_num = int(row[0])
        if frame_num>=min_frames and frame_num<=max_frames:
          size = float(row[3])
          if size >= thresh:
              for d in range(1,ghost_frames+2): #check if cell has disappeared for some time
                  temp_data = [dat for dat in other_data if dat[0]==frame_num-d]# checking previous frame
                  # print(row)
                  location = (float(row[1]),float(row[2])) # allow cell to be +/- 5 to be the same
                  ecc = float(row[4])
                  color = float(row[5])
                  abscolor = float(row[6])
                  darkestcolor = float(row[7])
                  # check if cell rolls left, can only move +/- 5 up or down
                  roll_poss = [loc[1] for loc in temp_data if loc[1][1]-30<=location[1] and loc[1][1]-5>location[1]]
                  roll_x_poss = [r_p for r_p in roll_poss if r_p[0]>=location[0]-5 and r_p[0]<=location[0]+5]
                  roll_left = []
                  for r_xp in roll_x_poss:
                    idxs = [i for i, x in enumerate(temp_data) if x[1] == r_xp]
                    for id in idxs:
                      if size>=1:#(temp_data[id][3] - (temp_data[id][3]*0.5)) and size<=(temp_data[id][3] + (temp_data[id][3]*0.5)): #check size
                        roll_left.append(1)
                  if roll_left:
                    roll = 'yes'
                  else:
                    roll = 'no'
                  # check if location inside a rectangle & check size
                  lower_bound = (location[0]-5, location[1]-5)
                  upper_bound = (location[0]+5, location[1]+5)
                  lower_bound_i = bisect.bisect_left([loc[1] for loc in temp_data], lower_bound)
                  upper_bound_i = bisect.bisect_right([loc[1] for loc in temp_data], upper_bound, lo=lower_bound_i)
                  nums = [loc[1] for loc in temp_data][lower_bound_i:upper_bound_i]
                  sizes = [loc[3] for loc in temp_data][lower_bound_i:upper_bound_i]
                  final_nums = []
                  for nn in range(len(nums)):
                    num = nums[nn]
                    low_size = sizes[nn] - (sizes[nn]*0.5) 
                    high_size = sizes[nn] + (sizes[nn]*0.5) #size w/in 50% of initial size
                    if num[0]>=lower_bound[0] and num[0]<=upper_bound[0] and num[1]>=lower_bound[1] and num[1]<=upper_bound[1]:
                      if size>=1:#low_size and size<=high_size:
                        final_nums.append(num)
                  if final_nums: #enter this if cell was in previous frame -- check if size w/in 50%
                    psn = [loc[1] for loc in other_data].index(final_nums[-1])
                    adhesion_data.append(adhesion_data[psn])
                    other_data.append([frame_num, location, ecc, size, roll, color, abscolor, darkestcolor])
                    break
                  elif not final_nums and roll=='yes':
                    psn = [loc[1] for loc in other_data].index(roll_x_poss[-1])
                    adhesion_data.append(adhesion_data[psn])
                    other_data.append([frame_num, location, ecc, size, roll, color, abscolor, darkestcolor])
                    break
                  elif not final_nums and roll=='no' and d==10: #check frame before
                    #if size>=init_thresh:
                    adhesion_data.append(cell_num)
                    other_data.append([frame_num, location, ecc, size, roll, color, abscolor, darkestcolor])
                    cell_num += 1
                  else:
                    pass
    
    fps = 1/1.2
    ecc_data, ecc_std, final_data = [], [], [] # list of corresponding avg ecc, stdev ecc, and adhesion times
    ecc_data_end, ecc_std_end, end_of_data = [], [], []
    toss = 1
    for cell in np.unique(adhesion_data):
      if adhesion_data.count(cell)<=toss:
        pass
      else:
        indices = [i for i, x in enumerate(adhesion_data) if x == cell] #should return all indices of a specific cell label
        eccs = []
        for j in indices:
          eccs.append(other_data[j][2])
        if other_data[j][0]==max_frames:
          ecc_data_end.append(np.mean(eccs))
          ecc_std_end.append(np.std(eccs))
          end_of_data.append(adhesion_data.count(cell))
        else:
          ecc_data.append(np.mean(eccs))
          ecc_std.append(np.std(eccs))
          final_data.append(adhesion_data.count(cell))
    
    """Write to .csv cell#, frame attached, frame detached, avg ecc, stddev ecc, avg size, avg location, rolling?"""
    finalized_data_filename = data_filename[:-10] + 'dynamic.csv'
    
    try:
      os.remove(finalized_data_filename)
    except OSError:
        pass
    
    with open(finalized_data_filename,'a') as fd:
      fd.write('cell number' + ',' + 'frame attached' + ',' + 'frame detached' + ',' + 'average eccentricity' + ',' +
               'average size' + ',' + 'average location[0]' + ',' + 'average location[1]' + ',' + 'rolling?' + ',' +
               'start loc[0]' + ',' + 'start loc[1]' + ',' + 'end loc [0]' + ',' + 'end loc[1]' + ',' + 'avg rel grey color' + '\n')
      
    for cell in np.unique(adhesion_data):
      indices = [i for i, x in enumerate(adhesion_data) if x == cell]
      rolling = 'no'
      frame_att = other_data[indices[0]][0]
      frame_det = other_data[indices[-1]][0]
      ecc_data = []
      size_data = []
      loc_data = []
      color_data = []
      for idx in indices:
        if other_data[idx][4]=='yes': #if rolling at any point, label as rolling
          rolling = 'yes'
        ecc_data.append(other_data[idx][2])
        size_data.append(other_data[idx][3])
        loc_data.append(other_data[idx][1])
        color_data.append(other_data[idx][5])
      #print("Stdev of cell size", np.std(size_data))
      avg_loc = [sum(x)/len(x) for x in zip(*loc_data)]
      with open(finalized_data_filename,'a') as fd:
        fd.write(str(cell) + ',' + str(frame_att) + ',' + str(frame_det) + ',' + str(np.mean(ecc_data)) + ',' +
                 str(np.mean(size_data)) + ',' + str(avg_loc[0]) + ',' + str(avg_loc[1]) + ',' + rolling + ',' + 
                 str(loc_data[0][0]) + ',' + str(loc_data[0][1]) + ',' + str(loc_data[-1][0]) + ',' + str(loc_data[-1][1]) + 
                 ',' + str(np.mean(color_data)) + '\n')
        
    # write over raw data file to include cell # as first element of each row
    try:
      os.remove(data_filename)
    except OSError:
        pass
    
    with open(data_filename,'a') as fd:
        fd.write('cell num' + ',' + 'frame' + ',' + 'locatin[0]' + ',' + 'location[1]' + ',' + 'area' + ',' + 'eccentricity' + ',' + 'rel grey color' + '\n')
        for jj in range(len(adhesion_data)):
            fd.write(str(adhesion_data[jj]) + ',' + str(other_data[jj][0]) + ',' + str(other_data[jj][1][0]) + ',' + str(other_data[jj][1][1]) +
                     ',' + str(other_data[jj][3]) + ',' + str(other_data[jj][2]) + ',' + str(other_data[jj][5]) + '\n')
            
def assign_variable_dynamically(expression):
    var_name, value = expression.split("=")
    globals()[var_name] = value

if __name__=="__main__":
    """ COMMAND LINE USAGE:
    python MBM_videprocessing.py model='model.h5' video_path='video_path.avi' celltype='cart' autoconvex='n' frames_range=1,10000 custom_thresh=40,40,10
     ******** note not to use parentheses around frames_range and custom_thresh ********
    """
    # get the arguments from the command line
    args = sys.argv[1:]
    for arg in args:
        assign_variable_dynamically(arg)
    if 'frames_range' in locals():
        frames_range = tuple(int(fff) for fff in frames_range.split(','))
    if 'custom_thresh' in locals():
        custom_thresh = tuple(int(ccc) for ccc in custom_thresh.split(','))
    main(model, video_path, celltype, autoconvex,frames_range=(),custom_thresh=())
    
    