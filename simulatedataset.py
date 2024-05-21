import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import os
import dask.array as da
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
import cv2
import scipy.io as sio
from scipy.ndimage import uniform_filter
from scipy import ndimage
from tqdm import tqdm

"""
This script processes depth and color images to generate intermediate frames between time frames using optical flow. 
"""

class ImageLoader:  
    def __init__(self, path):
        """
        Initialize the ImageProcessor with the given path.
        
        Parameters:
        path (str): The directory path where images are stored.
        """
        self.path = path 
    
    def loadimages(self,extension, idx1, idx2):
        """
        Parameters:
        extension (str): The file extension of the images to load.
        idx1 (int): The starting index of the images to load.
        idx2 (int): The ending index of the images to load.
        
        Returns:
        list: A list of images.
        """

        files = sorted(f for f in os.listdir(self.path) if f.endswith(extension))[idx1:idx2]
        if extension == '.npy':
            return [np.load(os.path.join(self.path, f)) for f in files]
        elif extension == '.png':
            return [cv2.imread(os.path.join(self.path, f)) for f in files]
    
    def generate_inter_frames(self, image1, image2, numframes):
        """
        Generate intermediate frames between two images using optical flow.
        
        Parameters:
        image1 (ndarray): The first image.
        image2 (ndarray): The second image.
        numframes (int): The number of intermediate frames to generate.
        
        Returns:
        list: A list of intermediate frames.
        """
        # Calculate optical flow between the two images 
        v, u = optical_flow_tvl1(image2, image1)
        nr, nc = image1.shape
        # Create meshgrid for coordinates
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),indexing='ij')
        frames = []
        for i in range(1, numframes + 1):
            fraction = i/(numframes)
            # Warp the image based on the calculated flow
            warped = warp(image1, np.array((row_coords+ fraction*v, col_coords + fraction*u)) , mode='edge')
            frames.append(warped)       
        return frames

    def process_depth(self, idx1, idx2, numframes):
        """
        Process depth images to generate intermediate frames and combine them into a single matrix.
        
        Parameters:
        idx1 (int): The starting index of the images to process.
        idx2 (int): The ending index of the images to process.
        numframes (int): The number of intermediate frames to generate.
        
        Returns:
        ndarray
        """
        # Load depth images
        images = self.loadimages('.npy',idx1, idx2)
        
        # Generate intermediate frames for each pair of images
        all_inter_frames = [self.generate_inter_frames(images[i], images[i+1],numframes) for i in tqdm(range(len(images) - 1), desc="Processing depth images")]
         
        num_images = len(all_inter_frames)*len(all_inter_frames[0])
        nr, nc = images[0].shape
        matrix_depth = np.empty((num_images, nr, nc))
        index = 0 
        for i in range(len(all_inter_frames)):
            for frames in range(len(all_inter_frames[0])):
                    matrix_depth[index, :,:] = all_inter_frames[i][frames]
                    index = index + 1
        return matrix_depth

    def process_color(self, idx1, idx2, numframes):
        """
        Process color images to generate intermediate frames for each color channel and combine them into a single matrix.
        
        Parameters:
        idx1 (int): The starting index of the images to process.
        idx2 (int): The ending index of the images to process.
        numframes (int): The number of intermediate frames to generate.
        
        Returns:
        ndarray
        """
        # Load color images
        images = self.loadimages('.png',idx1, idx2)
        # Generate intermediate frames for each pair of images and each color channel
        all_inter_frames = [
            [
                self.generate_inter_frames(images[i][:, :, channel] / 255,
                                           images[i + 1][:, :, channel] / 255, numframes)
                for channel in range(3)
            ]
            for i in tqdm(range(len(images) - 1), desc='Processing color images')
        ]

        # # Generate intermediate frames between each pair of images 
        num_frames = len(all_inter_frames[0][0])
        num_images =  len(all_inter_frames)
        nr, nc, _ = images[0].shape
        matrix_color = np.empty(((num_images)*num_frames, nr, nc, 3))
        index = 0 
        for i in range(num_images):
            for frames in range(num_frames):
                    matrix_color[index, :,:,0] = all_inter_frames[i][0][frames]
                    index = index + 1

        index = 0 
        for i in range(num_images):
            for frames in range(num_frames):
                    matrix_color[index, :,:,1] = all_inter_frames[i][1][frames]
                    index = index + 1

        index = 0 
        for i in range(num_images):
            for frames in range(num_frames):
                    matrix_color[index, :,:,2] = all_inter_frames[i][2][frames]
                    index = index + 1

        return matrix_color

    
def simulate_binary(SigIRF, coeffProba,matrix_depth,matrix_color):
    """
    Simulates binary frames from depth and color images.

    Parameters:
    SigIRF (float): Standard deviation for IRF simulation.
    coeffProba (float): Coefficient for probability calculation.
    matrix_depth (ndarray): Depth images.
    matrix_color (ndarray): Color images.

    Returns:
    new_RGB (ndarray): Simulated RGB images.
    LR_depth (ndarray): Simulated low-resolution depth images.
    """
    num_frames = matrix_depth.shape[0]
    num_rows, num_cols = matrix_depth.shape[1:3]
    # Initialize binary depth matrix and reshaped color matrix
    matrix_depth_binary = np.empty((num_rows,num_cols, num_frames))
    matrix_color_newshape = np.empty((num_rows,num_cols,num_frames ))

    for i in tqdm(range(num_frames), desc='Simulating binary frames'):

        D = matrix_depth[i,:,:]*100
        I = matrix_color[i,:,:,0]

        # Calculate probability and threshold to simulate binary frame
        MeanD = D + np.sqrt(SigIRF) * np.random.randn(num_rows, num_cols)
        Proba = coeffProba * I / np.max(matrix_color[i,:,:,0].flatten())
        Tind =  (Proba>0) & (np.round(np.random.exponential(Proba))==0)

        test = MeanD*Tind
        matrix_depth_binary[:,:,i] = test
        matrix_color_newshape[:,:,i] = I

    # Resize the binary depth and color frames
    new_data_image = ndimage.zoom(matrix_depth_binary[:16*20:,:16*20,:], [2,2,1], order=0)/255
    new_RGB= ndimage.zoom(matrix_color_newshape[:16*20:,:16*20,:], [2,2,1], order=0) *255

    # Downscale the binary depth frames to low-resolution depth frames
    LR_depth = ndimage.zoom(new_data_image, [1/16,1/16,1], order=0)  
    return new_RGB, LR_depth
    

if __name__ == '__main__':
    
    # Define paths for depth and color images
    path_depth = '/home/ar432/DVSR/data/demo_dydtof/depth_all/'
    path_color = '/home/ar432/DVSR/data/demo_dydtof/color_all/'
    path_save = '/home/ar432/DVSR/PnP_method/Dataset'
    idx1 = 17
    idx2 = 20
    numframes =  2
    if not os.path.exists(os.path.join(path_save, f'Initial_depth_images_{idx1}_{idx2}_{numframes}.npy')):
        print('Data doesnt exist already')
        
        Depthloader = ImageLoader(path_depth)
        Colorloader = ImageLoader(path_color)

        frames_depth = Depthloader.process_depth(idx1, idx2, numframes)
        frames_color = Colorloader.process_color(idx1, idx2, numframes)

        np.save(os.path.join(path_save, f'Initial_depth_images_{idx1}_{idx2}_{numframes}.npy'), frames_depth)
        np.save(os.path.join(path_save, f'Initial_color_images_{idx1}_{idx2}_{numframes}.npy'), frames_color)

    frames_depth = np.load(os.path.join(path_save, f'Initial_depth_images_{idx1}_{idx2}_{numframes}.npy'))
    print(frames_depth)
    frames_color = np.load(os.path.join(path_save, f'Initial_color_images_{idx1}_{idx2}_{numframes}.npy'))

    SigIRF = 0.7
    coeffProba = 3
    new_RGB, LR_depth = simulate_binary(SigIRF, coeffProba,frames_depth,frames_color)

    np.save(os.path.join(path_save,f'Processed_color_{idx1}-{idx2}-{numframes}_p{coeffProba}.npy'),new_RGB)
    np.save(os.path.join(path_save,f'Processed_depth_{idx1}-{idx2}-{numframes}_p{coeffProba}.npy'),LR_depth)






