'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: source image
    - Input im2: target image
    - Output im1_pts: correspondences coordiantes in the source image
    - Output im2_pts: correspondences coordiantes in the target image
'''
from cpselect import cpselect

def click_correspondences(im1, im2):
  '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''
  point_img1, point_img2 = cpselect(im1,im2)

  # TODO: Your code here
  
  return point_img1, point_img2
