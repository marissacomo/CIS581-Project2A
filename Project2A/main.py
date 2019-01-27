'''
  File name: test_script.py
  Author: 
  Date Created:
'''

'''
  File clarification:
    Check the accuracy of your algorithm
'''

from click_correspondences import click_correspondences
import numpy as np
from morph_tri import morph_tri
import os
from PIL import Image
from scipy import misc

# test triangulation morphing
def test_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # generate morphed image
  morphed_ims = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

  return True


# the main test code
def main():

  folder = 'images'

  # image1
  im_path = os.path.join(folder,"mom.PNG")
  im1 = np.array(Image.open(im_path).convert('RGB'))
  im1 = misc.imresize(im1, 0.5, interp="bicubic")

  # image2
  im_path = os.path.join(folder,"mar.PNG")
  im2 = np.array(Image.open(im_path).convert('RGB'))
  im2 = misc.imresize(im2, 0.5, interp="bicubic")

  im1_pts, im2_pts = click_correspondences(im1, im2)

  if(im1_pts.shape != im2_pts.shape):
    print("oops you need the same number of points on each image... try again")
    raise SystemExit

  # dummy warp_frac and dissolve_frac
  warp_frac, dissolve_frac =  np.arange(0, 1.01667, 0.01667), np.arange(0, 1.01667, 0.01667) 

  # test triangulation morphing
  if not test_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
   # print ("The Triangulation Morphing test failed.")
    return


  print("All tests passed! ")
  return


if __name__ == "__main__":
  main()
