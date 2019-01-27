'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: source image
    - Input im2: target image
    - Input im1_pts: correspondences coordiantes in the source image
    - Input im2_pts: correspondences coordiantes in the target image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as t
import matplotlib.pyplot as plt
import imageio
'''
#   Function Input 
#   v     M*N            the value lies on grid point which is corresponding to the meshgrid coordinates 
#   xq    M1*N1 or M2    the query points x coordinates
#   yq    M1*N1 or M2    the query points y coordinates
#         
##########
#   Function Output
#   interpv , the interpolated value at querying coordinates xq, yq, it has the same size as xq and yq.
##########
#   For project 1, v = Mag
#                  xq and yq are the coordinates of the interpolated location, i.e the coordinates computed based on the gradient orientation.
'''
def interp2(v, xq, yq):
  dim_input = 1
  if len(xq.shape) == 2 or len(yq.shape) == 2:
    dim_input = 2
    q_h = xq.shape[0]
    q_w = xq.shape[1]
    xq = xq.flatten()
    yq = yq.flatten()

  h = v.shape[0]
  w = v.shape[1]
  if xq.shape != yq.shape:
    raise 'query coordinates Xq Yq should have same shape'


  x_floor = np.floor(xq).astype(np.int32)
  y_floor = np.floor(yq).astype(np.int32)
  x_ceil = np.ceil(xq).astype(np.int32)
  y_ceil = np.ceil(yq).astype(np.int32)

  x_floor[x_floor<0] = 0
  y_floor[y_floor<0] = 0
  x_ceil[x_ceil<0] = 0
  y_ceil[y_ceil<0] = 0

  x_floor[x_floor>=w-1] = w-1
  y_floor[y_floor>=h-1] = h-1
  x_ceil[x_ceil>=w-1] = w-1
  y_ceil[y_ceil>=h-1] = h-1
       
  v1 = v[y_floor, x_floor]
  v2 = v[y_floor, x_ceil]
  v3 = v[y_ceil, x_floor]
  v4 = v[y_ceil, x_ceil]
	
  lh = yq - y_floor
  lw = xq - x_floor
  hh = 1 - lh
  hw = 1 - lw

  w1 = hh * hw
  w2 = hh * lw
  w3 = lh * hw
  w4 = lh * lw

  interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

  if dim_input == 2:
    return interp_val.reshape(q_h,q_w)
  return interp_val

def solveBary(Ax, Bx, Cx, Ay, By, Cy, X_f, Y_f):
  const = 1.0 /(Ax * By - Ax * Cy - Bx * Ay + Bx * Cy + Cx * Ay - Cx * By)
  alpha = const * ((By - Cy) * X_f + (Cx - Bx) * Y_f + Bx * Cy - Cx * By)
  beta = const * ((Cy - Ay) * X_f + (Ax - Cx) * Y_f + Cx * Ay - Ax * Cy)
  gamma = const * ((Ay - By) * X_f + (Bx - Ax) * Y_f + Ax * By - Bx * Ay)
  return alpha, beta, gamma

def getCorners(points, indices1, m_grid):
  Ax = points[indices1[m_grid][:,0]][:,0]
  Ay = points[indices1[m_grid][:,0]][:,1]
  Bx = points[indices1[m_grid][:,1]][:,0]
  By = points[indices1[m_grid][:,1]][:,1]
  Cx = points[indices1[m_grid][:,2]][:,0]
  Cy = points[indices1[m_grid][:,2]][:,1]
  return Ax, Bx, Cx, Ay, By, Cy

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):

  # 1. find the average
  average_pts = (im1_pts + im2_pts) / 2

  # 2. Delauny Triangulation
  tri = t.Triangulation(average_pts[:,0], average_pts[:,1]) # Delaunay(average_pts) # list of triangles with indices corresponding to the average_pts
  indices = tri.triangles #tri.simplices # np.array(tri.simplices)  # return indices of triangles

  # make a mesh grid
  nr = im1.shape[0];
  nc = im1.shape[1];
  X, Y = np.meshgrid(np.arange(nc), np.arange(nr))

  # store all images
  results = np.empty((warp_frac.shape[0], im1.shape[0], im1.shape[1], im1.shape[2])) # make an empty array of this size

  for i in range(len(warp_frac)):
    lerp = im1_pts * (1-warp_frac[i]) + im2_pts * warp_frac[i]

    # correlates the current lerp to the average(.5) triangulation
    lerp_tri = t.Triangulation(lerp[:,0], lerp[:,1], indices)

    find_triangle = lerp_tri.get_trifinder() # find_triangle(x,y)

    # mesh grid - stores the triangle index of each pixel
    m_grid = find_triangle(X,Y) 
    m_grid = m_grid.flatten() # make it 1D
    X_f = X.flatten()
    Y_f = Y.flatten()

# - LERP -
    # get ax,ay,bx,by,cx,cy using the mesh grid to triangle index correlation
    indices1 = np.array(tri.triangles) #np.array(tri.simplices)
    Ax, Bx, Cx, Ay, By, Cy = getCorners(lerp, indices1, m_grid)
  
    # solve the barycentric coordinate for each pixel
    alpha, beta, gamma = solveBary(Ax, Bx, Cx, Ay, By, Cy, X_f, Y_f)
 
    # image 1 - get ax,ay,bx,by,cx,cy using the mesh grid to triangle index correlation 
    Ax1, Bx1, Cx1, Ay1, By1, Cy1 = getCorners(im1_pts, indices1, m_grid)

    # image 2 - get ax,ay,bx,by,cx,cy using the mesh grid to triangle index correlation 
    Ax2, Bx2, Cx2, Ay2, By2, Cy2 = getCorners(im2_pts, indices1, m_grid)
    
    # x,y coordinates for image 1
    x_im1 = Ax1 * alpha + Bx1 * beta + Cx1 * gamma
    y_im1 = Ay1 * alpha + By1 * beta + Cy1 * gamma

    # x,y coordinates for image 2
    x_im2 = Ax2 * alpha + Bx2 * beta + Cx2 * gamma
    y_im2 = Ay2 * alpha + By2 * beta + Cy2 * gamma

    # use interp2 to get the exact x,y coordinates in the source and target images
    r1 = im1[:,:,0]
    g1 = im1[:,:,1]
    b1 = im1[:,:,2]

    r2 = im2[:,:,0]
    g2 = im2[:,:,1]
    b2 = im2[:,:,2]

    im1_R = interp2(r1, x_im1, y_im1)
    im1_G = interp2(g1, x_im1, y_im1)
    im1_B = interp2(b1, x_im1, y_im1)

    im2_R = interp2(r2, x_im2, y_im2)
    im2_G = interp2(g2, x_im2, y_im2)
    im2_B = interp2(b2, x_im2, y_im2)

    # # cross-dissolve, lerp image1 rbg, image2 rgb
    R = im1_R * (1-dissolve_frac[i]) + im2_R * dissolve_frac[i]
    G = im1_G * (1-dissolve_frac[i]) + im2_G * dissolve_frac[i]
    B = im1_B * (1-dissolve_frac[i]) + im2_B * dissolve_frac[i]
    
    R = R.reshape(nc, nr)
    G = G.reshape(nc, nr)
    B = B.reshape(nc, nr)

    # store resulting image 
    results[i, :, :, 0] = R
    results[i, :, :, 1] = G
    results[i, :, :, 2] = B

  res_list = []
  k = 0
  while k < results.shape[0]:
    res_list.append(results[k, :, :, :])
    k += 1

  # generate gif file
  imageio.mimsave('./eval_testimg.gif', res_list)
	
  return results
