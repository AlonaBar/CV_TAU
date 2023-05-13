
# from matplotlib import pyplot as plt

# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

import random as rand

def crop_image(image,margin = 0):
    rows, cols, ___ = np.shape(image)
    neg_image = np.subtract(255,image)
    sum_rows = np.sum(neg_image ,axis = (0,2))
    sum_cols = np.sum(neg_image ,axis = (1,2))
    j1 = 0
    flag = False
    while (flag == False):
        if (sum_rows[j1] > 0 or j1 == cols-1):
            flag = True
        j1 += 1

    j2 = cols-1
    flag = False
    while (flag == False):
        if (sum_rows[j2] > 0 or j2 == 0):
            flag = True
        j2 -= 1

    i1 = 0
    flag = False
    while (flag == False):
        if (sum_cols[i1] > 0 or i1 == rows-1):
            flag = True
        i1 += 1

    i2 = rows-1
    flag = False
    while (flag == False):
        if (sum_cols[i2] > 0 or i2 == 0):
            flag = True
        i2 -= 1

    return image[i1-margin:i2+margin,j1-margin:j2+margin,:]



def show_matching_pints(img_src,img_dst,match_p_src,match_p_dst):
    plt.figure()
    plt.subplot(1,2,1)
    panplot = plt.imshow(img_src)
    plt.scatter(match_p_src[0,:],match_p_src[1,:],color ='red')
    plt.title('source image')
    
    plt.subplot(1,2,2)
    panplot = plt.imshow(img_dst)
    plt.scatter(match_p_dst[0,:],match_p_dst[1,:],color ='red')
    plt.title('destination image')
    plt.show()
    
    return


def compute_homography_naive(mp_src, mp_dst):
    N = scipy.shape(mp_src)[1]

    # Build the A matrix:
    A = scipy.empty((1,9))
    A = scipy.delete(A,0,axis=0)
    for i in range(0,N):
        XiTrans = [mp_src[0,i],mp_src[1,i],1]
        uTag = mp_dst[0,i]
        vTag = mp_dst[1,i]
        row1i = scipy.append( scipy.append( XiTrans, [0,0,0]) , scipy.multiply(-uTag,XiTrans)) 
        row2i = scipy.append( scipy.append( [0,0,0], XiTrans) , scipy.multiply(-vTag,XiTrans))
        row1i = scipy.reshape(row1i, (1,9))
        row2i = scipy.reshape(row2i, (1,9))
        A = scipy.concatenate( (A, row1i, row2i))

    # Solve Ax=0  =>   A'Ax=0
    M = np.dot(scipy.transpose(A), A)
    eigVals, eigVecs = np.linalg.eig(M)
    e1 = eigVecs[:, np.argmin(eigVals)]
    #M = scipy.dot(scipy.transpose(A), A)
    #eigVals, eigVecs = scipy.linalg.eig(M)
    #e1 = eigVecs[:, scipy.argmin(eigVals)]
    # Normalize
    e1 = scipy.divide(e1, scipy.sqrt( scipy.sum( scipy.multiply(e1,e1)) ))

    # Build the matrix H from x:
    H = scipy.reshape(e1,(3,3))

    return H


def mapping(H,points_vec):
    m = scipy.shape(points_vec)[1]

    points_vec_homography = scipy.dot(H,scipy.concatenate((points_vec, scipy.ones((1,m)) )) )
    
    multFactor = points_vec_homography[2,:]
    points_vec_homography = scipy.divide(points_vec_homography,multFactor)  #+0.001
    points_vec_homography = scipy.delete(points_vec_homography,2,axis=0)

    return points_vec_homography



def check_forward_mapping(H,img_src,img_dst,match_p_src,match_p_dst):

    match_p_dst_homography = mapping(H,match_p_src)
    
    plt.figure()
    plt.subplot(1,2,1)
    panplot = plt.imshow(img_src)
    plt.scatter(match_p_src[0,:],match_p_src[1,:],color ='red')
    plt.title('source image')
    
    plt.subplot(1,2,2)
    panplot = plt.imshow(img_dst)
    plt.scatter(match_p_dst[0,:],match_p_dst[1,:],color ='red')
    plt.scatter(match_p_dst_homography[0,:],match_p_dst_homography[1,:],color ='blue')
    plt.title('destination image')
    plt.show()
    
    return


def L2_dist(vec1,vec2):
    return scipy.sqrt( scipy.sum( scipy.multiply(vec1-vec2,vec1-vec2) ,axis = 0 ) )

def test_homography(H, mp_src, mp_dst, max_err):
    match_p_dst_homography = mapping(H,mp_src)
    m = scipy.shape(mp_src)[1]
    dist = L2_dist(match_p_dst_homography,mp_dst)
    fit_percent = scipy.sum( dist <= max_err )/m
    dist_mse = scipy.sum( dist[dist <= max_err] )/m
    
    return fit_percent, dist_mse


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    m = scipy.shape(mp_src)[1]
    n = 4
    p = 0.99
    d = 15
    k = scipy.ceil(( scipy.log(1-p)/scipy.log(1-inliers_percent**n) ) )

    pointsVec = range(0,m)
    bestH = []
    bestError = 10000

    for i in range(int(k)):
        # Pick randomly n points
        indexVec = rand.sample(pointsVec,n)
        mp_src_sampled = mp_src[:,indexVec]
        mp_dst_sampled = mp_dst[:,indexVec]
        H = compute_homography_naive(mp_src_sampled, mp_dst_sampled)
        fit_percent, dist_mse = test_homography(H, mp_src, mp_dst, max_err)
        numOfInliers = fit_percent*m

        if numOfInliers>d:
            # Calculate the inliers only
            match_p_dst_homography = mapping(H,mp_src)
            dist = L2_dist(match_p_dst_homography,mp_dst)
            pointsVecAr = scipy.array(pointsVec)            # Move to scipy array format
            indexVec = pointsVecAr[scipy.array(dist <= max_err)]
            #print(indexVec)
            # Fit the model again, using the inliers
            mp_src_sampled = mp_src[:,indexVec]
            mp_dst_sampled = mp_dst[:,indexVec]
            H = compute_homography_naive(mp_src_sampled, mp_dst_sampled)
            #fit_percent, dist_mse = test_homography(H, mp_src, mp_dst, max_err)
            fit_percent, dist_mse = test_homography(H, mp_src_sampled, mp_dst_sampled, max_err)

            if dist_mse<bestError:
                bestError = dist_mse
                bestH = H

    if (bestH == []):
        print('no best H')

    return bestH


def full_forward_mapping (H,src_image, dst_image):
    dst_y_size, dst_x_size , ___ = np.shape(dst_image)
    src_y_size, src_x_size , ___ = np.shape(src_image)

    corners = [[0,0],[src_x_size,0],[0,src_y_size],[src_x_size,src_y_size]]
    dst_corners = np.transpose( [[0,0],[dst_x_size,0],[0,dst_y_size],[dst_x_size,dst_y_size]] )
    borders = mapping(H, np.transpose(corners))
    borders = np.append(borders, dst_corners, axis=1)
    src_extended_width = int( np.max(borders[0,:])-np.min(borders[0,:]) )
    src_extended_height = int( np.max(borders[1,:])-np.min(borders[1,:]) )
    shift_x = int( -np.min([0, np.min(borders[0,:]) ]) +10)
    shift_y = int( -np.min([0, np.min(borders[1,:]) ]) +10)
    new_image = np.ones((src_extended_height+20,src_extended_width+20,3))*255

    new_image [shift_y:shift_y+dst_y_size,shift_x:shift_x+dst_x_size,:] = dst_image/255

    # Now take all (x,y) and map to another (x',y')
    u_vec = np.transpose( np.arange(0,src_x_size) )
    v_vec = np.transpose( np.arange(0,src_y_size) )

    uv_mesh = np.zeros( (2,1) )

    xx, yy = np.meshgrid(u_vec,v_vec)
    uv_mesh = np.append(np.reshape(xx, (1, src_x_size*src_y_size)), np.reshape(yy, (1, src_x_size*src_y_size)), axis=0)

    uv_mesh = np.delete( uv_mesh, 0, axis =1)

    new_uv_mesh = mapping(H, uv_mesh)
    for i in np.arange(0,np.shape(uv_mesh)[1]):
        dst_point = new_uv_mesh[:,i]
        src_point = uv_mesh[:,i]
        src_point_val = src_image[int(src_point[1]),int(src_point[0]),:]/255
        new_image[int(shift_y + dst_point[1]), int(shift_x + dst_point[0]), :] = src_point_val

    plt.figure()
    panplot = plt.imshow(new_image)
    plt.title('Forward Mapping')
    plt.show()

    return


def bilinear_interpulator (image,pixel):
    u1, v1 = np.floor(pixel)
    u2, v2 = np.ceil(pixel)
    u1 = int(u1)
    u2 = int(u2)
    v1 = int(v1)
    v2 = int(v2)
    x, y = np.subtract(pixel, [u1, v1])
    left_down_val = image[u1,v2,:] *( (1-x)*y )
    left_up_val = image[u1,v1,:] *( (1-x)*(1-y) )
    right_up_val = image[u2,v1,:] *( x*(1-y) )
    right_down_val = image[u2,v2,:] *( x*y )

    return ( left_down_val+left_up_val+right_up_val+right_down_val )


def full_backward_mapping (H,src_image, dst_image):
    # H maps from src to dest
    # Hinv maps from dest to src

    # 1. First get the size of the mapped dst
    dst_y_size, dst_x_size , ___ = np.shape(dst_image)
    src_y_size, src_x_size , ___ = np.shape(src_image)

    corners = np.transpose( [[0,0],[dst_x_size,0],[0,dst_y_size],[dst_x_size,dst_y_size]] )
    src_corners = np.transpose( [[0,0],[src_x_size,0],[0,src_y_size],[src_x_size,src_y_size]] )
    borders = mapping(H, src_corners)

    borders = np.append(borders, corners, axis=1)
    src_extended_width = int(np.max(borders[0, :]) - np.min(borders[0, :]))
    src_extended_height = int( np.max(borders[1,:])-np.min(borders[1,:]) )
    shift_x = int(-np.min([0, np.min(borders[0, :])]) +10)
    shift_y = int(-np.min([0, np.min(borders[1,:]) ]) +10)
    new_image = np.ones((src_extended_height+20,src_extended_width+20,3))*255

    Hinv = np.linalg.inv(H)

    # 2. Run over the new image and ask from where it came:
    new_image_y_size, new_image_x_size , ___ = np.shape(new_image)
    u_vec = np.subtract( np.transpose( np.arange(0,new_image_x_size) ), shift_x )
    v_vec = np.subtract( np.transpose( np.arange(0,new_image_y_size) ), shift_y )
    uv_mesh = np.zeros((2,1))

    xx, yy = np.meshgrid(u_vec,v_vec)
    uv_mesh = np.append(np.reshape(xx, (1, new_image_x_size*new_image_y_size)), np.reshape(yy, (1, new_image_x_size*new_image_y_size)), axis=0)

    uv_mesh = np.delete( uv_mesh, 0, axis =1)
    new_uv_mesh = mapping(Hinv, uv_mesh)

    # 3. Run over all the src points and put the mapped from dst
    # uv_mesh - holds src points, new_uv_mesh - holds e_dst points
    for i in np.arange(0,np.shape(uv_mesh)[1]):
        src_point = new_uv_mesh[:,i]
        src_v_point = src_point[1]
        src_u_point = src_point[0]
        dst_point = uv_mesh[:, i]
        # If we "came" from valid coordinate (in picture)
        if (np.floor(src_v_point) >= 0 and np.floor(src_u_point) >= 0 and np.ceil(src_v_point) < src_y_size and np.ceil(src_u_point) < src_x_size):
            src_point_val = bilinear_interpulator(src_image,[src_v_point,src_u_point])/255
            new_image[int(shift_y + dst_point[1]), int(shift_x + dst_point[0]), :] = src_point_val

    new_image [shift_y:shift_y+dst_y_size,shift_x:shift_x+dst_x_size,:] = dst_image/255

    return new_image



def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    H_ransac = compute_homography(mp_src, mp_dst, inliers_percent, max_err)
    new_image = full_backward_mapping (H_ransac,img_src, img_dst)
    return new_image

###############################################################

# Read the data:
img_src = mpimg.imread('src.jpg')
img_dst = mpimg.imread('dst.jpg')
# matches = scipy.io.loadmat('matches') #matching points and some outliers
matches = scipy.io.loadmat('matches_perfect') #loading perfect matches
match_p_dst = matches['match_p_dst'].astype(float)
match_p_src = matches['match_p_src'].astype(float)

H_naive = compute_homography_naive(match_p_src, match_p_dst)
full_forward_mapping (H_naive,img_src, img_dst)

