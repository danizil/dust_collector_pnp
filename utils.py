import torch
import numpy as np
import cv2


def rodriguez_to_matrix(vector):
    theta = np.linalg.norm(vector)
    
    if theta == 0:
        return np.eye(3)
    
    vector = vector / theta
    K = np.array([[0, -vector[2], vector[1]],
                  [vector[2], 0, -vector[0]],
                  [-vector[1], vector[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)
    return R


def rotate_translate(rigid_3d, angle_axis=np.array([0.0,0.0,0.0]), translation=np.array([0.0,0.0,0.0])):
    ''' rotate and translate the 3D points'''
    assert angle_axis.shape == (3,)
    assert translation.shape == (3,)
    
    #* rotation matrix from angle axis vector
    R = rodriguez_to_matrix(angle_axis)
    rotated_translated = rigid_3d@R.T + translation
    return rotated_translated

def project_on_camera(rigid_3d, camera_mat=np.eye(3)):
    ''' project the 3D points to the 2D image'''
    # rigid_3d is the 3D points in the world
    # camera_mat is the camera matrix
    # image_hw is the height and width of the image
    # the output is the 2D points in the image
    XYW = rigid_3d@camera_mat.T
    XY = XYW[:, :2] / (XYW[:, 2] + 10**-8).reshape(-1, 1)
    return XY

def track_points(initial_points, frames):
    ''' gpt solution to track the black round qtm sensor points on the images based on optical flow, works not bad until the points disappear'''
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Convert points to the needed format
    initial_points = np.float32(initial_points).reshape(-1, 1, 2)

    # Array to hold the points for all frames
    points_all_frames = np.zeros((frames.shape[0],4, 2), dtype=np.float32)

    # Set initial points
    points_all_frames[0] = initial_points.reshape(4, 2)

    # Process each frame
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, frames.shape[0]):
        frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, initial_points, None, **lk_params)
        
        # Select good points
        good_new = new_points[status == 1]
        
        # Update the points array
        points_all_frames[i] = good_new.reshape(4, 2)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        initial_points = good_new.reshape(-1, 1, 2)

    return points_all_frames


def rodrigues_to_matrix_torch(vector):
    theta = torch.norm(vector)
    
    if theta == 0:
        return torch.eye(3)
    
    vector = vector / theta
    K = torch.tensor([[0, -vector[2], vector[1]],
                      [vector[2], 0, -vector[0]],
                      [-vector[1], vector[0], 0]])
    R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R

def rotate_translate_torch(rigid_3d, angle_axis=torch.tensor([0.0,0.0,0.0]), translation=torch.tensor([0.0,0.0,0.0])):
    ''' rotate and translate the 3D points'''
    assert angle_axis.shape == (3,)
    assert translation.shape == (3,)
    
    #* rotation matrix from angle axis vector
    R = rodrigues_to_matrix_torch(angle_axis)
    rotated_translated = rigid_3d@R.T + translation
    return rotated_translated

def project_on_camera_torch(rigid_3d, camera_mat=torch.eye(3)):
    ''' project the 3D points to the 2D image'''
    # rigid_3d is the 3D points in the world
    # camera_mat is the camera matrix
    # image_hw is the height and width of the image
    # the output is the 2D points in the image
    XYW = rigid_3d@camera_mat.T
    XY = XYW[:, :2] / (XYW[:, 2] + 10**-8).unsqueeze(1)
    return XY

def sum_distance(points0, points1):
    ''' calculate the sum of the distance between the points in points0 and the points in points1'''
    return torch.sum(torch.norm(points0 - points1, dim=1))