from utils import *
import cv2
import numpy as np

def find_pose_rigid_im_pts(rigid_3d, impts, camera_mat):
    ''' find the pose of the rigid body based on the 3D points and the 2D points in the image'''
    # rigid_3d is the 3D points in the world
    # im_pts is the 2D points in the image
    # camera_mat is the camera matrix
    distCoeffs = np.zeros(4)

    retval, rvec, tvec = cv2.solvePnP(rigid_3d, impts, camera_mat, distCoeffs, flags=cv2.SOLVEPNP_P3P)
    #! Note that several solutions may be found, and we must chose the ones that are in the correct angle range.
    rotated_translated = rotate_translate(rigid_3d, rvec[:,0], translation=tvec[:,0])
    return rvec, tvec, retval, rotated_translated


def find_poses_rigid_im_pts_set(rigid_3d, impts_set, camera_mat):
    ''' find the pose of the rigid body based on the 3D points and the 2D points in the image'''
    # rigid_3d is the 3D points in the world
    # im_pts is the 2D points in the image
    # camera_mat is the camera matrix
    distCoeffs = np.zeros(4)
    rvecs = []
    tvecs = []
    retvals = []
    rotated_translated_projecteds = []
    for i,impts in enumerate(impts_set):
        rvec, tvec, retval, rotated_translated = find_pose_rigid_im_pts(rigid_3d, impts, camera_mat)
        
        rotated_translated_projected = project_on_camera(rotated_translated, camera_mat)

        rvecs.append(rvec)
        tvecs.append(tvec)
        retvals.append(retval)
        rotated_translated_projecteds.append(rotated_translated_projected)
        if retval:
            rigid_3d = rotated_translated
        else:
            print(f'couldnt find the pose for frame {i}')
    return rigid_3d, rotated_translated_projecteds, rvecs, tvecs, retvals 