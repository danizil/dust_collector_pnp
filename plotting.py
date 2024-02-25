from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_rigid_3d(rigid, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(rigid[:,0], rigid[:,1], rigid[:,2])
    
    ax.plot([-150, 150], [0, 0], [0, 0], color='red')   # x-axis
    ax.plot([0, 0], [-150, 150], [0, 0], color='green') # y-axis
    ax.plot([0, 0], [0, 0], [-60, 600], color='blue')  # z-axis
    
    # Set the maximum number of ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(1))
    ax.zaxis.set_major_locator(MaxNLocator(1))

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_image_with_points(image, points_list, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    for points in points_list:
        ax.plot(points[:,0], points[:,1])
    