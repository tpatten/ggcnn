import os
import numpy as np
import torch
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import image
import grasp

center = [302, 285]
left = 135
top = 152
output_size = 300

def get_rgb_image(filename):
    rgb_img = image.Image.from_file(filename)
    rgb_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
    rgb_img.resize((output_size, output_size))
    normalise=False
    if normalise:
        rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img

def get_depth_image(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.tiff':
        depth_img = image.DepthImage.from_tiff(filename)
    elif file_extension == '.pcd':
        depth_img = image.DepthImage.from_pcd(filename, (480, 640))
    elif file_extension == '.png':
        depth_img = image.DepthImage.from_png(filename)
    else:
        raise ValueError('Cannot load depth image with extension', file_extension)
    print('depth_img', depth_img)
    print('depth_img.img', depth_img.img)
    print('depth_img.img.shape', depth_img.img.shape)
    depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
    depth_img.normalise()
    depth_img.resize((output_size, output_size))
    return depth_img.img
    
def get_images_from_cornell(data_path, id):
    rgb_file = data_path + id + 'r.png'
    dep_file = data_path + id + 'd.tiff'
    return get_rgb_image(rgb_file), get_depth_image(dep_file)

def get_images_from_ocid(data_path, id):
    rgb_file = data_path + 'rgb/' + id + '.png'
    dep_file = data_path + 'depth/' + id + '.png'
    return get_rgb_image(rgb_file), get_depth_image(dep_file)

def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a GG-CNN output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = grasp.Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2

        grasps.append(g)

    return grasps

def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    #plt.show()
    plt.savefig('/container/Data/ggcnn_output.png')
    input("Press Enter to continue...")
