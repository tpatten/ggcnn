import os
import argparse
import logging
import numpy as np
import torch
from skimage.filters import gaussian
import torch.utils.data
from utils.dataset_processing import evaluation, image

logging.basicConfig(level=logging.INFO)

center = [302, 285]
left = 135
top = 152
output_size = 300

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    return parser.parse_args()

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

if __name__ == '__main__':
    args = parse_args()

    # Load example data

    #root_path = '/container/Data/CornellGraspDatasetSmall/'
    #example = 'pcd0100'
    #rgb_img, dep_img = get_images_from_cornell(root_path, example)

    root_path = '/container/Data/OCID-dataset/ARID10/floor/bottom/fruits/seq07/'
    example = 'result_2018-08-24-17-29-40'
    rgb_img, dep_img = get_images_from_ocid(root_path, example)

    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    x1 = numpy_to_torch(dep_img)
    x1 = x1.view(1, x1.shape[0], x1.shape[1], x1.shape[2])
    with torch.no_grad():
        xc1 = x1.to(device)
        pos_output, cos_output, sin_output, width_output = net.forward(xc1)
        q_img, ang_img, width_img = post_process_output(pos_output, cos_output, sin_output, width_output)

    if args.vis:
        evaluation.plot_output(rgb_img, dep_img, q_img, ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)