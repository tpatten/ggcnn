import argparse
import logging
import torch.utils.data
import helpers

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load example data

    #root_path = '/container/Data/CornellGraspDatasetSmall/'
    #example = 'pcd0100'
    #rgb_img, dep_img = get_images_from_cornell(root_path, example)

    root_path = '/container/Data/OCID-dataset/ARID10/floor/bottom/fruits/seq07/'
    example = 'result_2018-08-24-17-29-40'
    rgb_img, dep_img = helpers.get_images_from_ocid(root_path, example)

    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    x1 = helpers.numpy_to_torch(dep_img)
    x1 = x1.view(1, x1.shape[0], x1.shape[1], x1.shape[2])
    with torch.no_grad():
        xc1 = x1.to(device)
        pos_output, cos_output, sin_output, width_output = net.forward(xc1)
        q_img, ang_img, width_img = helpers.post_process_output(pos_output, cos_output, sin_output, width_output)

    if args.vis:
        helpers.plot_output(rgb_img, dep_img, q_img, ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)