import os
import cv2
import torch
import numpy as np
import argparse
from src.constant import phantom_FB
from src.calibrator import FanbeamCalibrator
from src.utils import load_cameras_init

def main(args):
    cameras = load_cameras_init()
    calibrator = FanbeamCalibrator(phantom_FB, cameras, optim_f=args.optim_f, optim_beads=args.optim_beads, n_iter=args.n_iter)
    calibrator.save_fig = args.save_fig
    calibrator.run()
    plt = calibrator.viz()

    params = calibrator.params # Tx, Tz, theta
    params_f = calibrator.params_f # f
    params_beads = calibrator.params_beads # beads_3d
    params_all = torch.cat([params, params_f.unsqueeze(1)], dim=1)

    save_dir = './results/'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'calibration_results.npy'), params_all.detach().numpy())
    np.save(os.path.join(save_dir, 'beads_optimized.npy'), params_beads.detach().numpy())

    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim_f', type=bool, default=False, help='Optimize f')
    parser.add_argument('--optim_beads', type=bool, default=False, help='Optimize beads')
    parser.add_argument('--n_iter', type=int, default=20000, help='Number of iterations')
    parser.add_argument('--save_fig', type=bool, default=False, help='Save figure')
    args = parser.parse_args()
    main(args)