from flask import Flask, request, jsonify
import threading
import os
import sys
from argparse import Namespace
import json
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
import json
from urllib.request import urlretrieve
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.paste_pic import paste_pic

import inference_1

app = Flask(__name__)

@app.route("/generate_video", methods=["POST"])
def generate_video():
    data = request.get_json()
    # source_image_url = data["source_image"]
    # filename = os.path.basename(data["source_image"])
    # name, extension = os.path.splitext(filename)
    # print(extension)
    # if os.name == 'nt':  # 如果是 Windows 系统
    #     local_image_path = "E:\\SadTalker\\examples\\source_image\\" + data["uuid"] + extension  # 本地保存路径
    # else:  # 如果是 Linux 系统
    #     local_image_path = "./images/source_image.jpg"  # 本地保存路径
    #
    # urlretrieve(source_image_url, local_image_path)
    #
    #
    # parser = ArgumentParser()
    # parser.add_argument("--driven_audio", default=data["driven_audio"],
    #                     help="path to driven audio")
    # parser.add_argument("--source_image", default=local_image_path,
    #                     help="path to source image")
    # parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    # parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    # parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    # parser.add_argument("--result_dir", default=data["result_dir"], help="path to output")
    # parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    # parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    # parser.add_argument("--expression_scale", type=float, default=1., help="the batch size of facerender")
    # parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    # parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    # parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    # parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer, [gfpgan]")
    # parser.add_argument('--full_img_enhancer', type=str, default=None, help="Full image enhancer, [gfpgan]")
    # parser.add_argument("--cpu", dest="cpu", action="store_true")
    # parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks")
    # parser.add_argument("--still", action="store_true")
    # parser.add_argument("--preprocess", default='crop', choices=['crop', 'resize'])
    #
    # # net structure and parameters
    # parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'],
    #                     help='useless')
    # parser.add_argument('--init_path', type=str, default=None, help='Useless')
    # parser.add_argument('--use_last_fc', default=False, help='zero initialize the last fc')
    # parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    # parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    #
    # # default renderer parameters
    # parser.add_argument('--focal', type=float, default=1015.)
    # parser.add_argument('--center', type=float, default=112.)
    # parser.add_argument('--camera_d', type=float, default=10.)
    # parser.add_argument('--z_near', type=float, default=5.)
    # parser.add_argument('--z_far', type=float, default=15.)
    #
    # args = parser.parse_args()
    #
    # if torch.cuda.is_available() and not args.cpu:
    #     args.device = "cuda"
    # else:
    #     args.device = "cpu"
    time.sleep(180)
    print("jinlaile" + data["uuid"])
    # video_url = inference_1.generate_video(args)
    video_url = ''
    # filename = video_url[3:]  # 获取 2023_04_09_11.56.39\f0549474-5894-40b1-9042-ee5a8f9b20fb##japanese_full.mp4
    # print(filename)
    return jsonify({"video_url": video_url})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)