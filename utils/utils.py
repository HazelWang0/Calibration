import argparse

def arse_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, \
        default='/root/vmip/myProject/nerf/camera/Calibration_ZhangZhengyou_Method/pic/RGB_camera_calib_img',\
        help="图片所在文件位置")
    parser.add_argument('--config', type=format, default=[11,8],help="格子长宽数")
    args = parser.parse_args()

    return args