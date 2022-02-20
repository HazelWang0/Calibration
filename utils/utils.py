import argparse

def arse_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, \
        default='./chess/images',\
        help="图片所在文件位置")
    parser.add_argument('--inter_corner_shape', type=format, default=(11,8),help="格子长宽数")
    parser.add_argument('--size_per_grid', type=format, default=0.02)
    args = parser.parse_args()

    return args