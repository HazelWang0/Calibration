from utils.trans import json_write,json_write_llff
from utils.calib0218 import set_calibration


if __name__ == '__main__':
    f = 612
    c2w_metrix = set_calibration()
    json_write_llff(c2w_metrix,f)
    
