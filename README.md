相机校准
# how to run 
```
python run.py --[file]
```
[file]图片所在路径


>## to change the numbers of chessboard grid
```
python run.py --[w,h]
```
[w,h]：对应棋盘长、宽的格子数量

默认棋盘格规格为12乘9，格点长度0.02m，由于opencv输入参数为内角点个数，所以默认输入参数为11乘8
