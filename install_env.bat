@echo off
echo ========================================
echo  SZAtt-Net 环境一键安装脚本
echo ========================================
echo.

echo [1/2] 正在安装 Python 依赖包...
pip install numpy torch torchvision scikit-learn mne matplotlib seaborn scipy

echo.
echo [2/2] 正在创建数据目录...
if not exist "SZAtt-Net-main\norm_repod" mkdir "SZAtt-Net-main\norm_repod"
if not exist "SZAtt-Net-main\sch_repod" mkdir "SZAtt-Net-main\sch_repod"
if not exist "SZAtt-Net-main\COBRE-2D" mkdir "SZAtt-Net-main\COBRE-2D"

echo.
echo ========================================
echo  安装完成！
echo ========================================
echo.
echo 数据目录已创建：
echo   - SZAtt-Net-main\norm_repod\  (放入正常对照 .edf 文件)
echo   - SZAtt-Net-main\sch_repod\   (放入精神分裂症 .edf 文件)
echo   - SZAtt-Net-main\COBRE-2D\    (放入 fMRI .npy/.npz 文件)
echo.
echo 运行命令: python ours.py
echo.
pause
