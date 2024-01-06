@echo off
chcp 65001

set "env_name=%1"
call conda activate %env_name%
if errorlevel 1 (
    echo Error: 请在run.bat中修改your_env_name为你的环境名
    pause
    exit /b
)
echo 正在检查ipykernel...
pip show ipykernel >nul 2>&1
if errorlevel 1 (
    echo Error: ipykernel未安装, 正在安装...
    pip install ipykernel
    if errorlevel 1 (
        echo Error: ipykernel安装失败
        pause
        exit /b
    )
    python -m ipykernel install --user --name %env_name% --display-name %env_name%
    echo ipykernel安装成功
)
echo 正在启动jupyter notebook...
echo 启动后请手动更改内核为 %env_name%
for /L %%i in (1, 1, 5) do (
    timeout /t 5 /nobreak
    echo 正在启动jupyter notebook... 请等待 %%i/5
)
start jupyter notebook note.ipynb
pause