@echo off
chcp 65001
call conda activate %1
if errorlevel 1 (
    echo Error: 请在run.bat中修改your_env_name为你的环境名
    pause
    exit /b
)

call conda info | findstr "active environment"

for /f "tokens=6 delims= " %%i in ('nvcc --version ^| findstr /C:"release"') do (
    for /f "tokens=1,2 delims=." %%j in ("%%i") do set CUDA_VERSION=%%j.%%k
)
set CUDA_VERSION=%CUDA_VERSION:~1%
echo CUDA_VERSION: %CUDA_VERSION%

for /f "tokens=2" %%i in ('python --version') do (
    for /f "tokens=1,2 delims=." %%j in ("%%i") do set PYTHON_VERSION=%%j.%%k
)
echo PYTHON_VERSION: %PYTHON_VERSION%

echo 正在检查torchvision torchsummary tensorboard packaging 是否安装...
for %%A in (torchvision torchsummary tensorboard packaging) do (
    pip show %%A >nul 2>&1
    if errorlevel 1 (
        echo %%A 未安装
    ) else (
        echo %%A 已安装
    )
)

echo 正在检查torch 是否安装...
for %%A in (torch) do (
    pip show %%A >nul 2>&1
    if errorlevel 1 (
        echo %%A 未安装
    ) else (
        echo %%A 已安装
        echo 正在检查torch是否支持CUDA...

        python -c "import torch; print('torch 支持 CUDA: ' + ('是' if torch.cuda.is_available() else '否'))"
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

        echo %errorlevel%
    )
)

pause

