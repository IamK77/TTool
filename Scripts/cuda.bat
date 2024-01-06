@echo off
chcp 65001
call conda activate %1

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
echo 建议使用Python3.9

if "%PYTHON_VERSION%"=="3.5" (
    echo 不支持Python3.8以下的版本
    pause
    exit /b
) else if "%PYTHON_VERSION%"=="3.6" (
    echo 不支持Python3.8以下的版本
    pause
    exit /b
) else if "%PYTHON_VERSION%"=="3.7" (
    echo 不支持Python3.8以下的版本
    pause
    exit /b
) else if "%PYTHON_VERSION%"=="3.12" (
    echo 不支持Python3.12及以上的版本
    pause
    exit /b
)

echo 正在下载torch torchvision --cuda...
if "%PYTHON_VERSION%"=="3.9" (
    if "%CUDA_VERSION%"=="11.6" (
        curl -O https://download.pytorch.org/whl/cu116/torch-1.13.1%%2Bcu116-cp39-cp39-win_amd64.whl#sha256=80a6b55915ac72c087ab85122289431fde5c5a4c85ca83a38c6d11a7ecbfdb35
        curl -O https://download.pytorch.org/whl/cu116/torchvision-0.14.1%%2Bcu116-cp39-cp39-win_amd64.whl#sha256=4b75cfe80d1e778f252fce94a7dd4ea35bc66a10efd53c4c63910ee95425face
    ) else if "%CUDA_VERSION%"=="10.1" (
        curl -O https://download.pytorch.org/whl/cu101/torch-1.8.1%%2Bcu101-cp39-cp39-win_amd64.whl#sha256=c3281f9f52b0e8d45f15b3e7164d80bb8d4e57b36751afb3b1e89f4328aaf92e
        curl -O https://download.pytorch.org/whl/cu101/torchvision-0.9.1%%2Bcu101-cp39-cp39-win_amd64.whl#sha256=cd53058257c721f727856866cdd009bf0d7251cc111caf646ec5c147f3084bcb
    ) else if "%CUDA_VERSION%"=="10.2" (
        curl -O https://download.pytorch.org/whl/cu102/torch-1.9.1%%2Bcu102-cp39-cp39-win_amd64.whl#sha256=cfd2f0d63391e258be9e9c10fbc56ae879d0a9e70f2f6e09964b3743f4dc9fd4
        curl -O https://download.pytorch.org/whl/cu102/torchvision-0.9.1%%2Bcu102-cp39-cp39-win_amd64.whl#sha256=52e1f5abbb8e9e08f7e92434d2b9a2e6f75b34a1d0e360f2c1d13de216b33f2a
    ) else if "%CUDA_VERSION%"=="11.0" (
        curl -O https://download.pytorch.org/whl/cu110/torch-1.7.1%%2Bcu110-cp39-cp39-win_amd64.whl#sha256=bc1a73f18ff93192ee1aad4714b3b454eb7dee3a030b28a42b5f2f3e63ec9169
        curl -O https://download.pytorch.org/whl/cu110/torchvision-0.8.2%%2Bcu110-cp39-cp39-win_amd64.whl#sha256=632cc5ae100f1f403b7b55d6e6b38a053634b2a29f91b4d83943fa6f2aaf900d
    ) else if "%CUDA_VERSION%"=="11.1" (
        curl -O https://download.pytorch.org/whl/cu111/torch-1.9.1%%2Bcu111-cp39-cp39-win_amd64.whl#sha256=8f738eb4104e1028031353ee20e887ad9d8f282aef013935611d12dec102375b
        curl -O https://download.pytorch.org/whl/cu111/torchvision-0.9.1%%2Bcu111-cp39-cp39-win_amd64.whl#sha256=640be95627d1c6cf7691a67e05f27ca57d9f7bd3c5d374e43a64e2a44e23657a
    ) else if "%CUDA_VERSION%"=="11.3" (
        curl -O https://download.pytorch.org/whl/cu113/torch-1.12.1%%2Bcu113-cp39-cp39-win_amd64.whl#sha256=b20d19c379c8fc71f04f49b35c867732f0dfa19ec046af218dc77458c05424f7
        curl -O https://download.pytorch.org/whl/cu113/torchvision-0.13.1%%2Bcu113-cp39-cp39-win_amd64.whl#sha256=bb22ac5e7bd526c45b00825dcf0e300fde14d8f1c0459833d276cf12f4642a61
    ) else if "%CUDA_VERSION%"=="11.5" (
        curl -O https://download.pytorch.org/whl/cu115/torch-1.11.0%%2Bcu115-cp39-cp39-win_amd64.whl#sha256=f67fbe8aa4e720077f0d34349228ecad58afd84395be437eedaa7b53d1baa3da
        curl -O https://download.pytorch.org/whl/cu115/torchvision-0.12.0%%2Bcu115-cp39-cp39-win_amd64.whl#sha256=838875ebaabb2562ad40c764f086b634fc2766b68a6b0c062da7d09be3a04745
    ) else if "%CUDA_VERSION%"=="11.7" (
        curl -O https://download.pytorch.org/whl/cu117/torch-1.13.1%%2Bcu117-cp39-cp39-win_amd64.whl#sha256=e775fa85f412bd1bf816b8798dadb3b852b71e33e8008e9db29b6190ed94fe27
        curl -O https://download.pytorch.org/whl/cu117/torchvision-0.15.2%%2Bcu117-cp39-cp39-win_amd64.whl#sha256=8e0701a162cca13cee637397d7a67248bfeec35086cc753a1c513b2ea1dd476e
    ) else if "%CUDA_VERSION%"=="11.8" (
        curl -O https://download.pytorch.org/whl/cu118/torch-2.1.1%%2Bcu118-cp39-cp39-win_amd64.whl#sha256=c883a237149b3435af3b4f544f990dc946c428fd531a9d14be0407ee2112b581
        curl -O https://download.pytorch.org/whl/cu118/torchvision-0.16.1%%2Bcu118-cp39-cp39-win_amd64.whl#sha256=84f97feb1bd0e5256d40313b0cfc16c2d25b6fb14934f512a5ab048def648340
    ) else if "%CUDA_VERSION%"=="12.1" (
        curl -O https://download.pytorch.org/whl/cu121/torch-2.1.1%%2Bcu121-cp39-cp39-win_amd64.whl#sha256=2b5b58eff9efef68c25c1260e28e516c665fedae241ef426a43381d7a9076e64
        curl -O https://download.pytorch.org/whl/cu121/torchvision-0.16.1%%2Bcu121-cp39-cp39-win_amd64.whl#sha256=35e08d36a3ac242b2b775193699b3882a3e891e1bd65857b3d481568feccf708
    )
) else if "%PYTHON_VERSION%"=="3.8" (
    if "%CUDA_VERSION%"=="11.6" (
        curl -O https://download.pytorch.org/whl/cu116/torch-1.13.1%%2Bcu116-cp38-cp38-win_amd64.whl#sha256=1c33942d411d4dee25e56755cfd09538f53a497a6f0453d54ce96a5ca341627b
        curl -O https://download.pytorch.org/whl/cu116/torchvision-0.14.1%%2Bcu116-cp38-cp38-win_amd64.whl#sha256=fefa6bee4c3019723320c6e554e400d6781ccecce99c1772a165efd0696b3462
    ) else if "%CUDA_VERSION%"=="9.2" (
        curl -O https://download.pytorch.org/whl/cu92/torch-1.5.1%%2Bcu92-cp38-cp38-win_amd64.whl#sha256=c5f43abeebf9ee5756e2320b3797810d31b3b7dbb978791f8f37be4c202c3265
        curl -O https://download.pytorch.org/whl/cu92/torchvision-0.6.1%%2Bcu92-cp38-cp38-win_amd64.whl#sha256=0cdbc0c584b9b8bee89efe3b6394b5f65ad66f4e564b2de7ab1321a9b3835509
    ) else if "%CUDA_VERSION%"=="10.1" (
        curl -O https://download.pytorch.org/whl/cu101/torch-1.8.1%%2Bcu101-cp38-cp38-win_amd64.whl#sha256=d66f2d8ed642d4dcf9994cd12801aeef4038314e3ecf82f01f5b64555cfa2a87
        curl -O https://download.pytorch.org/whl/cu101/torchvision-0.9.1%%2Bcu101-cp38-cp38-win_amd64.whl#sha256=37636aac0850794762a371108db51bf2b9934967a2442b91068f00d34d60d1aa
    ) else if "%CUDA_VERSION%"=="10.2" (
        curl -O https://download.pytorch.org/whl/cu102/torch-1.9.1%%2Bcu102-cp38-cp38-win_amd64.whl#sha256=0f3f93913267d2b5d10fd4dd6a7db06c2f710cb6559d72b048f7dd130514025b
        curl -O https://download.pytorch.org/whl/cu102/torchvision-0.11.3%%2Bcu102-cp38-cp38-win_amd64.whl#sha256=c7ca625ba5eb8e6ebd393f8be1b6fa752695cbbb24ea80d533fa0a221180a642
    ) else if "%CUDA_VERSION%"=="11.0" (
        curl -O https://download.pytorch.org/whl/cu110/torch-1.7.1%%2Bcu110-cp38-cp38-win_amd64.whl#sha256=3a13c7117df5f89739878dba3d3ba833deb0fcfe7a657fe346ebdba31daec00a
        curl -O https://download.pytorch.org/whl/cu110/torchvision-0.8.2%%2Bcu110-cp38-cp38-win_amd64.whl#sha256=8d4974f19f00e041a091f566912b7c6a9960d5e5f103730f4290ea32a6f9f01a
    ) else if "%CUDA_VERSION%"=="11.1" (
        curl -O https://download.pytorch.org/whl/cu111/torch-1.9.1%%2Bcu111-cp38-cp38-win_amd64.whl#sha256=9860b5e564c5a4faec1215f650a9e4f3a64ea4614d4c1824d11048ab1b0b4f76
        curl -O https://download.pytorch.org/whl/cu111/torchvision-0.9.1%%2Bcu111-cp38-cp38-win_amd64.whl#sha256=f1d50c161e068adda52dd7c1ed2a1cdd4fbe77ed7afd1902fcc838290b5e8f96
    ) else if "%CUDA_VERSION%"=="11.3" (
        curl -O https://download.pytorch.org/whl/cu113/torch-1.12.1%%2Bcu113-cp38-cp38-win_amd64.whl#sha256=9852174c19d753b071393da6f45a2b3f68d94cfbf1ee85a85d1a1870da5aec48
        curl -O https://download.pytorch.org/whl/cu113/torchvision-0.13.1%%2Bcu113-cp38-cp38-win_amd64.whl#sha256=1b87006748e9eb6bd5bb9f544e685d37b6bdae07b48370bdfe3bbd4ac8f057cf
    ) else if "%CUDA_VERSION%"=="11.5" (
        curl -O https://download.pytorch.org/whl/cu115/torch-1.11.0%%2Bcu115-cp38-cp38-win_amd64.whl#sha256=b143f8a9a54c1e09e389e04f76fe9c4dd0c64c511eb784ce16ffee3c110e7131
        curl -O https://download.pytorch.org/whl/cu115/torchvision-0.12.0%%2Bcu115-cp38-cp38-win_amd64.whl#sha256=7de3af23e66b8351587c5075de608404c9b7d91721c552834c7acbb418dc5aa1
    ) else if "%CUDA_VERSION%"=="11.7" (
        curl -O https://download.pytorch.org/whl/cu117/torch-2.0.1%%2Bcu117-cp38-cp38-win_amd64.whl#sha256=0a56cf5d99f1c7fa29c328a6737c5e5108fa71d8f021c074f4ff0de9e8969302
        curl -O https://download.pytorch.org/whl/cu117/torchvision-0.15.2%%2Bcu117-cp38-cp38-win_amd64.whl#sha256=b41173e1032ffa7aa3f6810682dcd64c43ff0d33f97fbf3195fbd3d4c6629095
    ) else if "%CUDA_VERSION%"=="11.8" (
        curl -O https://download.pytorch.org/whl/cu118/torch-2.1.1%%2Bcu118-cp38-cp38-win_amd64.whl#sha256=43e72fc0043f47dfd85ba5326653a9d3dc173e1348108d75beb09d9483537233
        curl -O https://download.pytorch.org/whl/cu118/torchvision-0.16.1%%2Bcu118-cp38-cp38-win_amd64.whl#sha256=88e68a7f5b705688553d7c73f2acf7b6e4ebcca6ecba3883fe3f173980a89f0e
    ) else if "%CUDA_VERSION%"=="12.1" (
        curl -O https://download.pytorch.org/whl/cu121/torch-2.1.1%%2Bcu121-cp38-cp38-win_amd64.whl#sha256=1c2891bba2e76a07cd3395c165c6196d5ee0a7c6cba4f52d7aed4fe125ea1ddf
        curl -O https://download.pytorch.org/whl/cu121/torchvision-0.16.1%%2Bcu121-cp38-cp38-win_amd64.whl#sha256=1321cfda1a6e1a92a78747c010aba581a452c714acaf3c8a95bf7de16760f10c
    )
) else if "%PYTHON_VERSION%"=="3.10" (
    if "%CUDA_VERSION%"=="11.6" (
        curl -O https://download.pytorch.org/whl/cu116/torch-1.13.1%%2Bcu116-cp310-cp310-win_amd64.whl#sha256=6d59b73bbd83eee53e7978925168fe068709e1344050fdabf4043695084b2ccc
        curl -O https://download.pytorch.org/whl/cu116/torchvision-0.14.1%%2Bcu116-cp310-cp310-win_amd64.whl#sha256=0481f119c1ca5bf3704d0848161be956db04d26c01ffb95b3b3b08cbdfa301c2
    ) else if "%CUDA_VERSION%"=="11.3" (
        curl -O https://download.pytorch.org/whl/cu113/torch-1.12.1%%2Bcu113-cp310-cp310-win_amd64.whl#sha256=8b83783f6537b48b75b6ba5d62d7acfd94546689223bb0d3a7d39886148b8d17
        curl -O https://download.pytorch.org/whl/cu113/torchvision-0.13.1%%2Bcu113-cp310-cp310-win_amd64.whl#sha256=ee7882e31ef973e2cf067d88ddef0a8f5b369c2dd9195236b90b479de0507c15
    ) else if "%CUDA_VERSION%"=="11.5" (
        curl -O https://download.pytorch.org/whl/cu115/torch-1.11.0%%2Bcu115-cp310-cp310-win_amd64.whl#sha256=6c66502d4e30464abd8ede9b00ef85ac7eaf569bdf53663375a0ed3f49c4f1e5
        curl -O https://download.pytorch.org/whl/cu115/torchvision-0.12.0%%2Bcu115-cp310-cp310-win_amd64.whl#sha256=8b7cdcda36bd5af9c0a1c934a680a48ddfe67df7b2710d2b7f61ef06850662dd
    ) else if "%CUDA_VERSION%"=="11.7" (
        curl -O https://download.pytorch.org/whl/cu117/torch-1.13.0%%2Bcu117-cp310-cp310-win_amd64.whl#sha256=a0b87b3c87e16f472aa5c87dd31a071211cdec972de280ad1aacff0d245354f8
        curl -O https://download.pytorch.org/whl/cu117/torchvision-0.15.2%%2Bcu117-cp310-cp310-win_amd64.whl#sha256=6fe4d68a8a19b9c6a5d4c7b6e451b112af5a88c557ef9082b608757514653e8c
    ) else if "%CUDA_VERSION%"=="11.8" (
        curl -O https://download.pytorch.org/whl/cu118/torch-2.1.1%%2Bcu118-cp310-cp310-win_amd64.whl#sha256=765e93911984c813ddf74427eecd70c1efc785af7c231777632954b1bd1429d3
        curl -O https://download.pytorch.org/whl/cu118/torchvision-0.16.1%%2Bcu118-cp310-cp310-win_amd64.whl#sha256=4904755d361040ef41a79e15d6b1ca62a760f640670629fd53cb4d53e84decda
    ) else if "%CUDA_VERSION%"=="12.1" (
        curl -O https://download.pytorch.org/whl/cu121/torch-2.1.1%%2Bcu121-cp310-cp310-win_amd64.whl#sha256=44d1f28af7bd2c51b633175e9b99465f88ced80038f0cad57f25794f994637d2
        curl -O https://download.pytorch.org/whl/cu121/torchvision-0.16.1%%2Bcu121-cp310-cp310-win_amd64.whl#sha256=356bc7d77982e54714f987e6d5df522429e7ca1eca65d3adda0b357693e174df
    )
) else if "%PYTHON_VERSION%"=="3.11" (
    if "%CUDA_VERSION%"=="11.7" (
        curl -O https://download.pytorch.org/whl/cu117/torch-2.0.1%%2Bcu117-cp311-cp311-win_amd64.whl#sha256=a77ba4f4b13c8b6c2c863b84a98dde2ddf1feaad5f25700d41cf3236e11d2ee8
        curl -O https://download.pytorch.org/whl/cu117/torchvision-0.15.2%%2Bcu117-cp311-cp311-win_amd64.whl#sha256=79c144a2369dbb50feefad5822a3c2357dfd659c774c06970db399df8c74eede
    ) else if "%CUDA_VERSION%"=="11.8" (
        curl -O https://download.pytorch.org/whl/cu118/torch-2.1.1%%2Bcu118-cp311-cp311-win_amd64.whl#sha256=d99be44487d3ed0f7e6ef5d6689a37fb4a2f2821a9e7b59e7e04002a876a667a
        curl -O https://download.pytorch.org/whl/cu118/torchvision-0.16.1%%2Bcu118-cp311-cp311-win_amd64.whl#sha256=bfd85e941d286c09c49df972db156dbab565371be4add105f290dbc6e8029b20
    ) else if "%CUDA_VERSION%"=="12.1" (
        curl -O https://download.pytorch.org/whl/cu121/torch-2.1.1%%2Bcu121-cp311-cp311-win_amd64.whl#sha256=cbc1b55879aca47584172a885c35677073110311cdbba3589e80938b15bbc8ad
        curl -O https://download.pytorch.org/whl/cu121/torchvision-0.16.1%%2Bcu121-cp311-cp311-win_amd64.whl#sha256=c7e4c810ba1875278aa2adeeee292ee6cb9d6c17dc00f3dd5ffdf70c5608e8fd
    )
)


echo 正在安装torch torchvision --cuda...
pip install --upgrade --force-reinstall *.whl
echo 正在删除whl...
del *.whl

pause