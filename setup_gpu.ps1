$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$GpuPython = Join-Path $ProjectRoot ".venv_gpu\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $GpuPython)) {
    & py -3.12 -m venv (Join-Path $ProjectRoot ".venv_gpu")
    if ($LASTEXITCODE -ne 0) { throw "GPU環境の作成に失敗しました。" }
}
& $GpuPython -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA版PyTorchのインストールに失敗しました。" }
& $GpuPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") pyinstaller
if ($LASTEXITCODE -ne 0) { throw "依存パッケージのインストールに失敗しました。" }
& $GpuPython -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable: check NVIDIA driver"; x=torch.ones(16,device="cuda"); print(torch.cuda.get_device_name(0), torch.__version__, x.sum().item())'
if ($LASTEXITCODE -ne 0) { throw "GPUの動作確認に失敗しました。" }
Write-Host "GPU環境を準備しました。build_gpu_exe.ps1 で配布版を作成できます。"
