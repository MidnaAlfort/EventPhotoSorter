$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$GpuPython = Join-Path $ProjectRoot ".venv_gpu\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $GpuPython)) { throw "先に setup_gpu.ps1 を実行してください。" }
$PackageDir = Join-Path $ProjectRoot "配布用\MidnaUdon EventPhotoSorter"
if (-not (Test-Path -LiteralPath (Join-Path $PackageDir "MidnaUdon EventPhotoSorter.exe"))) {
    throw "先に build_exe.ps1 で通常版を作成してください。"
}
& $GpuPython -m PyInstaller --noconfirm --clean --onedir --windowed `
    --name "MidnaUdon EventPhotoSorter_GPU" `
    --icon (Join-Path $ProjectRoot "assets\app_icon.ico") `
    --add-data "$(Join-Path $ProjectRoot 'assets\app_icon.png');assets" `
    --add-data "$(Join-Path $ProjectRoot 'assets\app_icon.ico');assets" `
    --add-data "$(Join-Path $ProjectRoot 'RUNTIME_TERMS.txt');." `
    --collect-submodules "transformers.models.dinov2" `
    --collect-submodules "transformers.models.grounding_dino" `
    --collect-submodules "transformers.models.swin" `
    --collect-submodules "transformers.models.bert" `
    --distpath (Join-Path $ProjectRoot "dist") `
    --workpath (Join-Path $ProjectRoot "build_gpu") `
    --specpath (Join-Path $ProjectRoot "build_gpu") (Join-Path $ProjectRoot "app.py")
if ($LASTEXITCODE -ne 0) { throw "GPU版のビルドに失敗しました。" }
$GpuDestination = Join-Path $PackageDir "GPU版"
New-Item -ItemType Directory -Path $GpuDestination -Force | Out-Null
# Only this application-owned subfolder is updated. Photos, settings and history live in its parent.
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "dist\MidnaUdon EventPhotoSorter_GPU") | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $GpuDestination -Recurse -Force
}
Write-Host "GPU版を作成しました: $GpuDestination\MidnaUdon EventPhotoSorter_GPU.exe"
