param([switch]$SkipBuild)
$ErrorActionPreference = "Stop"

$ProjectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DistRoot = Join-Path $ProjectRoot "dist"
$PackageRoot = Join-Path $ProjectRoot "配布用"
$PackageDir = Join-Path $PackageRoot "MidnaUdon EventPhotoSorter"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "先に setup.bat を実行してください。"
}

if ($SkipBuild) {
    & $Python (Join-Path $ProjectRoot "tools\build_release.py") --verify-only
} else {
    & $Python (Join-Path $ProjectRoot "tools\build_release.py")
}
if ($LASTEXITCODE -ne 0) {
    throw "EXEのビルドに失敗しました。"
}

$ResolvedPackageRoot = [System.IO.Path]::GetFullPath($PackageRoot)
$ResolvedPackageDir = [System.IO.Path]::GetFullPath($PackageDir)
if (-not $ResolvedPackageDir.StartsWith($ResolvedPackageRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "配布先がプロジェクト内ではありません: $ResolvedPackageDir"
}
# Update in place: deployed folders contain user photos, credentials and history.
# Never recreate the whole package by recursively deleting that data.

New-Item -ItemType Directory -Path (Join-Path $ResolvedPackageDir "作業フォルダ") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ResolvedPackageDir "振り分け後") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ResolvedPackageDir "アプリデータ") -Force | Out-Null

$ExistingExe = Join-Path $ResolvedPackageDir "MidnaUdon EventPhotoSorter.exe"
if (Test-Path -LiteralPath $ExistingExe -PathType Leaf) {
    $BackupDir = Join-Path $ProjectRoot "artifacts\exe_backups"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    $BackupName = "EventPhotoSorter_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".exe"
    Copy-Item -LiteralPath $ExistingExe -Destination (Join-Path $BackupDir $BackupName)
}
Copy-Item -LiteralPath (Join-Path $DistRoot "MidnaUdon EventPhotoSorter.exe") -Destination $ResolvedPackageDir -Force
if (-not (Test-Path -LiteralPath (Join-Path $ResolvedPackageDir ".env"))) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot ".env.example") -Destination (Join-Path $ResolvedPackageDir ".env")
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\Readme.txt") -Destination (Join-Path $ResolvedPackageDir "Readme.txt")
& $Python (Join-Path $ProjectRoot "tools\collect_licenses.py")
if ($LASTEXITCODE -ne 0) { throw "ライセンス原文の収集に失敗しました。" }
foreach ($Document in @("LICENSE", "THIRD_PARTY_NOTICES.md", "ASSET_NOTICE.md", "RUNTIME_TERMS.txt")) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Document) -Destination $ResolvedPackageDir -Force
}
Copy-Item -LiteralPath (Join-Path $ProjectRoot "licenses") -Destination $ResolvedPackageDir -Recurse -Force

# Distributable packages start with an empty reference folder for each organizer.
# Existing references in an updated package are preserved.
New-Item -ItemType Directory -Path (Join-Path $ResolvedPackageDir "参考画像") -Force | Out-Null

$ModelCache = Join-Path $ProjectRoot ".model_cache"
& $Python -m photo_sorter.package_models --sources $ModelCache (Join-Path $ProjectRoot "アプリデータ\model_cache") (Join-Path $ProjectRoot "artifacts\gpu_fresh_cache") `
    --destination (Join-Path $ResolvedPackageDir "アプリデータ\model_cache")
if ($LASTEXITCODE -ne 0) { throw "配布モデルの検証・コピーに失敗しました。" }
Write-Host "手元の利用用フォルダを更新しました（写真やキーを保持しています。公開には使わないでください）:"
Write-Host $ResolvedPackageDir
& $Python (Join-Path $ProjectRoot "tools\export_release.py")
if ($LASTEXITCODE -ne 0) { throw "公開用ZIPの作成・検証に失敗しました。" }
& $Python (Join-Path $ProjectRoot "tools\export_source.py") --init-git
if ($LASTEXITCODE -ne 0) { throw "履歴を引き継がない公開用ソースの作成に失敗しました。" }
