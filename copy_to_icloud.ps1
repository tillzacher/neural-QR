# Define source and destination paths
$sourceFolder = Resolve-Path ".\output_images"
$destinationFolder = "C:\Users\Napi\iCloudDrive\generated_qr_codes"

# Ensure the source folder exists
if (-not (Test-Path $sourceFolder)) {
    Write-Error "Source folder does not exist: $sourceFolder"
    exit
}

# Ensure the destination folder exists, create if it doesn't
if (-not (Test-Path $destinationFolder)) {
    New-Item -ItemType Directory -Path $destinationFolder | Out-Null
    Write-Host "Created destination folder: $destinationFolder"
}

# Function to get the last write time of the most recently modified file
function Get-LastWriteTime($folder) {
    return (Get-ChildItem $folder -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
}

# Get the last write time of the most recently copied file (if exists)
$lastCopyFile = Join-Path $destinationFolder "last_copy_time.txt"
if (Test-Path $lastCopyFile) {
    $lastCopyTime = Get-Content $lastCopyFile | Get-Date
} else {
    $lastCopyTime = [DateTime]::MinValue
}

# Copy files
$copiedFiles = @()
Get-ChildItem $sourceFolder -Recurse | ForEach-Object {
    if ($_.LastWriteTime -gt $lastCopyTime) {
        $relativePath = $_.FullName.Substring($sourceFolder.Length).TrimStart('\')
        $destPath = Join-Path $destinationFolder $relativePath
        
        if ($_.PSIsContainer) {
            # If it's a directory, just create it
            if (-not (Test-Path $destPath)) {
                New-Item -ItemType Directory -Path $destPath | Out-Null
            }
        } else {
            # If it's a file, copy it
            $destDir = Split-Path $destPath -Parent
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir | Out-Null
            }
            Copy-Item $_.FullName -Destination $destPath -Force
            $copiedFiles += $_.FullName
        }
    }
}

# Update the last copy time
$currentTime = Get-Date
$currentTime.ToString("o") | Out-File $lastCopyFile -Force

# Report results
if ($copiedFiles.Count -gt 0) {
    Write-Host "Copied $($copiedFiles.Count) files:"
    $copiedFiles | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "No new files to copy."
}
