$splitFiles = @(
    "out/rotor_sweep.usda",
    "out/streamlines_anim.usda",
    "out_outputs/predictions.npz"
)

foreach ($outputPath in $splitFiles) {
    $directory = Split-Path $outputPath
    $name = Split-Path $outputPath -Leaf
    $parts = Get-ChildItem -LiteralPath $directory -Filter "$name.part*" | Sort-Object Name

    if ($parts.Count -eq 0) {
        Write-Warning "No parts found for $outputPath"
        continue
    }

    $out = [System.IO.File]::Open($outputPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        foreach ($part in $parts) {
            $in = [System.IO.File]::OpenRead($part.FullName)
            try {
                $in.CopyTo($out)
            }
            finally {
                $in.Dispose()
            }
        }
    }
    finally {
        $out.Dispose()
    }

    Write-Host "Rebuilt $outputPath from $($parts.Count) parts."
}
