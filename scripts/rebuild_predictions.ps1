param(
    [string]$OutputPath = "out_outputs/predictions.npz"
)

$parts = Get-ChildItem -LiteralPath (Split-Path $OutputPath) -Filter "$(Split-Path $OutputPath -Leaf).part*" |
    Sort-Object Name

if ($parts.Count -eq 0) {
    throw "No prediction parts found for $OutputPath"
}

$out = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
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

Write-Host "Rebuilt $OutputPath from $($parts.Count) parts."
