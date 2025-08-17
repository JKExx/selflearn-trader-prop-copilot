$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
  Write-Host "▶ Creating .venv …"
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip | Out-Null
pip install -r requirements.txt | Out-Null

$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
Write-Host "▶ Starting app on http://localhost:8501"
python -m streamlit run app/ui/st_app.py --server.address 0.0.0.0 --server.port 8501
