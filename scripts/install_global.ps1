$ErrorActionPreference = "Stop"
py -m pip install --upgrade -r requirements-global.txt
Write-Host "Global plotting package installed. No virtual environment was created."
