# setup.ps1
New-Item -ItemType Directory -Path .\app\api -Force
New-Item -ItemType Directory -Path .\app\services -Force
New-Item -ItemType Directory -Path .\models -Force
New-Item -ItemType Directory -Path .\data -Force

New-Item -ItemType File -Path .\app\main.py -Force
New-Item -ItemType File -Path .\app\api\llama_router.py -Force
New-Item -ItemType File -Path .\app\services\llama_model.py -Force
