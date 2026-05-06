Write-Host "Starting API Backend on Port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"& .\.venv\Scripts\Activate.ps1; python -m uvicorn api:app --reload --port 8000`""

Write-Host "Starting Next.js Frontend on Port 3000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd frontend; npm run dev`""

Write-Host "Both servers are starting! The frontend will be available at http://localhost:3000" -ForegroundColor Yellow