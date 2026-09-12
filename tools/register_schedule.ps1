# Register a daily scheduled task to run the scraper at 22:00 local time
$taskName = "news_analysis_scrape"
$scriptPath = Join-Path $PSScriptRoot "scraper.py"
$python = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"
$feeds = Join-Path $PSScriptRoot "feeds.json"
$time = "22:00"

# Build the action. Use quoted paths.
$action = "`"$python`" `"$scriptPath`" --feeds `"$feeds`""

# Create or replace scheduled task
schtasks /Create /SC DAILY /TN $taskName /TR $action /ST $time /F | Out-Null
Write-Output "Scheduled task '$taskName' set to run daily at $time (local time)."
