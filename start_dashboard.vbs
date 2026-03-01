'' Launches the dashboard silently (no console window)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c C:\Users\batbo\news-price-impact\start_dashboard.bat", 0, False
