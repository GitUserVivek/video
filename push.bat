@echo off

set "MESSAGE=%~1"

if "%MESSAGE%"=="" set "MESSAGE=updated"

for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set "DATE=%%c-%%a-%%b"
for /f "tokens=1-2 delims=:." %%a in ("%time%") do set "TIME=%%a:%%b"

git add .
git commit -m "%MESSAGE% %DATE% %TIME%"
git push
