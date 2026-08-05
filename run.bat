@echo off
REM STT Italiano - avvio con UN comando: run.bat
REM Crea il venv se manca (o se e' rotto), installa le dipendenze,
REM avvia il server e apre il browser.
cd /d "%~dp0"

REM Un venv spostato/copiato da un'altra cartella NON e' portabile:
REM gli exe in Scripts puntano al vecchio percorso. Lo rilevo testando pip.
if exist .venv (
  .venv\Scripts\pip.exe --version >nul 2>&1
  if errorlevel 1 (
    echo ==^> Il venv esistente non e' valido ^(progetto spostato?^): lo ricreo...
    rmdir /s /q .venv
  )
)

if not exist .venv (
  echo ==^> Creo l'ambiente virtuale...
  py -m venv .venv 2>nul
  if errorlevel 1 python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo ERRORE: Python non trovato. Installa Python 3.10+ da https://www.python.org/downloads/
  echo ^(durante l'installazione spunta "Add python.exe to PATH"^) e riprova.
  pause
  exit /b 1
)

echo ==^> Controllo dipendenze...
.venv\Scripts\pip.exe install -q -r requirements.txt
if errorlevel 1 (
  echo ERRORE: installazione dipendenze fallita. Controlla la connessione e riprova.
  pause
  exit /b 1
)

REM Apre il browser dopo 2 secondi, in background, senza bloccare l'avvio del server
start "" /min cmd /c "timeout /t 2 /nobreak >nul & start "" http://localhost:9753"

echo ==^> Server su http://localhost:9753  (Ctrl+C per fermare)
.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 9753