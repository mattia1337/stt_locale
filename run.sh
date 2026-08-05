#!/bin/bash
# STT Italiano — avvio con UN comando: ./run.sh
# Crea il venv se manca (o se è rotto), installa le dipendenze,
# avvia il server e apre il browser.
cd "$(dirname "$0")"

# Un venv spostato/copiato da un'altra cartella NON è portabile: i suoi script
# (pip, uvicorn) hanno shebang assoluti al vecchio percorso -> "bad interpreter".
# Lo rilevo testando pip e lo ricreo da zero.
if [ -d .venv ] && ! .venv/bin/pip --version >/dev/null 2>&1; then
  echo "==> Il venv esistente non è valido (progetto spostato?): lo ricreo..."
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  echo "==> Creo l'ambiente virtuale..."
  python3 -m venv .venv
fi

echo "==> Controllo dipendenze..."
.venv/bin/pip install -q -r requirements.txt || {
  echo "ERRORE: installazione dipendenze fallita." >&2
  exit 1
}

(sleep 2 && open http://localhost:9753) &

echo "==> Server su http://localhost:9753  (Ctrl+C per fermare)"
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 9753