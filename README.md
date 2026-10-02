# 🎙️ STT Italiano — Whisper locale (Mac · Windows · Linux)

App web **100% locale** per trascrivere audio/video in italiano. Niente Docker, niente cloud:
il motore Whisper viene scelto automaticamente in base alla piattaforma:

| Piattaforma | Backend | Accelerazione |
|---|---|---|
| macOS Apple Silicon (M1/M2/M3/M4) | **mlx-whisper** | GPU Metal (framework MLX di Apple) |
| Windows / Linux / Mac Intel | **faster-whisper** (CTranslate2) | CPU (int8); GPU NVIDIA CUDA se presente |

Su Apple Silicon si aggiunge **parakeet-mlx** (NVIDIA Parakeet v3, il default): non è un
modello Whisper ma usa la stessa GPU Metal, ed è multilingue con auto-rilevamento della lingua.

## Quickstart — macOS / Linux

```bash
# Opzione A — un comando (crea venv, installa, avvia, apre il browser):
./run.sh

# Opzione B — manuale (Python 3.10+):
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 9753
```

## Quickstart — Windows

```bat
REM Opzione A — doppio click su run.bat (crea venv, installa, avvia, apre il browser)
REM             oppure da Prompt dei comandi:
run.bat

REM Opzione B — manuale (Python 3.10+ da python.org, con "Add python.exe to PATH"):
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 9753
```

`pip install -r requirements.txt` installa automaticamente il backend giusto
(`mlx-whisper` solo su Apple Silicon, `faster-whisper` altrove) grazie ai
[marcatori di piattaforma](https://peps.python.org/pep-0508/) — nessun passo manuale.

Nessun prerequisito di sistema: la decodifica audio (mp3/ogg/mp4/... -> PCM per Whisper)
avviene via **PyAV**, che include le librerie ffmpeg dentro la wheel pip.
I modelli Whisper, infatti, non leggono file: accettano solo array numerici PCM 16kHz mono.

Apri **http://localhost:9753** → clicca "⬇ Scarica modello" sulle card che ti interessano
(il download avviene una sola volta, cache in `~/.cache/huggingface` su macOS/Linux,
`%USERPROFILE%\.cache\huggingface` su Windows) → seleziona uno o più modelli
→ trascina i file (fino a 10 in una volta) → Trascrivi.

## Modelli (italiano)

| Modello | Params | RAM | Qualità IT | Quando usarlo |
|---|---|---|---|---|
| **Parakeet v3** 🍎 ⭐ default | 600M | ~1.5 GB | ★★★★ | Uso quotidiano: modello di NVIDIA ma gira sulla GPU Apple via MLX (niente CUDA); fulmineo, multilingue (IT ✓), miglior equilibrio fedeltà/leggibilità nei nostri test comparativi; su audio difficili large-v3 resta superiore |
| **large-v3-turbo** | 809M | ~2 GB | ★★★★ | Qualità ~large-v3 a ~8x la velocità; è il default su Windows/Linux (dove Parakeet non gira) |
| large-v3 | 1.55B | ~4 GB | ★★★★★ | Audio difficili: rumore, accenti, sovrapposizioni |
| medium | 769M | ~2 GB | ★★★ | Bozze di qualità su audio pulito |
| small | 244M | ~1 GB | ★★ | Note vocali veloci |
| tiny | 39M | <1 GB | ★ | Solo test |

🍎 = visibile **solo su Apple Silicon** (gira via parakeet-mlx; l'API nasconde
automaticamente i modelli non compatibili con la piattaforma).

Con 16+ GB di RAM puoi tenere scaricati tutti i modelli senza pensieri.
Su CPU (Windows/Linux senza GPU NVIDIA) la velocità dipende dal processore:
per l'uso quotidiano conviene large-v3-turbo o small.

## Audio lunghi (Parakeet) e errore `metal::malloc`

Se trascrivi un file lungo con **Parakeet v3** su Mac puoi incontrare un errore tipo:

```text
[metal::malloc] Attempting to allocate 72721908896 bytes which is greater than
the maximum allowed buffer size of 30150672384 bytes.
```

Non è un problema di dimensione del file (un'ora di WhatsApp sono pochi MB):
è la **durata**. L'encoder di Parakeet usa attention a posizione relativa
**globale** (`rel_pos`, `att_context_size = [-1, -1]`), quindi il buffer di
attenzione cresce col **quadrato** della durata: circa `10.000 × secondi²` byte
in `float32`. Il file dell'esempio dura ~45 min → un singolo buffer da ~68 GB,
oltre il massimo che Metal concede per allocazione (nel nostro test ~28 GiB
su un Mac con 48 GB di RAM).

L'app lo evita **automaticamente spezzando l'audio in blocchi** (chunking):
i chunk vengono trascritti separatamente e poi ricuciti sull'overlap, con
timestamp assoluti (gli `.srt` restano corretti). Così la memoria dipende
**dalla dimensione del chunk, non dalla durata del file**:

| Durata audio | Senza chunking (buffer di attenzione) | Con chunking attivo (picco misurato) |
|---|---|---|
| 5 min | ~1 GB | ~3 GB (chunk 120s) |
| 15 min | ~7,5 GB | ~3–6 GB, **costante** |
| 30 min | ~30 GB → **oltre il limite Metal** | ~3–6 GB, **costante** |
| 45 min | ~68 GB → **errore `metal::malloc`** | ~6,4 GB (chunk 300s, misurato) |
| 3 ore | ~1.600 GB → errore | ~6,4 GB, **costante** |

Misurazioni reali su un file di 45 min (Mac Apple Silicon con 48 GB di RAM),
stesso testo, chunk diversi:

| `PARAKEET_CHUNK_SECONDS` | picco memoria GPU | tempo |
|---|---|---|
| `120` | 3,2 GB | 30 s |
| `300` (default) | 6,4 GB | 33 s |
| `600` | 17,5 GB | 44 s |

La durata **totale** del file non cambia il picco: cambia solo il tempo.

Due variabili d'ambiente regolano il comportamento (default: 300s e 15s):

```bash
# chunk più piccoli = meno memoria, ma più giunzioni da ricucire
PARAKEET_CHUNK_SECONDS=120 PARAKEET_OVERLAP_SECONDS=10 uvicorn app:app --host 127.0.0.1 --port 9753
# oppure in run.sh / run.bat, prima di avviare il server
```

- `PARAKEET_CHUNK_SECONDS` — lunghezza del blocco in secondi. Default `300`.
  Valori sensati: `120` su Mac con 16 GB di RAM, `300` (default) in generale,
  `600` solo con 32+ GB. Massimo accettato: `900` (oltre viene ridotto
  automaticamente, con un avviso su terminale, perché il picco risale).
  `0` disabilita il chunking (sconsigliato: torni esposto all'errore qui sopra).
- `PARAKEET_OVERLAP_SECONDS` — secondi di sovrapposizione tra blocchi adiacenti
  (servono a non perdere parole al confine). Default `15`, viene comunque
  limitata a metà del chunk.

I modelli **Whisper** (`large-v3-turbo`, `large-v3`, ...) non hanno questo
limite: `mlx-whisper` e `faster-whisper` lavorano sempre su finestre da 30 s,
quindi vanno bene a qualsiasi durata. In caso di dubbio, sono la scelta sicura
per file molto lunghi.

## Trascrizione multi-modello (cascata)

Puoi selezionare **più modelli insieme**: vengono eseguiti in sequenza sullo stesso
file, nell'ordine in cui li hai cliccati (il badge 1°, 2°, 3°… sulla card mostra
l'ordine di esecuzione). Se un modello fallisce, gli altri continuano comunque e
l'errore finisce nella sua sezione del risultato.

Il bottone **"⬇ Tutti i modelli (.txt)"** scarica un unico file `.multi.txt` con
tutte le trascrizioni, una sezione per modello:

```text
TRASCRIZIONI MULTIPLE — audio.ogg
Modelli eseguiti in sequenza: 2

################################################################
MODELLO 1/2 — Parakeet v3 [parakeet-v3]
Lingua: it · Tempo: 12.3s
################################################################

<testo completo del modello 1>

################################################################
MODELLO 2/2 — Large v3 Turbo [large-v3-turbo]
Lingua: it · Tempo: 31.0s
################################################################

<testo completo del modello 2>
```

È il formato ideale da dare in pasto a un LLM per confrontare le trascrizioni,
correggere la terminologia o ricavarne una versione pulita.

## Upload multi-file (batch)

Puoi caricare **fino a 10 file in una volta** (trascina o clicca): tutti i file
vengono trascritti con gli stessi modelli e la stessa lingua. Ogni file diventa
un job indipendente con la propria scheda nei risultati: stato a colpo d'occhio
(⏳ in coda, 🔄 in lavorazione, ✅ completato, ❌ errore) e pulsanti di download
singoli (.txt, .srt, oppure file unico combinato se hai scelto più modelli).
I job vengono eseguiti in coda, un file alla volta, per non sovraccaricare CPU/GPU.

## Rilevamento server spento

Se chiudi il terminale con il server, l'interfaccia se ne accorge entro pochi
secondi e mostra un banner rosso in alto (heartbeat su `/api/health`, più un
controllo al click di ogni bottone): la logica è nel browser, quindi funziona
identica su Windows e macOS/Linux. Quando riavvii il server (`run.bat` /
`run.sh`) il banner sparisce da solo e riprende il monitoraggio dei job in corso.

## Formati

`ogg · wav · mp3 · mp4 · m4a · webm · flac · aac · opus · mov · mkv` — la conversione
è automatica via PyAV (dagli mp4/mov/mkv viene estratta la traccia audio).

Export risultati: **copia negli appunti**, **.txt**, **.srt** (sottotitoli con timestamp)
per il modello visualizzato nel tab attivo; con più modelli selezionati c'è anche
l'**export combinato** `.multi.txt` (vedi sopra).

## Struttura

```
app.py            # FastAPI: API REST, upload singolo/batch (max 10), job manager, export, health
stt_engine.py     # registry modelli + download HF + backend (mlx-whisper / faster-whisper / parakeet-mlx)
static/index.html # UI (single-file, zero build)
run.sh            # avvio one-command (macOS/Linux)
run.bat           # avvio one-command (Windows)
```

API docs automatiche su **http://localhost:9753/docs**.

## Sicurezza e privacy

- Il server ascolta solo su **127.0.0.1**: non raggiungibile dalla rete locale.
  **Non** avviarlo con `--host 0.0.0.0`: non c'è autenticazione.
- Upload limitato a **2 GB** per file (≈3 ore di audio; variabile `MAX_UPLOAD_MB` per cambiarlo),
  batch massimo di 10 file per richiesta (`MAX_BATCH_FILES`),
  scritto su disco in streaming e cancellato subito dopo la trascrizione.
- Estensioni, lingue e id modello validati lato server; nomi file sanitizzati negli export.
- In memoria restano al massimo gli ultimi 100 job; gli upload orfani sono eliminati all'avvio.
- Unico traffico di rete: il download dei modelli da HuggingFace. Tutto il resto è 100% locale.
- Audit dipendenze (opzionale): `pip install pip-audit && pip-audit`.

## Troubleshooting

- **Primo download lento** → normale: parakeet ≈ 1.3 GB, turbo ≈ 1.6 GB, large-v3 ≈ 3.1 GB (una tantum)
- **Porta occupata** → cambia porta: `uvicorn app:app --port 8001`
- **`bad interpreter` dopo aver spostato/rinominato la cartella del progetto** → un venv non
  è portabile tra percorsi: `run.sh` e `run.bat` rilevano il venv rotto e lo ricreano da soli
- **Errore di decodifica su un file** → il container potrebbe essere corrotto: ri-esportalo
- **Windows: `py`/`python` non riconosciuto** → installa Python 3.10+ da
  [python.org](https://www.python.org/downloads/) spuntando **"Add python.exe to PATH"**
- **Windows con GPU NVIDIA** → faster-whisper usa CUDA automaticamente se le librerie
  cuBLAS/cuDNN sono disponibili; altrimenti ripiega sulla CPU senza errori
- **Trascrizione lenta su Windows senza GPU** → usa un modello più piccolo (small/turbo):
  large-v3 su CPU int8 è preciso ma lento

## Alternative (CLI)

```bash
# whisper.cpp: C++ nativo (Metal su Mac, CUDA/CPU su Windows), zero overhead Python
brew install whisper-cpp    # macOS
whisper-cli -m ggml-large-v3-turbo.bin -l it -f audio.wav
```

## Licenza

[MIT](LICENSE). I modelli scaricati da HuggingFace hanno licenze proprie:
Whisper è MIT (OpenAI), Parakeet TDT è CC-BY-4.0 (NVIDIA).
