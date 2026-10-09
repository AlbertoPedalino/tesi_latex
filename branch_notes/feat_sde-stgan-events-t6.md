# Branch: feat/sde-stgan-events-t6

## 1. Scopo dello step

- Rendere le analisi regionali degli eventi utilizzabili con le decisioni binarie già esportate da STGAN.
- Posizione: ramo di analisi successiva alla previsione, separato dallo sviluppo del detector.
- Revisione analizzata: `bdfff424`, riferimenti locale e `origin` concordi nella copia consultata il 10 settembre 2026.
- Predecessore effettivo: stato `4233cff6` della linea `feat/sde-direct-multihorizon-t1-t6`, genitore diretto del commit del ramo. Confronto `4233cff6..bdfff424`.
- Differenza: alle analisi della coda continua degli score si aggiungono aggregazioni che rispettano il flag salvato dal detector.
- Il nome del ramo richiama t+6, ma il solo delta Python non impone sei ore: la selezione concreta dell'orizzonte da parte di file esclusi dall'analisi non viene assunta.
- Periodo di selezione dei branch: 10 agosto–10 settembre 2026.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/reporting/anomaly_extremes.py` | Aggregazione degli eventi | Aggiunge `_boolean_flags`, `regional_flag_series` e `rank_flagged_days`; generalizza il modulo a detector puntuali. |

Il delta comprende anche un file di test Python; non introduce altre modifiche metodologiche e non è stato eseguito. Il codice del modello e del training non cambia rispetto al predecessore.

## 3. Dati e preprocessing

- Input del nuovo percorso: record con `location`, `timestamp`, `anomaly_score` e `is_anomaly`.
- Timestamp convertiti in UTC senza timezone nella tabella risultante.
- I flag ammettono booleani e rappresentazioni testuali true/false o 1/0; valori non riconosciuti generano un errore.
- Le righe prive di timestamp o score interpretabili vengono eliminate dal percorso di aggregazione.
- Nessun nuovo target supervisionato o tensore neurale: l'unità di elaborazione è una riga località–timestamp.
- Il nuovo aggregatore conta le righe per timestamp; non implementa una deduplicazione delle località. La lettura come numero di siti richiede quindi unicità delle coordinate in ingresso.

## 4. Modello o algoritmo

- `regional_flag_series` legge a blocchi e accumula, per timestamp, numero di record, numero di flag anomali, massimo e somma degli score.
- La quota regionale è `n_extreme/n_scored`; lo score medio è la somma degli score divisa per il numero di record.
- Non viene stimata una nuova soglia puntuale: il criterio di normalità è quello del campo `is_anomaly`.
- `rank_flagged_days` aggrega queste quantità per giorno. Un'ora è attiva se contiene almeno un flag anomalo.
- L'ordinamento giornaliero è decrescente per numero di anomalie, poi quota di anomalie, poi numero di ore attive.
- La media giornaliera degli score è pesata dai conteggi delle ore, non è una media semplice delle medie orarie.
- Gli helper ereditati per gli episodi possono usare la stessa struttura di tabella e applicare ulteriori criteri regionali: tali criteri vanno distinti dalla soglia puntuale del detector.
- Non sono aggiunti training, loss o inferenza neurale.

## 5. Elementi nuovi introdotti

- Un percorso di analisi fondato direttamente sulle decisioni del detector.
- Ranking dei giorni tramite frequenza e distribuzione temporale dei flag.
- Compatibilità strutturale con gli helper di episodi già presenti.
- Le funzioni esplorative basate sulla coda dello score restano disponibili; non sono sostituite.

## 6. Informazioni utili per la tesi

### Metodi

- Distinzione fra score continuo e decisione binaria salvata.
- Denominatore della quota regionale e requisito di unicità dei record.
- Aggregazione oraria/giornaliera e ordinamento degli eventi.
- Separazione fra analisi degli eventi e addestramento del previsore.

### Related Work

- Aggregazione spazio-temporale di anomalie puntuali.
- Rilevamento di episodi regionali da frequenze di superamento.
- Distinzione fra intensità dello score e prevalenza delle anomalie.

## 7. Aspetti da chiarire

- Il valore e il metodo della soglia STGAN non sono definiti dal nuovo aggregatore: dipendono dagli score forniti.
- Il solo delta Python non dimostra quale previsione a sei ore sia stata analizzata.
- La corrispondenza dei giorni selezionati con eventi fisici richiede evidenza esterna.
- Distribuzione dei flag, sovrapposizione degli errori e qualità delle previsioni richiedono risultati, esclusi da questa lettura.
- I conteggi grezzi favoriscono giorni con maggiore copertura: la copertura va riportata nel confronto sperimentale.

## 8. Sintesi

- Analisi regionale basata sui flag STGAN già disponibili.
- Nuove quote orarie e graduatorie giornaliere.
- Modello e training ereditati senza modifiche.
- L'uso effettivo dell'orizzonte t+6 non è attestato dal solo delta Python.
