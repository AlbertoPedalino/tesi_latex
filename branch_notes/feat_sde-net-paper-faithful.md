# Branch: feat/sde-net-paper-faithful

## 1. Scopo dello step

- Funzione principale del branch: integrare nel previsore spazio-temporale su grafo una dinamica neurale stocastica esplicita, con drift appreso, diffusione scalare appresa, integrazione Euler–Maruyama e uscita probabilistica eteroschedastica.
- Posizione nel percorso complessivo della tesi: costituisce lo step di forecasting probabilistico spazio-temporale usato per ottenere previsione puntuale, intervalli predittivi e separazione tra incertezza aleatoria ed epistemica sui nodi geografici PVGIS.
- La nota documenta esclusivamente lo stato corrente del codice Python versionato nel branch al commit `6fd0898e`.
- L'elemento metodologico distintivo dello step corrente è la combinazione tra grafo geografico, dinamica SDE nel latente, addestramento separato della diffusione e distribuzione gaussiana predittiva.

## 2. File di codice rilevanti

### `physiq_pv/data/pvgis_dataset.py`

- Responsabilità: caricamento degli anni PVGIS, controllo dell'allineamento temporale e geografico, costruzione delle feature, stima delle statistiche di normalizzazione e generazione delle finestre temporali.
- Contenuto metodologico attuale: `PVGISWindowDataset`, `build_year_raw`, `fit_normalization`, `assemble_feats` e `build_datasets` realizzano split cronologici di fit, validation e test; le finestre non attraversano il confine tra anni ed è disponibile un filtraggio opzionale delle finestre rare per un'ablation di training.

### `physiq_pv/data/pvgis_irradiance.py`

- Responsabilità: ricostruzione coerente della POA effettiva.
- Contenuto metodologico attuale: `with_effective_poa` calcola la POA come somma di irradianza diretta e diffusa sul piano inclinato, correggendo soltanto piccoli valori negativi attribuiti alla tolleranza numerica.

### `physiq_pv/data/pvgis_labels.py`

- Responsabilità: caricamento e associazione alle predizioni di etichette di anomalia o di evento.
- Contenuto metodologico attuale: le etichette sono metadati di valutazione e stratificazione e non vengono inserite nel tensore di input del modello.

### `physiq_pv/model/graph_builder.py`

- Responsabilità: costruzione del grafo geografico sparso.
- Contenuto metodologico attuale: usa distanza Haversine, archi entro un raggio massimo, collegamento al vicino più prossimo per nodi isolati, archi nei due versi, self-loop e prior gaussiano dipendente dalla distanza.

### `physiq_pv/model/sde_net.py`

- Responsabilità: componenti fondamentali della SDE neurale.
- Contenuto metodologico attuale: contiene `VectorDrift`, `ScalarDiffusion` e `PaperSDEBlock`; include inoltre `YearMSDSDENet` come composizione di riferimento e `diffusion_bce_loss` per la supervisione ID/pseudo-OOD della diffusione.

### `physiq_pv/model/st_gnn.py`

- Responsabilità: modello spazio-temporale complessivo.
- Contenuto metodologico attuale: `STGNN` combina encoder BiLSTM per nodo, pooling temporale con attenzione, layer GAT sparsi, blocco SDE e teste probabilistiche; `GATLayer` incorpora il prior geografico sommando il logaritmo del peso di distanza ai logit di attenzione appresi.

### `physiq_pv/training/losses.py`

- Responsabilità: obiettivo della testa probabilistica.
- Contenuto metodologico attuale: `gaussian_nll` implementa la negative log-likelihood gaussiana eteroschedastica; l'eventuale obiettivo ausiliario sull'irradianza è una MSE pesata.

### `physiq_pv/training/train_loop.py`

- Responsabilità: addestramento, validazione e selezione del checkpoint.
- Contenuto metodologico attuale: `train_model` alterna l'ottimizzazione del sottosistema di forecasting/drift e quella della sola diffusione; applica warm-up dell'ampiezza del rumore, clipping del gradiente, decadimento del learning rate ed early stopping su una metrica di validation diurna.

### `physiq_pv/training/uncertainty.py`

- Responsabilità: inferenza stocastica e quantificazione dell'incertezza.
- Contenuto metodologico attuale: `predict_sde` esegue più traiettorie Browniane; `_gaussian_mixture_quantile` calcola gli intervalli dalla miscela delle gaussiane prodotte dalle traiettorie; vengono separate varianza epistemica, aleatoria e totale.

### `physiq_pv/experiments/pvgis_stgnn_runner.py`

- Responsabilità: configurazione ed esecuzione riproducibile dell'intera pipeline.
- Contenuto metodologico attuale: `run_from_args` coordina split, preprocessing, grafo, training, inferenza e salvataggio delle predizioni per coppia località-timestamp; espone gli iperparametri SDE e calcola metriche puntuali e di intervallo, anche per sottoinsiemi diurni e categorie di valutazione.

### `physiq_pv/reporting/run_metrics.py`

- Responsabilità: aggregazione delle metriche dalle predizioni salvate.
- Contenuto metodologico attuale: calcola MAE, RMSE e metriche degli intervalli per l'intero insieme e per gli strati di valutazione disponibili, senza intervenire nel training.

## 3. Dati e preprocessing

- Input: file NetCDF/xarray annuali PVGIS con griglia oraria regolare, dimensioni temporali e di località coerenti tra anni, coordinate geografiche finite e variabili meteorologiche/solari.
- Feature del set completo: `temperature_2m`, `solar_irradiance_poa`, `wind_speed_10m`, `sin_elev`, `cos_elev`, `kt_poa`, `kt_poa_std_3h`, `dpoa_dt`, `direct_irradiance_tilted`, `diffuse_irradiance_tilted` e `pv_lag_pvgis`.
- Target: `pv_power_output` all'orizzonte configurato; il default del runner è una previsione a un'ora dopo una finestra di 24 ore.
- La POA viene ricostruita come diretta inclinata più diffusa inclinata. Sono inoltre calcolati geometria solare, POA clear-sky, indice di chiarezza limitato all'intervallo configurato, deviazione standard mobile a tre ore e differenza temporale della POA.
- Temperatura, POA, vento e variazione della POA sono standardizzati mediante statistiche globali stimate esclusivamente sugli anni di fit. La potenza PV è scalata per località usando il percentile 99 dei valori positivi diurni del fit; non è applicato clipping superiore per default.
- Lo split predefinito usa 2016–2018 come anni di training dichiarati e 2019 come test. L'anno più recente del gruppo di training è riservato alla validation; pertanto, con i default, le statistiche e i parametri sono stimati su 2016–2017 e la validation è il 2018.
- `PVGISWindowDataset` restituisce `x` con forma `(N, T, F)`, `y` con forma `(N,)` e l'indice temporale globale. Dopo il batching: `X` ha forma `(B, N, T, F)` e `Y` forma `(B, N)`, dove `N` è il numero di località, `T` la lunghezza della finestra e `F` il numero di feature selezionate.
- L'indice del target è `inizio_finestra + T + orizzonte - 1`; le finestre non attraversano anni diversi.
- Le etichette di anomalia/evento sono associate soltanto alle righe di output per analisi stratificate. Il training standard non usa tali etichette; il filtraggio dei periodi rari è una modalità opzionale e disattivata per default.

## 4. Modello o algoritmo

### Componenti e flusso dei dati

1. Per ogni nodo geografico, la sequenza `(T, F)` è elaborata dallo stesso encoder BiLSTM bidirezionale a due strati.
2. Un'attenzione temporale aggrega gli stati della BiLSTM; una proiezione con attivazione e normalizzazione produce il latente iniziale `x0` di forma `(B, N, D)`.
3. I layer GAT propagano informazione soltanto sugli archi geografici sparsi. Il prior gaussiano di distanza modifica i logit dell'attenzione, mentre i coefficienti restano appresi dai dati.
4. `PaperSDEBlock` evolve il latente secondo una SDE neurale. Il drift vettoriale è applicato a ogni nodo; la diffusione aggrega i nodi con una media e restituisce uno scalare per elemento del batch, condiviso tra nodi e dimensioni latenti.
5. L'integrazione usa Euler–Maruyama. Con i default, l'intervallo temporale astratto è suddiviso in quattro passi; durante il training e l'inferenza stocastica viene aggiunto un incremento Browniano, mentre la validation deterministica usa il solo drift.
6. La testa PV restituisce, per ogni nodo, media non negativa e deviazione standard positiva. Una testa ausiliaria opzionale stima anche una componente legata all'irradianza/indice clear-sky.

### Grafo geografico

- I nodi sono le località PVGIS, non le variabili meteorologiche.
- `edge_index` ha forma `(2, E)` e il vettore dei prior forma `(E,)`.
- Sono creati archi per distanze entro 20 km per default; i nodi senza vicini vengono connessi al nodo più vicino.
- Il peso geografico è `exp(-0.5 * (d / scala)^2)` con self-loop di peso uno; la scala è configurabile e vale 10 km nella configurazione della pipeline.
- Softmax e aggregazione del GAT operano sugli archi, evitando una matrice di attenzione geografica densa `N x N`.

### Architettura, input e output

- Input del modello: `X` di forma `(B, N, T, F)`, grafo sparso e prior degli archi.
- Uscita latente dopo encoder, GAT e SDE: `(B, N, D)`.
- Uscite principali: `pred_pv_mean` e `pred_pv_sigma`, entrambe `(B, N)`; la diffusione è uno scalare per elemento del batch. La testa ausiliaria, se attiva, produce una previsione per nodo.
- L'output tabellare è indicizzato da `timestamp` e `location`; riporta target, previsione, incertezze e limiti dell'intervallo nelle unità fisiche ricostruite.

### Funzione di perdita e training

- Forecasting: negative log-likelihood gaussiana eteroschedastica, con eventuale MSE ausiliaria pesata sull'uscita solare.
- Drift/backbone e diffusione usano due ottimizzatori SGD separati. Il primo aggiorna encoder temporale, GAT, drift e teste predittive.
- Nel passo della diffusione, un batch pseudo-OOD è ottenuto aggiungendo rumore gaussiano all'input. Le rappresentazioni sono staccate dal grafo dei gradienti e una BCE addestra la diffusione a produrre valori bassi per ID e alti per pseudo-OOD.
- L'ampiezza massima del termine stocastico passa dal valore iniziale a quello configurato dopo il warm-up. Il gradiente del sottosistema predittivo è limitato in norma.
- La validation è deterministica ed è eseguita sull'ultimo anno di training. Per default, checkpoint ed early stopping dipendono dall'RMSE diurna in unità fisiche.

### Inferenza e incertezza

- L'inferenza predefinita genera dieci traiettorie SDE indipendenti.
- La media predittiva è la media delle medie gaussiane delle traiettorie.
- La varianza epistemica è la varianza tra le medie delle traiettorie; la varianza aleatoria è la media delle varianze gaussiane; la varianza totale è la loro somma.
- L'intervallo predittivo principale è l'intervallo equal-tail della miscela gaussiana, ottenuto invertendo numericamente la CDF. L'intervallo gaussiano moment-matched è mantenuto come diagnostica secondaria.
- La valutazione comprende MAE/RMSE e, per gli intervalli, copertura empirica, ampiezza media, ampiezza normalizzata e una metrica congiunta di copertura e sharpness. Non sono ricavabili dal codice risultati numerici del modello.

## 5. Elementi nuovi introdotti

- `VectorDrift`: campo di drift neurale vettoriale nel latente.
- `ScalarDiffusion`: diffusione scalare dipendente dal grafo, ottenuta dal pooling dei nodi.
- `PaperSDEBlock`: simulazione discreta della dinamica tramite Euler–Maruyama.
- `YearMSDSDENet`: composizione SDE di riferimento presente nel modulo dedicato.
- `gaussian_nll`: obiettivo probabilistico per media e varianza dipendenti dall'input.
- `diffusion_bce_loss`: obiettivo binario ID/pseudo-OOD per la sola rete di diffusione.
- `predict_sde` e `_gaussian_mixture_quantile`: campionamento di traiettorie, decomposizione dell'incertezza e intervalli da miscela gaussiana.
- `PVGISWindowDataset` e funzioni di costruzione: pipeline esplicita con split cronologico e statistiche stimate soltanto sul fit.
- Prior geografico gaussiano integrato nei logit del GAT.
- Testa eteroschedastica e traiettorie Browniane della SDE per la previsione probabilistica.

## 6. Informazioni utili per la tesi

### Metodi

- Definizione del problema come forecasting probabilistico multivariato su un grafo di località geografiche.
- Split cronologico fit/validation/test e prevenzione del leakage nelle statistiche di preprocessing.
- Costruzione delle feature solari e meteorologiche e normalizzazione della potenza per località.
- Costruzione sparsa del grafo tramite distanza Haversine, raggio geografico e prior gaussiano.
- Encoder temporale BiLSTM con attenzione e propagazione spaziale GAT.
- Formulazione del blocco SDE, ruoli distinti di drift e diffusione e discretizzazione Euler–Maruyama.
- Alternanza tra obiettivo predittivo e classificazione ID/pseudo-OOD della diffusione.
- Regressione gaussiana eteroschedastica e decomposizione aleatoria/epistemica dell'incertezza.
- Costruzione degli intervalli predittivi dalla miscela delle distribuzioni ottenute dalle traiettorie SDE.
- Protocollo di validation diurna, early stopping e metriche puntuali/di intervallo.
- Uso delle etichette di anomalia soltanto per analisi post-hoc e ablation opzionale del training sui periodi normali.

### Related Work

- Neural stochastic differential equations.
- Schema numerico Euler–Maruyama.
- SDE-Net e apprendimento della diffusione mediante dati ID e pseudo-OOD.
- Regressione eteroschedastica con negative log-likelihood gaussiana.
- Separazione tra incertezza aleatoria ed epistemica mediante legge della varianza totale.
- Intervalli predittivi da miscele gaussiane.
- BiLSTM e attention pooling per serie temporali.
- Graph Attention Networks e message passing sparso.
- Grafi geografici basati su Haversine e kernel gaussiani di distanza.
- Modellazione clear-sky, indice di chiarezza e decomposizione dell'irradianza sul piano inclinato.

## 7. Aspetti da chiarire

- Il codice non esplicita la motivazione scientifica dei valori scelti per raggio del grafo, scala gaussiana, numero di passi SDE, warm-up, rumore pseudo-OOD e peso della perdita ausiliaria.
- Non è ricavabile dal solo codice la provenienza esatta dei file PVGIS, il criterio di selezione delle località o la semantica fisica completa di `pv_power_output`.
- Il numero effettivo di località e la configurazione finale di un esperimento dipendono dai file e dagli argomenti CLI usati a runtime.
- La validità empirica, la significatività delle differenze e la capacità di generalizzare non possono essere dedotte senza risultati sperimentali.
- La corrispondenza scientifica completa con il paper di riferimento richiede un confronto esterno con paper e repository originale; i nomi e i commenti nel codice non sono una prova sufficiente.
- Il rumore gaussiano usato per creare pseudo-OOD è un meccanismo di training/diagnostica; dal codice non si può stabilire quanto rappresenti anomalie fisiche reali.
- La scelta del detector e delle etichette associate a una specifica esecuzione non è determinabile dal modello, perché tali dati sono forniti separatamente alla fase di valutazione.
- La nota descrive il codice versionato al commit `6fd0898e`; eventuali modifiche locali non committate non rientrano nello stato documentato.

## 8. Sintesi

- Il branch contiene una SDE neurale integrata nello ST-GNN geografico.
- BiLSTM e GAT producono un latente spazio-temporale, poi evoluto mediante drift e diffusione appresi.
- La testa gaussiana eteroschedastica e le traiettorie Browniane forniscono previsione e decomposizione dell'incertezza.
- Preprocessing e split sono cronologici e le etichette di anomalia restano esterne al training standard.
- Gli intervalli sono calcolati dalla miscela gaussiana delle traiettorie e valutati insieme agli errori puntuali.
