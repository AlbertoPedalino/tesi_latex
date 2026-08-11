# Branch: feat/sde-net-true-minimal

## 1. Scopo dello step

- Funzione principale: implementare nello ST-GNN PVGIS-only una dinamica stocastica con percorsi paralleli di drift e diffusione allineati agli stadi dell'encoder.
- Posizione nella tesi: metodo probabilistico alternativo al Monte Carlo Dropout per ottenere distribuzioni predittive mediante perturbazioni Browniane apprese.
- Stato analizzato: codice Python del tip remoto `origin/feat/sde-net-true-minimal`, commit `2d9a783c`.
- Il modello conserva una testa di regressione puntuale e interpreta la dispersione tra traiettorie SDE come incertezza predittiva, senza una testa di varianza separata.

## 2. File di codice rilevanti

### `physiq_pv/data/pvgis_dataset.py`

- Responsabilità: preparazione dei dati PVGIS-only, split cronologico, normalizzazione train-only, finestre temporali e filtraggio opzionale degli eventi rari.
- Contenuto attuale: `build_datasets` costruisce fit, validation e test disgiunti; `PVGISWindowDataset` mantiene target fisici e metadati evento per la valutazione; `make_model` istanzia la configurazione BiLSTM--GAT con due stadi stocastici.

### `physiq_pv/data/pvgis_irradiance.py`

- Responsabilità: ricostruzione della POA.
- Contenuto attuale: `with_effective_poa` somma irradianza diretta e diffusa inclinate, valida i valori e corregge piccoli negativi entro la tolleranza numerica.

### `physiq_pv/data/pvgis_labels.py`

- Responsabilità: associazione delle etichette di anomalia o evento alle predizioni.
- Contenuto attuale: le label rimangono esterne al tensore del modello; il codice espone anche un filtraggio opzionale con sole finestre normali, ma questa funzionalità non appartiene al percorso sperimentale dichiarato per la tesi.

### `physiq_pv/model/graph_builder.py`

- Responsabilità: costruzione del grafo geografico sparso.
- Contenuto attuale: usa Haversine, raggio massimo, fallback al vicino più prossimo, self-loop e prior gaussiano della distanza.

### `physiq_pv/model/st_gnn.py`

- Responsabilità: backbone spazio-temporale e dinamica SDE.
- Contenuto attuale: `STGNN` contiene il percorso di drift BiLSTM--GAT; `MonacoDiffusionEncoder` replica gli stessi stadi con parametri indipendenti e produce gate sigmoid di diffusione per nodo e dimensione latente.

### `physiq_pv/training/losses.py`

- Responsabilità: funzione obiettivo puntuale.
- Contenuto attuale: `make_loss_fn` restituisce una MSE per la potenza e per l'eventuale uscita solare ausiliaria.

### `physiq_pv/training/noise.py`

- Responsabilità: creazione dei campioni pseudo-OOD.
- Contenuto attuale: `inject_input_noise` aggiunge rumore gaussiano soltanto ai canali continui selezionati del batch di training, preservando la codifica ciclica della geometria solare.

### `physiq_pv/training/train_loop.py`

- Responsabilità: ottimizzazione alternata di drift e diffusione.
- Contenuto attuale: `train_model` usa due ottimizzatori, valida su un anno separato e applica early stopping su RMSE diurna, MAE diurna oppure MSE.

### `physiq_pv/training/uncertainty.py`

- Responsabilità: inferenza deterministica e stocastica.
- Contenuto attuale: `predict_sde` campiona traiettorie Browniane e ricava media, deviazione standard e quantili empirici.

### `physiq_pv/experiments/pvgis_stgnn_runner.py`

- Responsabilità: esecuzione della pipeline e produzione degli output.
- Contenuto attuale: espone split, grafo, SDE, pseudo-OOD, validation, uncertainty e parametri delle analisi per evento.

## 3. Dati e preprocessing

- Input: serie PVGIS orarie con un nodo per località e coordinate geografiche coerenti tra anni.
- Feature complete: `temperature_2m`, `solar_irradiance_poa`, `wind_speed_10m`, `sin_elev`, `cos_elev`, `kt_poa`, `kt_poa_std_3h`, `dpoa_dt`, `direct_irradiance_tilted`, `diffuse_irradiance_tilted` e `pv_lag_pvgis`.
- Target: `pv_power_output`, ossia la produzione fotovoltaica teorica della località PVGIS e non una misura reale di impianto, all'orizzonte configurato; default `T=24` e `horizon=1`.
- POA: somma delle componenti diretta e diffusa sul piano inclinato.
- Feature derivate: geometria solare, POA clear-sky, indice di chiarezza inclinato, deviazione standard mobile a tre ore e differenza temporale della POA.
- Normalizzazione: statistiche stimate esclusivamente sugli anni di fit; potenza scalata per località tramite il percentile 99 dei valori positivi diurni.
- Split predefinito della pipeline: 2016--2017 per il fit, 2018 per la validation e 2019 per il test.
- `PVGISWindowDataset` restituisce `x: (N,T,F)` e `y: (N,)`; il batch è `X: (B,N,T,F)` e `Y: (B,N)`.
- Le finestre non attraversano anni diversi e le località sono riallineate per identificativo e coordinate.
- Le anomalie non entrano nel training standard. L'opzione `train_normal_only` può filtrare intere finestre, ma è una funzionalità opzionale del codice non utilizzata negli esperimenti dichiarati per la tesi.

## 4. Modello o algoritmo

### Grafo e backbone di drift

- I nodi sono località PVGIS.
- Gli archi connettono località entro 20 km; i nodi isolati ricevono un collegamento al vicino più prossimo e tutti i nodi hanno self-loop.
- Il prior di un arco è `exp(-0.5*(d/scala)^2)` e viene sommato in log-spazio ai logit appresi del GAT.
- La BiLSTM bidirezionale con attention pooling produce una rappresentazione temporale, proiettata a 96 dimensioni.
- Un GAT a quattro teste realizza il message passing spaziale.

### Percorso di diffusione

- `MonacoDiffusionEncoder` contiene una seconda BiLSTM e un secondo GAT con parametri separati dal drift.
- La diffusione restituisce un gate sigmoid `(B,N,D)` per lo stadio temporale e uno per lo stadio GAT.
- La configurazione corrente richiede esattamente due stadi stocastici: uno temporale e uno spaziale.
- L'orizzonte astratto è `4`; l'incremento usa una scala `sigma_max*sqrt(dt)`, con `sigma_max=0.5` per default.
- Durante un forward stocastico viene estratto rumore gaussiano indipendente a ogni stadio e moltiplicato per il gate di diffusione corrispondente.
- Durante la validation deterministica il percorso di drift è valutato senza perturbazioni Browniane.

### Teste e loss

- La testa PV usa Softplus e restituisce una previsione puntuale non negativa `(B,N)`.
- Una testa opzionale stima l'indice clear-sky inclinato, limitato superiormente, dal quale può essere ricostruita la POA.
- Loss di drift: MSE sulla potenza normalizzata, con MSE solare ausiliaria pesata quando attiva.
- Non è presente una testa che stimi direttamente la varianza aleatoria.

### Training alternato

1. Il percorso di drift e le teste vengono aggiornati su un forward stocastico minimizzando la MSE.
2. Un batch pseudo-OOD è creato aggiungendo rumore gaussiano ai canali ammessi dell'input.
3. I gate della diffusione sono spinti verso zero sui dati ID e verso uno sui dati pseudo-OOD mediante la somma di BCE per tutti gli stadi.
4. Drift e diffusione sono ottimizzati con due Adam separati.
5. Il checkpoint migliore è selezionato sulla validation deterministica; il default è RMSE diurna con patience 10.

### Inferenza probabilistica

- L'inferenza standard della pipeline usa 20 traiettorie Browniane.
- La media predittiva è la media delle previsioni puntuali delle traiettorie.
- La deviazione standard rappresenta la dispersione complessiva delle traiettorie SDE.
- L'intervallo principale al 95% usa i quantili empirici delle traiettorie; la banda gaussiana è soltanto diagnostica.
- Il codice dichiara esplicitamente che non separa incertezza aleatoria ed epistemica.

## 5. Elementi nuovi introdotti

- `MonacoDiffusionEncoder` parallelo al backbone di drift.
- Gate di diffusione per nodo, dimensione latente e stadio dell'encoder.
- Perturbazioni Browniane nello stadio temporale e nello stadio GAT.
- Due ottimizzatori distinti per forecasting e diffusione.
- BCE ID/pseudo-OOD applicata a tutti i gate della diffusione.
- Creazione controllata degli input pseudo-OOD.
- Validation cronologica separata ed early stopping su metrica fisica diurna.
- `predict_sde` con quantili empirici delle traiettorie.

## 6. Informazioni utili per la tesi

### Metodi

- Adattamento di un'architettura SDE con percorsi paralleli al backbone BiLSTM--GAT.
- Corrispondenza tra stadi di drift e gate di diffusione.
- Iniezione di rumore Browniano e definizione della scala `sigma*sqrt(dt)`.
- Obiettivo MSE per il forecasting puntuale.
- Apprendimento della diffusione mediante classificazione ID/pseudo-OOD.
- Generazione del pseudo-OOD con rumore gaussiano sulle feature continue.
- Split cronologico fit/validation/test e selezione del checkpoint.
- Inferenza con traiettorie multiple e intervalli a quantili empirici.
- Uso post-hoc delle anomalie per la valutazione; esclusione di `train_normal_only` dal protocollo sperimentale della tesi.

### Related Work

- Neural stochastic differential equations.
- Drift, diffusione e moto Browniano.
- Architetture SDE con encoder paralleli.
- Apprendimento della diffusione tramite dati ID e pseudo-OOD.
- Discretizzazione stocastica per reti neurali profonde.
- Intervalli predittivi basati su traiettorie Monte Carlo.
- Graph Attention Network con prior geografico.
- Forecasting fotovoltaico probabilistico.

## 7. Aspetti da chiarire

- La motivazione scientifica dell'adattamento degli stadi di un encoder convoluzionale a BiLSTM e GAT non è dimostrabile dal solo codice.
- Il codice denomina `dt` mediante una ripartizione dell'orizzonte astratto, ma non implementa una successione temporale Euler--Maruyama classica sullo stesso stato: le perturbazioni sono associate agli stadi della rete.
- La dispersione delle traiettorie non è scomposta in parte aleatoria ed epistemica.
- Non è ricavabile dal codice se il rumore pseudo-OOD rappresenti anomalie climatiche reali.
- I valori di `sigma_max`, rumore OOD e numero di traiettorie richiedono giustificazione o un'analisi di sensibilità.
- I risultati quantitativi richiedono gli output delle esecuzioni.
- La nota descrive esclusivamente il tip remoto `2d9a783c`.

## 8. Sintesi

- Il branch realizza uno ST-GNN con percorsi paralleli di drift e diffusione.
- BiLSTM e GAT possiedono gate di diffusione allineati e perturbazioni Browniane dedicate.
- Drift e diffusione sono addestrati con MSE e BCE ID/pseudo-OOD tramite ottimizzatori separati.
- Gli intervalli derivano dai quantili empirici di più traiettorie SDE.
- Il metodo produce dispersione predittiva, ma non separa incertezza aleatoria ed epistemica.
