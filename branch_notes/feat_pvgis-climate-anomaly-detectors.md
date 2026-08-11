# Branch: feat/pvgis-climate-anomaly-detectors

## 1. Scopo dello step

- Funzione principale: applicare MTGFlow in modalità label-free alle condizioni climatiche PVGIS e produrre score di anomalia globali e per variabile.
- Posizione nella tesi: stadio di anomaly detection che assegna etichette normale/anomalo indipendenti dal modello di forecasting e consente la successiva analisi post-hoc degli errori.
- Stato analizzato: codice Python del tip remoto `origin/feat/pvgis-climate-anomaly-detectors`, commit `85d799b6`.
- L'implementazione addestra un modello MTGFlow separato per ciascuna località. Il grafo dinamico interno collega le variabili climatiche della località e non rappresenta relazioni geografiche tra località diverse.

## 2. File di codice rilevanti

### `physiq_pv/anomaly_detection/pvgis_climate.py`

- Responsabilità: preparazione leakage-safe delle serie climatiche per ogni località.
- Contenuto attuale: `prepare_pvgis_climate_data` legge gli anni PVGIS, verifica l'ordine delle località e scrive train, validation opzionale e test separati per sito; `SeasonalRobustScaler` offre una normalizzazione mensile-oraria opzionale.

### `physiq_pv/anomaly_detection/common.py`

- Responsabilità: selezione sicura delle feature, validazione numerica e riproducibilità.
- Contenuto attuale: `detector_features` impedisce l'ingresso di target o label nel detector; `seed_everything` configura le sorgenti casuali e gli algoritmi deterministici.

### `physiq_pv/anomaly_detection/mtgflow/config.py`

- Responsabilità: iperparametri del profilo MTGFlow di riferimento.
- Contenuto attuale: `MTGFlowReferenceConfig` definisce epoche, finestra, stride, batch, learning rate, flow blocks, hidden size, dropout e parametri IQR; sono previste cinque seed.

### `physiq_pv/anomaly_detection/mtgflow/model.py`

- Responsabilità: architettura MTGFlow.
- Contenuto attuale: contiene `DynamicGraphAttention`, `SpatioTemporalConditioner`, `ConditionalMAFBlock`, `EntityAwareMAF` e `MTGFlow`; espone likelihood globale e likelihood per entità.

### `physiq_pv/anomaly_detection/mtgflow/pipeline.py`

- Responsabilità: normalizzazione, finestre, training non supervisionato, checkpoint e scoring.
- Contenuto attuale: `fit_and_score_mtgflow` usa esclusivamente il training per z-score, ottimizzazione e score di calibrazione; la validation opzionale è soltanto validata nello schema e non seleziona il checkpoint.

### `physiq_pv/anomaly_detection/thresholds.py`

- Responsabilità: conversione degli score continui in decisioni anomale.
- Contenuto attuale: implementa soglia IQR o quantile globale e soglie IQR separate per ciascuna entità, tutte stimate esclusivamente sugli score di training.

### `physiq_pv/anomaly_detection/evaluation.py`

- Responsabilità: collegamento post-hoc con le predizioni del forecasting.
- Contenuto attuale: associa score e flag tramite `(location,timestamp)` e calcola MAE, RMSE, bias e, se disponibili, metriche d'incertezza per gruppi normali/anomali.

### `scripts/prepare_pvgis_mtgflow.py`

- Responsabilità: interfaccia CLI per la preparazione dei dati.
- Contenuto attuale: usa per default 2005--2018 come training e 2019 come test; supporta sharding e batching delle località.

### `scripts/run_pvgis_mtgflow.py`

- Responsabilità: esecuzione per località e seed, soglie e output aggregati.
- Contenuto attuale: per ogni riga del manifest crea un modello e un checkpoint indipendenti; produce score globali e per entità e riepiloga il tasso di anomalie senza aggregare gli score delle diverse seed.

## 3. Dati e preprocessing

- Input del detector: tre serie climatiche per località, `solar_irradiance_poa`, `temperature_2m` e `wind_speed_10m`.
- Esclusioni metodologiche: la produzione PV, le label e qualsiasi target supervisionato non possono entrare nelle feature del detector.
- Protocollo temporale predefinito: training dal 2005 al 2018 e test sul 2019; una validation opzionale deve essere distinta, ma non è utilizzata per training o soglia.
- Ogni località viene trasformata in un CSV cronologico distinto. Le località non sono concatenate come nodi dello stesso modello.
- Il preprocessing MTGFlow standard applica z-score per variabile usando media e deviazione standard calcolate esclusivamente sul training della località.
- È disponibile una normalizzazione robusta mensile-oraria opzionale basata su mediana e MAD, ma il runner del profilo di riferimento la rifiuta.
- Le finestre hanno lunghezza predefinita 60; train e scoring usano stride 10 nel profilo di riferimento, mentre stride 1 è previsto come adattamento per ottenere score orari.
- Una finestra per una località ha forma `(K,T,1)`, con `K=3` entità climatiche e `T=60`; dopo il batching l'input è `(B,K,T,1)`.
- Le finestre che attraversano lacune o cambi di cadenza temporale sono eliminate.

## 4. Modello o algoritmo

### Grafo dinamico tra entità

- Le entità sono le tre variabili climatiche della singola località.
- `DynamicGraphAttention` appiattisce la finestra di ogni entità e calcola query e key.
- Il prodotto scalato query-key genera una matrice di adiacenza dinamica `(B,K,K)`, normalizzata per riga con softmax.
- La matrice può cambiare tra finestre e modella dipendenze come irradianza--temperatura o temperatura--vento.
- Non usa latitudine, longitudine, distanza Haversine o informazioni provenienti da altre località.

### Condizionamento temporale e normalizing flow

1. Una LSTM condivisa elabora separatamente la sequenza di ogni entità e produce `H: (B,K,T,Hd)`.
2. `SpatioTemporalConditioner` combina il message passing tramite l'adiacenza dinamica con lo stato temporale precedente.
3. Il condizionamento alimenta due blocchi MAF scalari condivisi tra entità.
4. Ogni entità possiede una media gaussiana di base distinta, mentre i parametri trasformativi sono condivisi.
5. Il modello calcola log-likelihood esatte per punto, entità e finestra includendo il log-Jacobiano del flow.

### Training

- Obiettivo: massimizzare la log-likelihood media delle finestre di training, implementato minimizzandone il negativo.
- Ottimizzatore: Adam, learning rate `2e-3`, weight decay `5e-4`.
- Profilo predefinito: 40 epoche, batch 256, hidden size 32, due blocchi flow, un livello nascosto e attention dropout 0.2.
- I gradienti sono limitati per valore a 1.0.
- Il checkpoint è quello dell'ultima epoca fissa; non avviene selezione mediante label o validation.
- Seed di riferimento: 15, 16, 17, 18 e 19, con un training separato per ogni località e seed.

### Score e soglie

- Score per entità: negativo della log-likelihood della finestra diviso per il numero di entità.
- Score aggregato della località, denominato `global score` nel codice: somma dei contributi delle entità del modello relativo a quella località, equivalente alla media delle negative log-likelihood di entità. Non aggrega località differenti.
- Lo score della finestra è assegnato al timestamp finale della finestra; vengono salvati anche inizio e fine.
- Soglia sullo score aggregato, stimata separatamente per località e seed: `Q3 + 1.5*IQR` sugli score di training.
- Soglia per entità: `0.8*(Q3 + 1.5*IQR)` stimata separatamente per ogni variabile sugli score di training.
- Una osservazione è anomala quando lo score è maggiore o uguale alla soglia.

### Granularità spaziale effettiva

- Il manifest contiene tutte le località, ma il runner itera sulle righe e chiama `fit_and_score_mtgflow` separatamente.
- Ogni modello vede soltanto la serie di una località.
- Il termine `global` usato internamente indica soltanto l'aggregazione delle entità nel modello della località corrente.
- Gli output di località diverse sono concatenati soltanto dopo lo scoring.
- Di conseguenza, il branch consente mappe geografiche post-hoc degli score, ma MTGFlow non apprende interazioni spazio-temporali tra località.

## 5. Elementi nuovi introdotti

- Pipeline PVGIS per serie climatiche separate per località.
- Implementazione autonoma dell'architettura MTGFlow base.
- Grafo dinamico tra variabili climatiche.
- Condizionatore spazio-temporale interno alle entità della singola serie multivariata.
- Normalizing flow MAF condizionale con target gaussiani entity-aware.
- Score globali e per variabile.
- Soglie IQR stimate esclusivamente sul training.
- Esportazione di `anomaly_scores.csv`, `train_anomaly_scores.csv` ed `entity_anomaly_scores.csv` per ogni seed.
- Collegamento post-hoc tra decisioni MTGFlow ed errori del forecasting.

## 6. Informazioni utili per la tesi

### Metodi

- Definizione label-free del problema di anomaly detection climatica.
- Separazione temporale 2005--2018/2019 e prevenzione del leakage.
- Un modello indipendente per località e ruolo delle tre variabili come entità MTGFlow.
- Forma `(B,K,T,1)` e finestratura regolare.
- Self-attention per il grafo dinamico tra entità.
- LSTM e condizionamento grafo-temporale.
- Masked Autoregressive Flow condizionale e likelihood esatta.
- Training mediante massima likelihood anche in presenza potenziale di anomalie.
- Score aggregato per località e score per entità.
- Soglia IQR sullo score aggregato e soglie per variabile.
- Interpretazione dello score per entità come associazione statistica alla bassa likelihood, non come identificazione causale dell'evento fisico.
- Uso di più seed senza mediare direttamente gli score.
- Join post-hoc con le predizioni PV e metriche separate normali/anomale.
- Limite spaziale: nessun message passing tra località.

### Related Work

- Anomaly detection label-free e anomaly-unaware.
- Density estimation per rilevamento delle anomalie.
- Normalizing flows e Masked Autoregressive Flow.
- Self-attention per grafi dinamici di serie multivariate.
- Modelli entity-aware per anomaly localization.
- Soglie robuste basate su intervallo interquartile.
- Anomaly detection climatica su dati meteorologici multivariati.
- Valutazione post-hoc del forecasting condizionata da eventi anomali.

## 7. Aspetti da chiarire

- Il codice non dimostra che le tre variabili e la finestra di 60 ore siano sufficienti a identificare tutti gli eventi climatici di interesse.
- Il modello non utilizza dipendenze tra località; un'eventuale componente geografica richiederebbe un'architettura diversa o un secondo livello di grafo.
- La semantica fisica di un'anomalia dipende dalla soglia e dalla likelihood, non da una label causale.
- Gli score sono definiti su finestre e attribuiti al loro termine; uno score orario denso richiede `score_stride=1`, che è un adattamento rispetto al profilo di riferimento.
- La validation opzionale non seleziona checkpoint o iperparametri.
- La soglia IQR non è ottimizzata rispetto agli errori del forecasting o a label esterne.
- I risultati numerici, il tasso di anomalie e la stabilità tra seed richiedono gli output sperimentali.
- La nota descrive esclusivamente il tip remoto `85d799b6`.

## 8. Sintesi

- Il branch applica MTGFlow label-free alle tre variabili climatiche di ogni località PVGIS.
- Il grafo dinamico collega variabili della stessa località, non località geografiche diverse.
- LSTM, condizionamento grafo-temporale e MAF producono likelihood globali e per entità.
- Le soglie IQR sono stimate esclusivamente sugli score di training 2005--2018.
- Gli score del 2019 vengono collegati post-hoc agli errori del forecasting.
