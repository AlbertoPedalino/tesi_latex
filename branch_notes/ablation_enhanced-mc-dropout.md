# Branch: ablation/enhanced-mc-dropout

## 1. Scopo dello step

- Funzione principale: valutare una variante probabilistica dello ST-GNN PVGIS-only mediante Monte Carlo Dropout.
- Posizione nella tesi: metodo di quantificazione dell'incertezza applicato al forecasting spazio-temporale, confrontabile con l'approccio basato su SDE.
- Stato analizzato: codice Python del tip remoto `origin/ablation/enhanced-mc-dropout`, commit `73283af6`.
- L'architettura produce una previsione puntuale della potenza e usa la variabilità tra passaggi con maschere dropout diverse per stimare dispersione e intervalli predittivi.

## 2. File di codice rilevanti

### `physiq_pv/data/pvgis_stgnn_dataset.py`

- Responsabilità: caricamento PVGIS, feature engineering, normalizzazione, finestre temporali, training, inferenza deterministica e Monte Carlo Dropout.
- Contenuto attuale: contiene `PVGISWindowDataset`, `build_datasets`, `train_model`, `enable_dropout_only` e `predict_mc`; associa inoltre le etichette climatiche soltanto alle predizioni per la valutazione stratificata.

### `physiq_pv/model/bilstm_encoder.py`

- Responsabilità: codifica temporale condivisa per tutti i nodi.
- Contenuto attuale: `BiLSTMEncoder` usa una BiLSTM bidirezionale a due strati, LayerNorm e pooling temporale configurabile; la configurazione del branch usa attention pooling.

### `physiq_pv/model/graph_builder.py`

- Responsabilità: costruzione del grafo geografico.
- Contenuto attuale: collega bidirezionalmente località entro il raggio configurato mediante distanza Haversine e assegna pesi inversamente proporzionali alla distanza; se il grafo è vuoto usa collegamenti al vicino più prossimo.

### `physiq_pv/model/st_gnn.py`

- Responsabilità: modello ST-GNN e punti di dropout utilizzati nell'inferenza Monte Carlo.
- Contenuto attuale: `STGNN` combina BiLSTM, proiezione latente, GAT sparso e teste per indice clear-sky e potenza; con `enhanced_dropout=True` applica dropout all'embedding temporale, alla rappresentazione proiettata e nella testa PV, oltre al dropout dell'attenzione GAT.

### `physiq_pv/experiments/pvgis_stgnn_runner.py`

- Responsabilità: configurazione della pipeline e calcolo delle metriche.
- Contenuto attuale: espone `stgnn_enhanced_dropout`, numero di campioni MC, coverage target e iperparametri di training; produce metriche puntuali e di intervallo, anche per giorno/notte e strati di anomalia.

### `physiq_pv/reporting/daytime_bin_anomaly_report.py`

- Responsabilità: analisi post-hoc delle predizioni diurne.
- Contenuto attuale: calcola MAE, RMSE, copertura e ampiezza degli intervalli per bin di produzione e categorie normali/anomale.

## 3. Dati e preprocessing

- Input: file NetCDF/xarray annuali PVGIS con serie orarie e nodi corrispondenti alle località geografiche.
- Feature del set completo: `temperature_2m`, `solar_irradiance_poa`, `wind_speed_10m`, `sin_elev`, `cos_elev`, `kt`, `kt_std_3h`, `dghi_dt`, `dni_norm`, `dhi_norm` e `pv_lag_pvgis`.
- Target: `pv_power_output`, normalizzato per località mediante una scala stimata sui dati di training.
- Temperatura, irradianza, vento e variazione dell'irradianza sono standardizzati usando statistiche di training; la produzione e il suo ritardo sono scalati per località.
- Il valore normalizzato della potenza è limitato inferiormente a zero e, per default, superiormente a `1.5`.
- Il dataset costruisce finestre senza attraversare il confine tra anni. La configurazione predefinita usa `T=24` e orizzonte `t+1`.
- Un elemento del dataset ha `x: (N,T,F)` e `y: (N,)`; dopo il batching, `X: (B,N,T,F)` e `Y: (B,N)`.
- La label climatica, se fornita, è un metadato associato tramite `(location,timestamp)` dopo l'inferenza e non è un input o un target di training.

## 4. Modello o algoritmo

### Architettura di forecasting

1. La sequenza di ogni nodo viene rimodellata in `(B*N,T,F)` e passata nella BiLSTM condivisa.
2. L'attention pooling restituisce un embedding temporale di 256 componenti per nodo.
3. Una proiezione `Linear-GELU-LayerNorm` porta la rappresentazione a 96 componenti.
4. Un GAT a quattro teste propaga informazione lungo gli archi geografici; il calcolo dell'attenzione e l'aggregazione avvengono in forma sparsa.
5. La testa solare stima un indice clear-sky limitato; la testa PV usa Softplus e produce una previsione non negativa.

### Enhanced dropout

- La variante `stgnn_enhanced_dropout` contiene moduli `nn.Dropout` dopo l'encoder temporale, dopo la proiezione e prima dell'ultimo livello della testa PV.
- In training il dropout agisce come regolarizzatore con probabilità predefinita `0.2`.
- In inferenza `enable_dropout_only` porta l'intero modello in `eval()` e riattiva esclusivamente i moduli dropout. BiLSTM e normalizzazioni restano quindi in modalità di valutazione.
- Ogni passaggio produce una previsione PV diversa per la stessa coppia località-timestamp. Il default è di 30 passaggi Monte Carlo.

### Loss e training

- Loss PV principale: MSE sulla produzione normalizzata.
- Loss ausiliaria opzionale: MSE tra indice clear-sky previsto e target.
- Termine aggiuntivo: errore assoluto pesato in funzione dei picchi, con penalizzazione maggiore delle sottostime.
- Loss complessiva: somma pesata dei tre contributi attivi.
- Ottimizzatore: AdamW con learning rate predefinito `1e-3` e weight decay `1e-4`.
- Configurazione CLI predefinita: 10 epoche, batch size 8 e grafo con raggio 20 km.

### Inferenza probabilistica

- La previsione puntuale è la media delle predizioni Monte Carlo.
- `y_pred_std_raw` è la deviazione standard tra i passaggi.
- L'intervallo principale al 95% è ottenuto direttamente dai quantili empirici `0.025` e `0.975` delle predizioni.
- La banda `media ± 1.96*deviazione standard` è mantenuta come diagnostica secondaria.
- Le metriche comprendono MAE, RMSE, PICP, MPIW, NMPIL e CLC.

## 5. Elementi nuovi introdotti

- Modalità `stgnn_enhanced_dropout` selezionabile dal runner.
- Dropout esplicito nell'embedding temporale, nella rappresentazione latente e nella testa PV.
- `enable_dropout_only` per l'attivazione selettiva del dropout in inferenza.
- `active_dropout_names` per verificare quali moduli contribuiscono alla variabilità Monte Carlo.
- `predict_mc` per media, dispersione e quantili empirici delle predizioni.
- Output tabellari con limiti dell'intervallo principale e banda gaussiana diagnostica.
- Metriche di affidabilità e sharpness degli intervalli.

## 6. Informazioni utili per la tesi

### Metodi

- Uso del Monte Carlo Dropout come approssimazione della distribuzione predittiva.
- Posizionamento dei moduli dropout nello ST-GNN.
- Riattivazione del solo dropout durante l'inferenza.
- Numero di passaggi stocastici e aggregazione per località-timestamp.
- Costruzione di intervalli mediante quantili empirici.
- Distinzione tra intervallo empirico principale e banda gaussiana diagnostica.
- Architettura BiLSTM--GAT e grafo geografico usati come backbone.
- Loss di forecasting, termine solare ausiliario e penalizzazione dei picchi.
- Metriche puntuali e probabilistiche, anche stratificate per anomalie.

### Related Work

- Dropout come regolarizzazione delle reti neurali.
- Monte Carlo Dropout e interpretazione bayesiana approssimata.
- Incertezza epistemica nel forecasting neurale.
- Intervalli predittivi tramite quantili empirici di campioni Monte Carlo.
- Valutazione di coverage e sharpness.
- Forecasting spazio-temporale con BiLSTM e Graph Attention Network.

## 7. Aspetti da chiarire

- Il codice non fornisce una motivazione teorica per la scelta di probabilità dropout, numero di campioni e posizioni dei moduli aggiuntivi.
- Non è ricavabile dal codice se la dispersione prodotta sia calibrata sui dati reali.
- Non è presente una separazione esplicita tra componente aleatoria ed epistemica: la deviazione standard rappresenta soltanto la variabilità indotta dalle maschere dropout.
- Il training non contiene un obiettivo diretto di calibrazione degli intervalli.
- Gli anni effettivamente usati dipendono dagli argomenti CLI e non hanno default nel runner.
- I risultati quantitativi e il confronto con SDE-Net richiedono gli output sperimentali.
- La nota descrive esclusivamente il tip remoto `73283af6`.

## 8. Sintesi

- Il branch applica Monte Carlo Dropout allo ST-GNN PVGIS-only.
- Il dropout è presente nel percorso temporale, latente, spaziale e nella testa PV.
- La previsione è la media di più passaggi e l'intervallo deriva dai quantili empirici.
- Il backbone mantiene BiLSTM, GAT geografico e previsione non negativa della potenza.
- Le anomalie sono usate soltanto per valutare separatamente gli errori e gli intervalli.
