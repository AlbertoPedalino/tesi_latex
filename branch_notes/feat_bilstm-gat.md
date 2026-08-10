# Branch: feat/bilstm-gat

## 1. Scopo dell'implementazione

Il branch implementa un modello ST-GNN per la previsione oraria della produzione fotovoltaica distribuita. Ogni impianto è elaborato temporalmente da un BiLSTM; le rappresentazioni ottenute vengono poi scambiate tra impianti mediante un GAT costruito su un grafo geografico. Il modello produce congiuntamente una previsione dell'irradianza e una della produzione fotovoltaica.

## 2. File di codice rilevanti

- `physiq_pv/model/bilstm_encoder.py`: encoder temporale BiLSTM per singolo nodo e pooling della sequenza.
- `physiq_pv/model/st_gnn.py`: composizione di encoder temporale, proiezione latente, GAT e teste di uscita.
- `physiq_pv/model/graph_builder.py`: costruzione del grafo geografico tramite distanza di Haversine.
- `physiq_pv/model/physics_loss.py`: loss congiunta per irradianza, produzione PV e coerenza fisica.
- `physiq_pv/data/dataset.py`: feature, target e finestre temporali PyTorch.
- `physiq_pv/data/sentinel_hourly_loader.py`: caricamento e allineamento dei dati degli impianti con PVGIS.
- `train.py`: training, validazione, early stopping e salvataggio del modello migliore.
- `main.py`: configurazione ed esecuzione della pipeline.

## 3. Dati e preprocessing

- Input: serie orarie di produzione degli impianti, coordinate geografiche e variabili meteorologiche PVGIS associate alla località più vicina.
- Variabili meteorologiche principali: temperatura a 2 m, irradianza sul piano, velocità del vento e componenti irradiative diretta e diffusa, quando disponibili.
- Feature temporali: geometria solare, produzione PV passata, indice di cielo sereno, variabilità a breve termine dell'irradianza e componenti diretta/diffusa.
- Target irradianza: `solar_irradiance_poa / 1000`.
- Target PV: `ENERGIA` normalizzata per impianto tramite il 99° percentile dei valori diurni positivi e limitata a `[0, 1.5]`.
- Finestra temporale: 24 ore precedenti per prevedere l'ora successiva.
- Forma generale del batch: `x: (B, N, 24, C)` e target `(B, N)`.
- Split: primo 80% delle finestre di ogni mese per il training e restante 20% per la validazione.
- Le componenti irradiative inclinate di PVGIS vengono usate direttamente quando disponibili; altrimenti viene applicata la decomposizione Erbs.

## 4. Architettura

### Encoder temporale

- BiLSTM applicato separatamente a ciascun impianto.
- Due livelli ricorrenti.
- 128 unità nascoste per direzione.
- Dropout pari a `0.2`.
- Output temporale di dimensione 256.
- La configurazione in `main.py` usa attention pooling.
- L'uscita viene proiettata da 256 a 96 dimensioni tramite proiezione lineare, GELU e LayerNorm.

### Grafo geografico e GAT

- Un nodo per ogni impianto.
- Archi bidirezionali tra impianti entro 20 km.
- Peso degli archi inversamente proporzionale alla distanza.
- Un livello GAT con quattro teste.
- Aggregazione sugli archi entranti, connessione residua e LayerNorm.

### Teste di uscita

- Testa irradianza: produce un indice clear-sky limitato tramite sigmoid e lo moltiplica per `ghi_cs`.
- Testa PV: usa Softplus per garantire una previsione non negativa.
- Entrambe le teste sono MLP indipendenti con struttura `96 -> 48 -> 1`.

## 5. Loss e addestramento

La loss principale combina:

1. errore quadratico sulla previsione dell'irradianza;
2. errore quadratico sulla previsione PV;
3. penalità di coerenza fisica tra irradianza e produzione PV;
4. termine asimmetrico che assegna maggiore peso ai picchi e alle sottostime della produzione.

Configurazione principale ricavata dal codice:

- ottimizzatore AdamW;
- learning rate `1e-3`;
- weight decay `1e-4`;
- batch size `8`;
- massimo 15 epoche;
- early stopping con patience `5`;
- rumore moltiplicativo gaussiano sui canali meteorologici durante il training;
- ripristino dei pesi corrispondenti alla migliore loss di validazione;
- inferenza con `model.eval()` e `torch.no_grad()`.

## 6. Informazioni utili per la tesi

### Metodi

- Finestra storica di 24 ore e previsione one-step-ahead.
- BiLSTM bidirezionale applicato per impianto.
- Attention pooling della sequenza temporale.
- Costruzione del grafo geografico e message passing GAT.
- Doppia testa per irradianza e produzione PV.
- Vincolo clear-sky e penalità di coerenza fisica.
- Normalizzazione del target PV e preprocessing delle variabili meteorologiche.
- Loss asimmetrica per picchi e sottostime.
- Protocollo di training e validazione.

### Concetti da trattare nel Related Work

- BiLSTM per serie temporali.
- Attention pooling.
- Graph Attention Network.
- Modelli spaziotemporali per forecasting fotovoltaico.
- Grafi geografici basati sulla distanza.
- Indice clear-sky e decomposizione Erbs.
- Apprendimento physics-informed.
- Loss asimmetriche per la previsione energetica.

## 7. Aspetti da chiarire

- Motivazione scientifica delle principali scelte architetturali.
- Semantica di `solar_irradiance_poa`, indicata in alcuni punti come GHI nonostante il riferimento al piano inclinato.
- Unità e significato temporale esatto di `ENERGIA`.
- Opportunità di calcolare statistiche e percentile di normalizzazione prima dello split temporale.
- Disponibilità effettiva delle componenti PVGIS inclinate oppure uso del fallback Erbs.
- Numero effettivo di impianti, archi e timestamp, dipendente dai dati caricati.

## 8. Sintesi

Il branch implementa una pipeline ST-GNN con encoder temporale BiLSTM per impianto e propagazione spaziale GAT su un grafo geografico. Una finestra di 24 ore viene utilizzata per prevedere irradianza e produzione fotovoltaica all'ora successiva. Le uscite sono regolarizzate mediante vincoli fisici e una penalità specifica per picchi e sottostime.
