# Branch: feat/stgan-paper

## 1. Scopo dello step

- Introdurre un detector generativo spazio-temporale STGAN per le serie meteorologiche PVGIS.
- Posizione: linea di anomaly detection distinta dalla regressione della produzione PV; le sue decisioni possono essere utilizzate successivamente nelle analisi del previsore.
- Revisione analizzata: `origin/feat/stgan-paper` a `777df6bc`, disponibile localmente il 10 settembre 2026. Il riferimento locale è fermo a `47b58e44`; i commit successivi non aggiungono modifiche Python al metodo.
- Predecessore effettivo: `feat/pvgis-climate-anomaly-detectors` a `85d799b6`, genitore del commit iniziale `004bb756`. Confronto `85d799b6..777df6bc`.
- Differenza: nuovo generatore/discriminatore con convoluzioni sul grafo e contesto temporale, accanto alla preparazione meteorologica ereditata.
- Selezione del ramo per aggiornamenti nel periodo 10 agosto–10 settembre 2026; descrizione dello step complessivo, inclusa l'introduzione precedente al periodo.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/anomaly_detection/stgan/config.py` | Configurazione | Introduce `STGANReferenceConfig`, dimensioni del contesto e parametri di training. |
| `physiq_pv/anomaly_detection/stgan/graph.py` | Contesto spaziale | Introduce `GeographicalSubgraphs`, distanze haversine e sottografi kNN geografici. |
| `physiq_pv/anomaly_detection/stgan/data.py` | Dati allineati e finestre | Cubi temporali, contesto train/test, calendario e dataset per località target. |
| `physiq_pv/anomaly_detection/stgan/model.py` | Reti neurali | `GraphConvolution`, `GCGRUCell`, `GCGRU`, generatore, discriminatore e coppia `STGAN`. |
| `physiq_pv/anomaly_detection/stgan/pipeline.py` | Training e scoring | Normalizzazione, aggiornamenti avversari, componenti dello score e inferenza. |
| `physiq_pv/anomaly_detection/stgan/result.py` | Interfaccia dei risultati | Contenitore `STGANResult` per score, coordinate e metadati. |
| `physiq_pv/anomaly_detection/pvgis_climate.py` | Preparazione ereditata | Esporta anche coordinate geografiche nel manifest per costruire i sottografi. |
| `scripts/prepare_pvgis_stgan.py` | Preparazione del detector | Configura il percorso PVGIS dedicato a STGAN. |
| `scripts/run_pvgis_stgan.py` | Esecuzione e decisioni | Collega il detector ai dati e introduce `paper_top_k_ranking`. |

## 3. Dati e preprocessing

- Feature del percorso PVGIS: `solar_irradiance_poa`, `temperature_2m`, `wind_speed_10m`.
- La produzione PV futura non è il target della GAN: il generatore stima il successivo vettore meteorologico dell'intero sottografo.
- Preparazione configurabile; default storico 2005–2018 e test 2019. Questi sono default, non una certificazione dei dati realmente usati.
- I cubi hanno forma `[T,N,F]` e richiedono timestamp sincronizzati tra località.
- Minimo e intervallo per feature sono stimati sul training aggregando tempo e località; trasformazione `2*(x-min)/range-1`, senza clipping generale al range del training.
- Il test viene preceduto dalla coda contigua del training, usata solo come contesto.
- Default: sottografo di nove siti, contesto recente di un passo, storia lunga di 168 passi della località centrale.
- Batch: recente `[B,1,9,F]`, trend `[B,168,F]`, adiacenza `[B,9,9]`, calendario `[B,31]`, osservazione target `[B,9,F]`.
- I 31 canali temporali sono one-hot dell'ora e del giorno della settimana.
- Il sottografo contiene la località centrale in posizione zero e i vicini geografici. Il numero di nove è un default; il costruttore lo limita anche al numero di siti disponibili.

## 4. Modello o algoritmo

- Grafo: kNN diretto sulle distanze geografiche, pesi `exp(-d²/sigma²)`, self-loop e normalizzazione `D^(-1/2) A D^(-1/2)`.
- La scala geografica di default è la deviazione standard campionaria delle distanze degli archi, con fallback per configurazioni degeneri.
- La convoluzione applica `activation(Linear(A*x))`; nei GCGRU i gate di reset, aggiornamento e candidato usano questa operazione.
- Generatore: GCGRU sul contesto recente, LSTM sul trend del sito centrale e proiezione del calendario. I tre rami vengono concatenati per nodo e trasformati da una convoluzione sul grafo con uscita tanh.
- Discriminatore: GCGRU sul contesto storico, proiezione dello stato appiattito, convoluzione sul dato corrente con massimo sui nodi, quindi MLP sigmoidale.
- Configurazione di riferimento: hidden 64, due strati ricorrenti e dimensione GCGRU 32.
- Loss del discriminatore: `0.5*(BCE(D(reale),0)+BCE(D(generato),1))`.
- Loss del generatore: `500*MSE(predetto,osservato)+BCE(D(generato),0)` nel default.
- Due ottimizzatori Adam, learning rate di default 0.001, sei epoche, batch 256, seed 20. L'output sigmoidale del discriminatore è addestrato a distinguere reale/generato; non costituisce una probabilità di anomalia calibrata su etichette reali.
- Il training usa per default tutto il prodotto finestre–località mescolato; un budget positivo abilita campionamento con reinserimento.
- Componente generativa dello score: errore quadratico medio su tutti i nodi e le feature del sottografo.
- Componente discriminativa: `D(reale)-D(generato)`.
- Le due componenti sono normalizzate separatamente con min–max sull'intero insieme test tempo–località e sommate con peso unitario.
- Le diagnostiche per feature sono invece errori quadratici della sola località centrale.
- Decisione: ranking globale stabile, con `ceil(n*percentuale/100)` anomalie; percentuale di default 1. Non è una soglia separata per località.
- Lo score richiede l'osservazione al tempo valutato per misurare errore e risposta del discriminatore: non è una previsione anticipata dell'etichetta di anomalia.

## 5. Elementi nuovi introdotti

- Pacchetto STGAN e sottografi geografici di dimensione locale.
- Generazione meteorologica condizionata a storia recente, trend e calendario.
- Score combinato generativo/discriminativo e decisione globale top-K.
- Esportazione separata di score globali e residui per feature.
- Nessuna sostituzione del previsore SDE nel delta del ramo.

## 6. Informazioni utili per la tesi

### Metodi

- Differenza tra detector meteorologico e previsore della produzione.
- Costruzione del sottografo, ordinamento del centro e pesatura degli archi.
- I due contesti temporali e il calendario.
- Obiettivi avversari, score sui nodi e diagnostiche centrali.
- Normalizzazione delle feature sul training rispetto a normalizzazione dello score sul test.
- Ranking globale e gestione dei pareggi.

### Related Work

- GAN per anomaly detection spazio-temporale.
- Convoluzioni sui grafi, GCGRU e LSTM.
- Grafi kNN e kernel geografici gaussiani.
- Score basati su errore generativo e discriminatore.
- Ranking top-K e protocolli retrospettivi di anomaly detection.

## 7. Aspetti da chiarire

- La fedeltà all'articolo citato nei commenti richiede confronto bibliografico separato; non è verificata qui.
- La normalizzazione dello score e il top-K usano la distribuzione test completa: il codice descrive un protocollo retrospettivo, senza dimostrare una soglia pronta per uso online causale.
- Numero reale di siti, campioni, durata e accuratezza non sono inferiti dai default.
- L'eventuale regolarità della griglia geografica e la coincidenza fra kNN e blocchi 3×3 richiedono dati di coordinate, esclusi da questa analisi.
- Non sono implementati intervalli gaussiani predittivi per l'uscita della GAN.
- Le motivazioni fisiche dei singoli eventi e la validità delle etichette richiedono fonti indipendenti.

## 8. Sintesi

- Detector STGAN con GCGRU spaziale, trend LSTM e calendario.
- Generazione del vettore meteorologico successivo sul sottografo.
- Score combinato normalizzato globalmente sul test e top-K.
- Nessun risultato sperimentale o confronto di efficacia dedotto dal sorgente.
