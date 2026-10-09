# Branch: experiment/stgan-cnn

## 1. Scopo dello step

- Implementare una variante STGAN con convoluzioni spaziali su patch geografiche, conservando il ramo temporale LSTM e il protocollo avversario.
- Posizione: nuova variante del detector, integrata nella linea che contiene le analisi successive del previsore.
- Revisione analizzata: `28e532c1`, ramo corrente e riferimento `origin` concordi nella copia consultata il 10 settembre 2026.
- Predecessore Git effettivo: `feat/anomaly-spatial-threshold-comparison` a `6356f1f9`, genitore diretto. Confronto primario `6356f1f9..28e532c1`.
- Predecessore metodologico del detector: codice STGAN di `origin/feat/stgan-paper` a `777df6bc`, importato nella linea corrente. Non è il genitore Git di questo commit.
- Rispetto alla base Git, il pacchetto del detector viene aggiunto; rispetto alla sorgente STGAN, i moduli GCGRU e le convoluzioni sul grafo vengono sostituiti da CNN.
- Periodo di selezione: 10 agosto–10 settembre 2026. Sono esclusi dati, risultati e configurazioni contenute fuori dai sorgenti Python.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/anomaly_detection/stgan/config.py` | Configurazione della variante | Introduce `STGANCNNConfig`, dimensioni CNN, dimensione della patch e parametri della griglia. |
| `physiq_pv/anomaly_detection/stgan/grid.py` | Geometria senza adiacenza | `SpatialGrid` e `build_spatial_grid`: proiezione, verifica del reticolo, indici e maschera. |
| `physiq_pv/anomaly_detection/stgan/data.py` | Finestre geografiche | Adatta il dataset importato a patch `[F,K,K]` e celle assenti. |
| `physiq_pv/anomaly_detection/stgan/model.py` | Reti CNN–LSTM | `SpatialEncoder`, generatore, discriminatore, riduzione mascherata e conteggio dei parametri. |
| `physiq_pv/anomaly_detection/stgan/pipeline.py` | Training e scoring | Conserva gli obiettivi avversari e applica loss/score solo alle celle valide. |
| `physiq_pv/anomaly_detection/stgan/result.py` | Interfaccia | Mantiene il contenitore degli score importato da STGAN. |
| `physiq_pv/anomaly_detection/common.py` | Controlli condivisi | Validazione delle feature e gestione del seed importate nella linea corrente. |
| `physiq_pv/anomaly_detection/pvgis_climate.py` | Preparazione | Riutilizza le tre variabili meteorologiche e le coordinate. |
| `scripts/prepare_pvgis_stgan.py` | Preparazione dedicata | Collega i dati PVGIS al detector CNN. |
| `scripts/run_pvgis_stgan.py` | Configurazione ed esportazione | Aggiunge audit della griglia, parametri CNN e riepiloghi distinti per patch complete/incomplete. |
| `physiq_pv/reporting/stgan_cnn_comparison.py` | Confronto fra detector | Introduce `compare_stgan_exports` su coordinate comuni e gruppi geografici. |

## 3. Dati e preprocessing

- Input del percorso PVGIS: POA, temperatura a 2 m e vento a 10 m; coordinate geografiche per assegnare le celle.
- I cubi temporali rimangono `[T,N,F]`; il sito centrale e i vicini vengono disposti in una patch con orientamento determinato dal reticolo.
- Default della configurazione: CRS `EPSG:32632`, passo 5.000 m, tolleranza 25 m. La funzione geometrica può anche ricevere `grid_crs="auto"` e verificare più CRS.
- Le coordinate vengono arrotondate agli indici del reticolo; residui eccessivi o siti duplicati nella stessa cella provocano un errore.
- Le righe seguono nord→sud e le colonne ovest→est della griglia scelta. Non è un semplice reshape dell'ordine originario dei siti.
- `K` può essere 1, 3 o 5. Con `K=3` una patch completa contiene il sito centrale e otto celle circostanti; ai bordi può contenere meno osservazioni.
- Celle assenti: indice -1, valore zero dopo la normalizzazione, maschera binaria separata. Non vengono sostituite da vicini più lontani.
- La maschera geografica non è una procedura di imputazione di misure temporali mancanti: il percorso pubblico di training richiede dati finiti e griglia temporale oraria contigua.
- Normalizzazione e prefisso temporale del test sono ereditati: min–max delle feature sul training e coda storica contigua usata come contesto.
- Batch: recente `[B,1,F,K,K]`, trend centrale `[B,168,F]` nel default, maschera `[B,1,K,K]`, calendario `[B,31]`, target meteorologico `[B,F,K,K]`.
- Le convoluzioni ricevono `F+1` canali per includere la maschera; per le tre variabili PVGIS il primo ingresso CNN ha quattro canali.

## 4. Modello o algoritmo

- `SpatialEncoder`: convoluzioni ordinarie 3×3 con padding 1 e ReLU; default due strati da 32 canali.
- Generatore: CNN sul passo recente; LSTM a due strati e hidden 64 sul trend del sito centrale; proiezione del calendario a 64 dimensioni. Le rappresentazioni temporale e di calendario sono replicate sulle celle della patch.
- I rami vengono concatenati e trasformati da convoluzione 1×1 e tanh. Le uscite delle celle assenti vengono azzerate.
- Discriminatore: CNN sulla storia recente, mascheramento e proiezione delle celle appiattite; convoluzione 1×1 sul dato corrente più maschera, seguita da massimo sulle sole celle valide; MLP sigmoidale finale.
- Il contesto recente è vincolato a un passo. La sequenza del discriminatore contiene quel passo e il dato corrente reale o generato.
- Non è costruita né consumata una matrice di adiacenza dal modello CNN. L'audit geografico confronta eventualmente gli insiemi di celle complete con i vicini più prossimi, senza usare tale confronto nel forward.
- Non sono partial convolutions: la maschera è un canale di ingresso. Le normali convoluzioni possono propagare attivazioni attraverso posizioni vuote negli strati intermedi.
- Loss del discriminatore invariata: media delle BCE con reale=0 e generato=1.
- Loss del generatore: BCE verso reale=0 più errore quadratico mascherato pesato, con peso 500 nel default.
- `masked_cell_mean` divide la somma degli errori per numero di celle valide × feature, separatamente per ogni patch, quindi il training media sulle patch. Una patch incompleta non pesa meno solo perché contiene meno celle.
- Due ottimizzatori Adam, learning rate 0.001, sei epoche e batch 256 sono i default; non sono risultati o configurazioni effettivamente eseguite.
- Score generativo: errore quadratico medio sulle celle valide. Score discriminativo: differenza reale–generato. Normalizzazione min–max globale sul test e somma sono mantenute.
- Anche ranking top-K e gestione stabile dei pareggi sono mantenuti. Le diagnostiche per feature sono estratte dalla cella centrale.
- `compare_stgan_exports` confronta i flag salvati sulle coordinate comuni, verifica la corrispondenza geografica dei siti e restituisce concordanze, discordanze e Jaccard delle anomalie. Tiene separate patch complete, incomplete e siti presenti soltanto nella baseline.
- Tale confronto usa decisioni salvate senza ricalcolo del ranking né esclusioni ulteriori; è un protocollo diverso dalle analisi che ricalcolano il top-K dopo esclusioni.

## 5. Elementi nuovi introdotti

- `SpatialGrid`, `STGANCNNConfig`, `SpatialEncoder`, `masked_cell_mean` e `compare_stgan_exports`.
- Sostituzione dei GCGRU recenti di generatore e discriminatore.
- Sostituzione della convoluzione finale del generatore e di quella corrente del discriminatore con proiezioni convoluzionali.
- Mantenimento della LSTM lunga, del calendario, delle convenzioni avversarie e della definizione combinata dello score.
- Gestione esplicita dei bordi mediante maschera e valutazione separata.
- `parameter_counts` consente il conteggio dei parametri; non impone uguaglianza con STGAN originale.
- I parametri rendono configurabili profondità, canali, hidden e ampiezza spaziale. Nei file Python rilevanti non è definita una procedura completa di ricerca e selezione automatica della configurazione migliore.

## 6. Informazioni utili per la tesi

### Metodi

- Doppia provenienza: base Git delle analisi e importazione del detector originale.
- Reticolo verificato, CRS, orientamento e patch geografiche.
- Maschera come canale, zero dopo normalizzazione e loss sulle celle valide.
- Componenti sostituite e componenti temporali conservate.
- Distinzione fra patch complete/incomplete e confronto degli insiemi di input.
- Protocollo di score e confronto delle decisioni sulle coordinate comuni.

### Related Work

- CNN su reticoli spaziali e condivisione dei kernel.
- CNN–LSTM per serie spazio-temporali.
- Gestione del padding e maschere di validità.
- GAN per anomaly detection.
- Studi di ablazione e confronto a capacità parametrica controllata.
- Jaccard e accordo fra detector senza ground truth.

## 7. Aspetti da chiarire

- I soli sorgenti non dimostrano che tutte le coordinate reali formino il reticolo configurato, né quanti siti abbiano patch complete.
- La coincidenza dei nove siti con il kNN originale deve essere verificata sull'input; ai bordi il criterio spaziale cambia e i vicini mancanti non vengono rimpiazzati.
- Conservare i siti non equivale a conservare l'operatore: ordinamento direzionale e kernel CNN sostituiscono il mixing sul grafo.
- Il numero di parametri dipende dalla configurazione; non si assume una parità esatta.
- Velocità, memoria, stabilità e qualità del detector richiedono misure sperimentali.
- Un alto accordo con il detector precedente non è una misura di accuratezza rispetto ad anomalie vere.
- La scelta della configurazione e di un eventuale obiettivo di validazione per la ricerca dei parametri resta da specificare.
- La normalizzazione sul test e il top-K globale conservano il carattere retrospettivo del protocollo originale.

## 8. Sintesi

- Variante STGAN con CNN spaziale e LSTM temporale conservata.
- Patch orientate su griglia verificata, con maschera delle celle assenti.
- Loss e score adattati alle sole celle valide.
- Confronto dei detector per coordinate comuni e condizione dei bordi.
