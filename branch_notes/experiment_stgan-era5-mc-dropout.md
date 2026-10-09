# Branch: experiment/stgan-era5-mc-dropout

## 1. Scopo dello step

- Funzione principale: trasformare il detector ConvGRU–LSTM su ERA5 in un detector con dropout nel generatore e stima Monte Carlo dello score (media e deviazione standard per coppia tempo–località), con fusione delle due componenti secondo lo schema del paper STGAN.
- Attorno a questo nucleo il branch aggiunge: split train/validation/test, strumenti di monitoraggio della GAN senza label, due varianti dell'input (codifica temporale ciclica, normalizzazione stagionale), una normalizzazione climatologica post-hoc degli score e una variante di training sul campo intero (`full_grid`).
- Posizione: prosecuzione della linea ERA5 con encoder spaziale convoluzionale. È il ramo parallelo a `experiment/stgan-era5-gat-mc-dropout`, che usa un GAT nel generatore; i due rami non sono uno la continuazione dell'altro.
- Revisione analizzata: `728ebe7a`, 8 ottobre 2026; riferimento locale e `origin` coincidenti.
- Predecessore effettivo: `experiment/stgan-era5-cube` a `19858ab4`, antenato diretto. Confronto `19858ab4..728ebe7a`, 27 commit.
- Differenza rispetto allo step precedente: il predecessore non ha dropout, non ha campionamento MC, non ha uno split di validation e produce uno score deterministico. Download, feature ERA5 ed estrazione degli eventi sono ereditati.
- Esclusi dall'analisi: notebook, sweep, documentazione, log e risultati. Nessun esperimento è stato eseguito per questa nota.

## 2. File di codice rilevanti

Percorsi riferiti a `728ebe7a`.

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/anomaly_detection/stgan/model.py` | Reti G e D | Tre dropout nel generatore; separazione `encode`/`decode`; uscita di D come logit o probabilità; attivazioni del penultimo layer di D; `score_draws` per i campioni MC. |
| `physiq_pv/anomaly_detection/stgan/training.py` | Passo di ottimizzazione | Numero di step D e G per batch configurabile; termini separati delle loss; percorso BF16 con BCE sui logit. |
| `physiq_pv/anomaly_detection/stgan/scoring.py` | Score e incertezza | Campionamento MC, momenti delle componenti, score "paper" con min–max globale sul test, deviazione standard MC. |
| `physiq_pv/anomaly_detection/stgan/pipeline.py` | Orchestrazione | Modalità di score, periodo di validation, ripresa da checkpoint, tassi di apprendimento distinti, monitor per epoca, scelta fra modalità `patch` e `full_grid`. |
| `physiq_pv/anomaly_detection/stgan/config.py` | Configurazione | Nuovi campi: dropout e MC, tassi G/D, rapporto di aggiornamento D:G, peso della ricostruzione, validation, codifica temporale, normalizzazione, precisione, `cnn_training_mode`. |
| `physiq_pv/anomaly_detection/stgan/data.py` | Finestre | `recent_steps` configurabile, codifica temporale ciclica per località, `ContextArray` per concatenare partizioni su disco. |
| `physiq_pv/anomaly_detection/stgan/seasonal.py` | Normalizzazione stagionale | Nuovo: climatologia per località, giorno dell'anno e ora, stimata sul solo training. |
| `physiq_pv/anomaly_detection/stgan/monitoring.py` | Monitoraggio per epoca | Nuovo: sottoinsieme fisso di validation, discrepanza delle feature di D, stabilità dello score, MMD sulle feature di D. |
| `physiq_pv/anomaly_detection/stgan/pca_reference.py` | Spazio PCA di riferimento | Nuovo: PCA esatta dei campi di training, salvata e mai rifittata. |
| `physiq_pv/anomaly_detection/stgan/mmd.py` | MMD in spazio PCA | Nuovo: sottoinsiemi fissi di validation, banda congelata, MMD multiscala fra osservazioni e ricostruzioni. |
| `physiq_pv/anomaly_detection/stgan/objective.py` | Statistiche di validation | Nuovo: errore di ricostruzione medio MC su tutta la validation, calcolato a fine training. |
| `physiq_pv/anomaly_detection/stgan/full_grid.py` | Variante sul campo intero | Nuovo: dataset per timestamp, generatore sul reticolo completo, discriminatore locale applicato a ogni cella. |
| `physiq_pv/anomaly_detection/stgan/precision.py` | Precisione numerica | Nuovo: FP32 oppure BF16 nativo con pesi FP32. |
| `physiq_pv/era5/data.py` | Preparazione ERA5 | Tre partizioni (`ERA5Cubes`); compatibilità con cache scritte prima dello split di validation. |
| `physiq_pv/era5/seasonality.py` | Score e stagionalità | Nuovo: z-score climatologico leave-one-year-out; distribuzione stagionale delle celle sopra soglia. |
| `physiq_pv/era5/cube.py`, `physiq_pv/era5/events.py` | Cluster ed eventi | Incertezza MC annotata su cluster ed eventi; non entra in soglia né collegamenti. |
| `scripts/run_era5_stgan.py` | Comandi | Nuovi sottocomandi `pca-reference`, `mmd-reference`, `climatology`; nuove opzioni di `train`; `events` su componente selezionabile. |

`physiq_pv/experiments/stgan_wandb.py`, `scripts/run_stgan_wandb.py` e `scripts/create_stgan_sweep.py` collegano il tracking e non introducono metodo. `physiq_pv/era5/visualization.py` contiene solo grafici.

## 3. Dati e preprocessing

- Input: le 15 feature ERA5 su griglia 0.5° ogni tre ore, ereditate senza modifiche.
- Split di default della preparazione: training 1980–2003, validation 2004, test dal 2005. Il comando di training richiede una cache con questi tre confini.
- Con `--validation-holdout` il 2004 resta fuori dal training e viene letto solo dal monitoraggio per epoca e dalle statistiche di validation. Senza l'opzione il training usa 1980–2004.
- La validation non stima intervalli dello score, soglie o normalizzazioni del test.
- Normalizzazione di default: min–max per feature stimato sul training, in `[-1,1]`.
- Normalizzazione `seasonal` (opzionale): per ogni località, feature, ora del giorno e giorno dell'anno si stimano media e deviazione standard sul training, raggruppando una finestra circolare di ±15 giorni (default). La deviazione è calcolata attorno alla media lisciata di ciascun giorno, con un minimo pari a 0.01 volte la deviazione complessiva della feature. Dopo la standardizzazione si applica lo stesso min–max.
- Codifica temporale `onehot` (default): 7 giorni della settimana + 24 ore, vettore di 31 elementi per timestamp.
- Codifica temporale `cyclic` (opzionale): seno e coseno dell'ora solare locale (ora UTC + longitudine/15) e seno e coseno della posizione nell'anno; 4 elementi, diversi per località.
- Target: i valori normalizzati correnti, non la produzione fotovoltaica.
- Contesto: validation e test sono preceduti dalla storia contigua necessaria al trend; i target restano nel proprio periodo.
- Tensori in modalità `patch`: recente `[B,T,15,3,3]`, trend `[B,56,15]`, maschera `[B,1,3,3]`, calendario `[B,31]` oppure `[B,4]`, osservato `[B,15,3,3]`. `T` è `recent_steps` (default 1).
- Tensori in modalità `full_grid`: recente `[B,T,15,H,W]`, trend `[B,56,N,15]`, maschera `[B,1,H,W]`, calendario `[B,31]` oppure `[B,N,4]`, osservato `[B,15,H,W]`. Qui `B` conta timestamp.
- Le celle del reticolo senza località valgono zero dopo la normalizzazione e hanno maschera 0, in entrambe le modalità.

## 4. Modello o algoritmo

### Generatore e discriminatore

- Generatore: ConvGRU sui passi recenti, LSTM sul trend, proiezione lineare con ReLU del calendario. Le tre rappresentazioni passano ciascuna per un dropout (spaziale, temporale) e la loro concatenazione per un terzo (fusione); segue una convoluzione 1×1 con `tanh`. Probabilità di default 0.2.
- Discriminatore: architettura invariata rispetto al predecessore e senza dropout. Può restituire il logit invece della probabilità; le attivazioni del penultimo layer sono esposte per il monitoraggio.
- Convenzione dell'uscita di D: probabilità di anomalia, target 0 per il dato reale e 1 per il generato.

### Funzioni di perdita e training

- D: `0.5 · [BCE(D(reale), 0) + BCE(D(generato), 1)]`.
- G: `w · MSE mascherato + BCE(D(generato), 0)`, con `w = 500` di default, configurabile.
- La riduzione dell'MSE è la media su celle valide e feature per campione, poi la media sui campioni.
- Due ottimizzatori Adam. Il tasso di D si esprime come rapporto rispetto a quello di G oppure come valore indipendente; le due forme sono mutuamente esclusive.
- Rapporto di aggiornamento `D:G` (default `1:1`): numero di step di ottimizzazione di ciascuna rete sullo stesso batch, prima tutti quelli di D e poi tutti quelli di G, ognuno con un proprio forward.
- Il passo di D e il passo di G usano forward indipendenti del generatore.
- Campionamento del training: prodotto completo e rimescolato tempo × località, oppure campionamento con reinserimento se è fissato un numero di campioni per epoca.
- Checkpoint a ogni epoca con stato dei due ottimizzatori; ripresa ammessa solo con configurazione, normalizzazione e riferimento MMD identici.
- Precisione: FP32 oppure BF16 (autocast sui forward, BCE sui logit, pesi FP32).

### Score con MC dropout

- In scoring il modello è in valutazione, salvo i dropout del generatore che restano attivi.
- Per ogni batch si estraggono `M` campioni (default 20). Encoder di G, storia di D e `D(reale)` non dipendono dal dropout e sono calcolati una volta sola.
- Componenti per coppia tempo–località:
  - `sG`: errore quadratico medio sulle celle valide della patch;
  - `sD = D(reale) − D(generato)`;
  - errori quadratici per feature della cella target (diagnostici).
- Modalità `paper`, usata dal comando ERA5: si calcolano le mappe medie MC di `sG` e `sD`; minimo e massimo di ciascuna sono presi sull'intero prodotto tempo × località del test; lo score è `minmax(sG) + minmax(sD)` con peso 1. Gli stessi intervalli trasformano ogni campione MC e la deviazione standard di popolazione dei campioni è l'incertezza dello score.
- Con `M = 1` la formula coincide con lo score deterministico.
- Il codice marca questa normalizzazione come transduttiva: usa tutti gli score del test, senza label.
- Modalità `components`: momenti delle due componenti e loro covarianza, senza fusione. Modalità `calibrated`: intervalli stimati su un periodo di calibrazione, rimasta per il percorso PVGIS.

### Monitoraggio e selezione senza label

Nessuna di queste quantità entra in una loss, in un passo di ottimizzazione, nello score o nella valutazione sul test.

- Sottoinsieme di monitoraggio: 32 timestamp di validation equispaziati (default) per tutte le località; un forward deterministico con dropout spento.
- Per epoca: differenza assoluta media fra le feature del penultimo layer di D su osservazione e ricostruzione; variazione dello score fra epoche consecutive (media, mediana, correlazione di Spearman); MMD² non distorta con kernel RBF sulle stesse feature, banda dall'euristica della mediana.
- Riferimento PCA: PCA esatta (autodecomposizione della matrice di Gram) su al più 4096 campi di training equispaziati, normalizzati e appiattiti; si salvano le componenti principali e la varianza spiegata. Viene costruito una volta, prima delle run, e solo caricato.
- Riferimento MMD: sottoinsiemi fissi di timestamp di validation (default 5 da 512, disgiunti se possibile), loro feature PCA e banda `σ` pari alla distanza euclidea mediana fra quelle feature.
- Per epoca: MMD² non distorta fra osservazioni e ricostruzioni di G nello spazio PCA, con kernel pari alla media di tre RBF di banda `0.5σ`, `σ`, `2σ`; media e varianza sui sottoinsiemi; media mobile sulle ultime 5 epoche (default) come quantità di selezione del modello.
- Dopo l'ultima epoca: media, deviazione standard, mediana e 95° percentile dell'errore di ricostruzione medio MC su tutti i punti validi della validation.
- `--skip-final-scoring` termina dopo training e validation, senza score del test.

### Dagli score agli eventi

- L'algoritmo di soglia, morfologia e collegamento è quello del predecessore.
- La deviazione standard MC viene ricostruita sul reticolo e riportata per cluster (media e massimo) e per evento (media pesata sulle celle); non partecipa a soglia, morfologia o collegamenti.
- `events` può operare sullo score combinato oppure su una singola componente.
- `climatology` (post-hoc): ogni score diventa uno z-score rispetto a media e deviazione della stessa località, mese e ora UTC negli altri anni (leave-one-year-out); gruppi con meno di 30 valori o varianza nulla diventano NaN. Gli eventi possono essere estratti da questi z-score.

### Variante `full_grid`

- Scelta esplicita `cnn_training_mode`: `patch` (default) oppure `full_grid`.
- Unità del campione: in `patch` una cella target con la sua patch 3×3; in `full_grid` l'intero campo di un timestamp.
- Stessi moduli e stessi nomi dei parametri del modello a patch.
- Generatore: la ConvGRU scorre sull'intero reticolo, `[B,T,F,H,W] → [B,F,H,W]`. LSTM di trend e codifica temporale sono calcolati per cella e poi disposti sul reticolo.
- Discriminatore: resta locale. La patch di ogni cella viene estratta dal campo e il discriminatore a patch le valuta tutte in una passata, con le stesse maschere. Non è una convoluzione che scorre sul campo: la ConvGRU di D ha padding interno alla patch.
- Loss: ricostruzione come media su celle con località e feature del campo, poi sui timestamp; avversaria come media della BCE sugli output locali, uno per cella.
- Score: stesse definizioni per coppia tempo–località; `sG` è l'errore medio sulla patch 3×3 della cella calcolato sul campo ricostruito.
- Step di ottimizzazione per epoca: `ceil(T·N / batch)` in `patch`, `ceil(T / batch)` in `full_grid`, con `T` timestamp utili e `N` località.
- La ripresa da un checkpoint dell'altra modalità è rifiutata.

## 5. Elementi nuovi introdotti

- Classi: `ERA5Cubes`, `ContextArray`, `SeasonalClimatology`, `ValidationMonitor`, `PCAReference`, `MMDReference`, `MMDMonitor`, `STGANFullGridDataset`, `GridLayout`, `FullGridGenerator`, `FullGridDiscriminator`, `FullGridSTGAN`.
- Funzioni: `score_draws`, `summarize_raw_mc_components`, `normalize_paper_mc_scores`, `fit_seasonal_climatology`, `cyclic_time_features`, `annual_phase_features`, `build_pca_reference`, `fit_pca`, `build_mmd_reference`, `multiscale_mmd2`, `median_sigma`, `rbf_mmd2`, `score_stability`, `validation_objective`, `climatology_zscores`.
- Comportamenti modificati: lo score ERA5 passa da deterministico a media MC con deviazione standard; il passo di training accetta più step per rete e tassi distinti; `recent_steps` diventa un parametro; lo split passa da due a tre partizioni.
- Rimossi o sostituiti: il forward condiviso del generatore fra passo di D e passo di G (ora sempre indipendenti); il contenitore a due partizioni per ERA5.

## 6. Informazioni utili per la tesi

### Metodi

- Posizione dei tre dropout nel generatore e assenza di dropout nel discriminatore.
- Procedura MC: quali moduli restano stocastici, numero di campioni, cosa è condiviso fra i campioni.
- Definizione di `sG`, `sD` e dello score combinato con min–max globale sul test; definizione dell'incertezza come deviazione standard di popolazione.
- Carattere transduttivo del min–max: aggiungere anni al test può cambiare minimi e massimi.
- Split 1980–2003 / 2004 / 2005+ e ruolo della validation, limitato a monitoraggio e selezione.
- Loss di G e D, peso della ricostruzione, tassi separati, rapporto di aggiornamento D:G.
- Le due codifiche temporali e le due normalizzazioni, con la precisazione di quali statistiche sono stimate sul solo training.
- Monitoraggio senza label: feature di D, stabilità dello score, MMD in spazio PCA con riferimento fisso e banda congelata.
- Normalizzazione climatologica post-hoc e sua distinzione dalla normalizzazione stagionale dell'input.
- Uso dell'incertezza negli eventi: solo annotazione.
- Variante `full_grid`: unità del campione, numero di step per epoca, località del discriminatore, riduzione delle loss.

### Related Work

- STGAN e fusione delle componenti di ricostruzione e discriminazione.
- MC dropout come approssimazione dell'incertezza del modello.
- GAN con termine di ricostruzione e bilanciamento fra generatore e discriminatore (tassi distinti, rapporto di aggiornamento).
- Maximum Mean Discrepancy, kernel RBF, stima non distorta, euristica della mediana, kernel multiscala.
- Metriche di valutazione delle GAN basate su feature apprese.
- PCA per campi geofisici.
- Codifiche cicliche del tempo e ora solare locale.
- Climatologie, anomalie standardizzate e validazione leave-one-year-out.
- ConvGRU su griglie geografiche e discriminatori locali a patch.
- Precisione mista nel training di reti neurali.

## 7. Aspetti da chiarire

- Quale configurazione rappresenta lo step nella tesi: `recent_steps`, codifica temporale, normalizzazione, modalità `patch` o `full_grid`, numero di epoche. I default del codice non lo stabiliscono.
- Numero di componenti PCA effettivamente usato: il default del codice è 100, il valore è un argomento dei comandi.
- Motivazione del peso 500, di `λ = 1`, di `p = 0.2`, di 20 campioni MC e della finestra stagionale di 15 giorni.
- Se la deviazione standard MC sia calibrata come misura di incertezza: il codice la calcola, non la valida.
- Relazione fra normalizzazione stagionale dell'input e z-score climatologico post-hoc, che rimuovono ciclicità in due punti diversi della catena.
- Confrontabilità fra `patch` e `full_grid` a parità di epoche, dato il diverso numero di step di ottimizzazione.
- Ruolo effettivo della MMD in spazio PCA nella selezione: il codice la espone come obiettivo, la scelta dipende dagli sweep, esclusi da questa analisi.
- Stato intermedio del branch: al primo commit lo score ERA5 usava intervalli stimati su un periodo di calibrazione 2003–2004; è stato poi sostituito dallo score `paper` con validation 2004. Va deciso se menzionarlo.
- Esiti: nessun training completato, score o evento reale è attestato dal codice.

## 8. Sintesi

- Il detector ConvGRU–LSTM su ERA5 acquisisce dropout nel generatore e score MC con media e deviazione standard.
- Lo score combinato segue lo schema del paper: min–max globale sul test delle due componenti, somma con peso 1.
- Si aggiungono validation 2004, monitoraggio senza label (feature di D, MMD in spazio PCA) e controlli sul bilanciamento G/D.
- Opzioni di input: tempo ciclico locale e normalizzazione stagionale; post-hoc: z-score climatologico.
- Variante `full_grid`: campo intero come campione, discriminatore ancora locale 3×3.
