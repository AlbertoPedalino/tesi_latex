# Branch: experiment/stgan-cnn

## 1. Scopo dello step

- Funzione principale: adattare il detector STGAN a patch geografiche regolari, sostituendo le operazioni sul grafo con operazioni convoluzionali e conservando il ramo LSTM di lungo periodo e il protocollo avversario.
- Posizione nel percorso: detector meteorologico distinto dal previsore fotovoltaico SDE; le decisioni del detector possono essere usate per stratificare gli errori del previsore.
- Revisione analizzata: `4410e3e9`, 18 settembre 2026, riferimenti `experiment/stgan-cnn` e `origin/experiment/stgan-cnn` coincidenti nella copia consultata.
- Predecessore effettivo della linea: `feat/anomaly-spatial-threshold-comparison` a `6356f1f9`, genitore del primo commit CNN `28e532c1`. Confronto complessivo: `6356f1f9..4410e3e9`.
- Confronto per aggiornare la nota esistente: `28e532c1..4410e3e9`. Il detector su grafo di `origin/feat/stgan-paper` a `777df6bc` è il riferimento metodologico importato, non il genitore Git.
- Differenza essenziale rispetto alla nota: il modello finale usa **ConvGRU + LSTM**, mentre `experiment_stgan-cnn.md` descrive ancora lo `SpatialEncoder` CNN senza ricorrenza. La nota è riferita al commit del 9 settembre, consultato il 10 settembre.

### Perimetro e verifica degli altri branch

Analisi del repository locale `AlbertoPedalino/physiq-pv`, indicato e confermato durante la richiesta. Finestra aggiornata richiesta: **28 agosto–18 settembre 2026**, ultime tre settimane. Sono stati consultati Git in sola lettura e sorgenti Python; le note Markdown esistenti sono state lette soltanto come oggetto del controllo di aggiornamento. Nessun notebook, sweep, risultato, checkpoint o dato sperimentale è stato usato come evidenza. Nessun esperimento o test è stato eseguito.

I riferimenti `origin/*` sono quelli già presenti localmente: non è stato eseguito un fetch. La verifica non certifica l'assenza di commit più recenti sul server remoto.

| Branch | Revisione disponibile | Predecessore effettivo per lo step | Copertura nelle note esistenti |
|---|---|---|---|
| `experiment/stgan-cnn` | `4410e3e9`, 18/09 | `6356f1f9`; primo commit `28e532c1` | **Da aggiornare**: nota ferma a `28e532c1`. |
| `experiment/stgan-era5-cube` | `98958b04`, 18/09 | `4410e3e9`, genitore diretto | **Nota assente**: documentato nella seconda parte di questo file. |
| `ablation/no-gat` | `6bda0521`, 09/09 | `28e532c1`, genitore diretto | `ablation_no-gat.md` cita già questa revisione. |
| `feat/anomaly-spatial-threshold-comparison` | `6356f1f9`, 09/09 | `657a3fa8`; primo commit successivo `0a0444f1` | `feat_anomaly-spatial-threshold-comparison.md` cita già questa revisione. |
| `feat/sde-net-paper-faithful` | `7998a75b`, 03/09 | `91753344` prima dell'introduzione `4dc591d8`; aggiornamenti recenti rispetto a `90075d68` | `feat_sde-net-paper-faithful.md` cita già questa revisione e distingue le due basi. |
| `feat/sde-direct-multihorizon-t1-t6` | `657a3fa8`, 30/08 | `90075d68`, genitore di `7e51c560` | `feat_sde-direct-multihorizon-t1-t6.md` cita già questa revisione. |
| `feat/sde-stgan-events-t6` | `bdfff424`, 30/08 | `4233cff6`, genitore diretto | `feat_sde-stgan-events-t6.md` cita già questa revisione. |
| `feat/stgan-paper` | remoto locale `777df6bc`, 30/08; branch locale `47b58e44`, 29/08 | `85d799b6`, genitore di `004bb756` | `feat_stgan-paper.md` cita già entrambe le revisioni; `47b58e44..777df6bc` non modifica Python. |

Per i sei branch già coperti, la coincidenza della revisione significa che non è emerso un nuovo delta da aggiungere rispetto allo stato documentato; non equivale a certificare ogni frase delle rispettive note. La nota no-GAT distingue già correttamente l'assenza di message passing dalla diffusione SDE condivisa tramite pooling sui nodi.

Le ablation kernel 1×1 e 5×5 sono configurazioni del branch CNN, non branch separati fra i riferimenti trovati. La disponibilità del codice non dimostra che le run siano state completate.

## 2. File di codice rilevanti

Tutti i percorsi della tabella sono riferiti a `4410e3e9`. Il delta di aggiornamento è `28e532c1..4410e3e9`.

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/anomaly_detection/stgan/model.py` | Generatore e discriminatore | Sostituisce `SpatialEncoder` con `ConvGRUCell` e `ConvGRU`; introduce kernel configurabile, mascheramento dello stato e condivisione del calcolo della storia nel discriminatore. |
| `physiq_pv/anomaly_detection/stgan/config.py` | Contratto di configurazione | Aggiunge `kernel_size` indipendente da `patch_size`; permette `recent_steps >= 1` con `trend_steps >= recent_steps`; distingue parametri scientifici e opzioni di esecuzione. |
| `physiq_pv/anomaly_detection/stgan/grid.py` | Reticolo e patch | Ereditato rispetto alla nota: CRS, orientamento geografico, indici e maschera. Serve a distinguere supporto osservato e kernel. |
| `physiq_pv/anomaly_detection/stgan/data.py` | Cubi, finestre e normalizzazione | Aggiunge estrazione vettoriale dei batch, contesto virtuale e cache normalizzata su disco; preserva la struttura dei campioni. |
| `physiq_pv/anomaly_detection/stgan/training.py` | Aggiornamento GAN | Estrae `gan_train_step`: D prima di G; riutilizza il forward del generatore e la storia del discriminatore solo quando compatibile con l'aggiornamento dei pesi. |
| `physiq_pv/anomaly_detection/stgan/loading.py` | Caricamento dei batch | Separa caricamento vettoriale e scalare e gestione dei worker; nessuna nuova loss. |
| `physiq_pv/anomaly_detection/stgan/sampling.py` | Ordine dei campioni | Introduce `EpochShuffleSampler`: modalità `legacy`, `global` e `block`, con differenze esplicite nell'ordine. |
| `physiq_pv/anomaly_detection/stgan/scoring.py` | Inferenza e score | Introduce `ScoreStore`, `score_components` e normalizzazione a blocchi con estremi globali sul test. |
| `physiq_pv/anomaly_detection/stgan/ranking.py` | Decisioni top-K | Ranking stabile in memoria o tramite ordinamento esterno; conteggio globale esatto e riepiloghi per bordi/interno. |
| `physiq_pv/anomaly_detection/stgan/pipeline.py` | Coordinamento | Collega ConvGRU, training e scoring; registra kernel e finestre nella configurazione; il loader rifiuta il vecchio formato del modello CNN senza ricorrenza. |
| `physiq_pv/anomaly_detection/stgan/result.py` | Risultati in memoria/disco | Aggiunge gestione esplicita della durata delle risorse di scoring. |
| `scripts/run_pvgis_stgan.py` | Interfaccia del detector | Espone `--kernel-size`; separa batch di training e inferenza; esporta decisioni e diagnostiche a blocchi. |
| `physiq_pv/reporting/detector_spatial_frequency.py` | Distribuzione geografica delle decisioni | Nuove frequenze per sito e copertura; filtro diurno applicato dopo le decisioni; confronto con regole MTGFlow esplicite. |
| `physiq_pv/reporting/mtgflow_top_percent.py` | Protocollo comparativo esplorativo | Top-percent globale della coordinata firmata `(score - soglia_salvata) / IQR_train`. |
| `physiq_pv/reporting/stgan_regional.py` | Aggregazione regionale | Generalizza la precedente API STGAN a entrambi i detector, conservando le decisioni puntuali. |
| `physiq_pv/reporting/stgan_meteo_errors.py` | Errori per tipo meteorologico | Interseca anomalie STGAN con condizioni climatologiche direzionali; distingue alto/basso, sovrapposizioni e anomalie non tipizzate. |
| `physiq_pv/reporting/detector_threshold_sensitivity.py` | Sensibilità alla soglia | Aggiunge condizioni meteorologiche e statistiche per sottogruppo al variare della soglia. |
| `physiq_pv/reporting/pointwise_detector_posthoc.py` | Join previsione–detector | Aggiunge riuso verificato delle valutazioni; il delta non introduce una nuova definizione dell'anomalia. |
| `tests/test_stgan_cnn.py`, `tests/test_stgan_performance.py` | Specifiche eseguibili | Contengono controlli di maschera, kernel, ordine dei campioni, equivalenza dei percorsi e ranking. Consultati come codice, senza esecuzione né uso delle parti che richiedono notebook/sweep. |

Le rimozioni dei vecchi moduli `anomaly_driver.py`, `anomaly_extremes.py`, `event_onset.py`, `hourly_event_response.py` e `report_dump.py` compaiono nel delta. Non vengono interpretate come prova dell'abbandono di una linea scientifica o come risultato sperimentale; non costituiscono modifiche al backbone del previsore.

## 3. Dati e preprocessing

- Input PVGIS del detector: irradianza POA, temperatura a 2 m e velocità del vento a 10 m; coordinate per la griglia. La produzione fotovoltaica appartiene al percorso separato di previsione e valutazione.
- Cubi `[T,N,F]`, con `F=3` nel percorso PVGIS; campioni individuati dalla coppia istante target–sito centrale.
- Target del generatore: le feature meteorologiche osservate al passo corrente per tutte le celle valide della patch. Non sono etichette manuali normale/anomalo.
- Normalizzazione per feature tramite minimo e massimo del solo training, mappata a `[-1,1]`; nessun clipping del test è introdotto dall'estrazione delle finestre.
- Il test riceve la coda del training come contesto iniziale; i target valutati rimangono nel test. La pipeline di questo branch richiede una cadenza oraria contigua.
- Griglia predefinita: CRS `EPSG:32632`, passo 5.000 m, tolleranza 25 m; righe orientate verso sud, colonne verso est.
- Le celle geografiche assenti sono mascherate e azzerate dopo normalizzazione. Questa maschera non autorizza misure temporali mancanti nelle celle osservate.
- Notazione: `P` è il lato della patch, `k` il kernel convoluzionale, `R` i passi recenti, `L` la lunghezza del trend. **P e k sono parametri diversi.**
- Forme: recente `[B,R,F,P,P]`; trend centrale `[B,L,F]`; maschera `[B,1,P,P]`; calendario `[B,31]`; target e predizione `[B,F,P,P]`.
- Default: `P=3`, `k=3`, `R=1`, `L=168`. Calendario one-hot: 7 giorni della settimana e 24 ore.
- La cache normalizzata e i batch vettoriali cambiano la modalità di accesso ai dati; la selezione delle finestre e la formula di normalizzazione rimangono esplicite nel codice.

## 4. Modello o algoritmo

### Architettura finale e differenza rispetto alla nota

- `ConvGRUCell` calcola gate reset/update e candidato tramite `Conv2d`; ogni gate riceve valori, maschera e stato nascosto. Il padding è `k // 2`.
- `ConvGRU` azzera lo stato all'inizio di ogni finestra e applica gli strati per ciascun passo recente. Non conserva stato tra batch o finestre.
- Il generatore combina ConvGRU della patch recente, LSTM del trend del solo sito centrale e proiezione del calendario. Trend e calendario sono replicati spazialmente; una proiezione 1×1 con tanh produce le feature della patch.
- Il discriminatore codifica la storia con ConvGRU, appiattisce tutta la patch e la proietta con un layer lineare. Il dato corrente passa per proiezione 1×1 e massimo sulle celle valide; un MLP produce l'uscita sigmoidale.
- Lo stato delle celle assenti viene azzerato dopo ogni aggiornamento ConvGRU. La vecchia descrizione della propagazione di attivazioni attraverso celle vuote negli strati dello `SpatialEncoder` non descrive più correttamente il modello finale.
- Non viene usata una matrice di adiacenza nel forward.
- Implicazione diretta del codice: con `recent_steps=1`, la ConvGRU parte da stato zero ed esegue un solo aggiornamento per finestra; non introduce una lunga ricorrenza temporale nel ramo recente. Il contesto lungo rimane nella LSTM. Il parametro ora consente anche finestre recenti più lunghe, senza dimostrare che siano state usate.

### Ablation 1×1, 3×3 e 5×5

| Configurazione | Effetto ricavabile dal codice | Interpretazione da evitare |
|---|---|---|
| `kernel_size=1` | I gate ConvGRU elaborano ogni posizione senza convoluzione fra celle diverse. | Non equivale a eliminare ogni dipendenza spaziale del detector: discriminatore, riduzioni degli errori e supporto della patch restano. |
| `kernel_size=3` | Kernel di riferimento nei gate di generatore e discriminatore. | Non dimostra che la configurazione sia ottimale. |
| `kernel_size=5` | Gate con kernel più ampio e padding 2, sempre sul supporto fornito dalla patch. | Con `patch_size=3` i siti osservati rimangono al massimo nove; non si osservano automaticamente 25 siti. |

- Le proiezioni finali 1×1 del generatore e del discriminatore restano 1×1 per tutte le ablation; `kernel_size` modifica i gate ConvGRU.
- Variare il kernel cambia anche il numero di parametri: non è un confronto a capacità parametrica automaticamente pareggiata.
- Variare `patch_size` cambia invece il supporto osservato e la dimensione della proiezione della storia nel discriminatore. Non va confuso con la sola ablation del kernel.
- `parameter_counts`, configurazione del modello e metadati permettono di registrare la variante. Nessun conteggio o risultato di una run è stato letto.

### Loss, training e inferenza

- Discriminatore: `0.5 * (BCE(D_reale, 0) + BCE(D_generato, 1))`.
- Generatore: `BCE(D_generato, 0) + lambda * MSE_mascherato`, con `lambda=500` nel default.
- L'errore viene mediato per campione sulle celle valide e sulle feature, poi sui campioni. Una patch incompleta non viene pesata solo in proporzione al numero di celle.
- Due Adam, default learning rate 0.001, sei epoche e batch di training 256. Questi sono default del codice, non parametri osservati nelle run.
- L'aggiornamento è D poi G. Il percorso ottimizzato riusa la predizione del generatore, ma ricalcola la storia del discriminatore con i pesi aggiornati prima della loss di G.
- Il prodotto completo tempo–località rimane la modalità predefinita. Il campionamento con rimpiazzo è ancora un'opzione distinta.
- `shuffle_mode=global` mira a preservare l'ordine della modalità storica usando indici tensoriali; `block` preserva insieme e molteplicità dei campioni, ma cambia la distribuzione dell'ordine. Non è una differenza puramente di memoria.
- Inferenza in `eval()` e senza gradienti; batch di inferenza configurabile separatamente.
- Score generativo: MSE sulle celle valide della patch. Score discriminativo: `D_reale - D_generato`. Min–max separato per le due componenti sull'intero prodotto tempo–località del test, poi somma con peso unitario.
- Il calcolo a blocchi non trasforma la normalizzazione globale in normalizzazione per chunk.
- Ranking decrescente stabile, pareggi ordinati secondo l'indice originale; top-K con `ceil(N * percentuale / 100)`. Il percorso su disco conserva questa regola.
- Gli score per feature sono residui quadratici della cella centrale: non coincidono con una decomposizione completa dello score combinato, che contiene anche il discriminatore.

### Analisi successive aggiunte dal giorno 11 settembre

- Frequenze geografiche: anomalie divise per ore con score valido per sito; copertura riportata separatamente. Score mancanti non diventano decisioni normali.
- Le esclusioni degli istanti di caduta solare regionale e recupero sono applicate prima del ricalcolo del ranking STGAN. Il filtro diurno POA > 10 W/m², quando richiesto, seleziona successivamente le decisioni già calcolate: non ricalcola il top-1% sulle sole ore diurne.
- MTGFlow conserva normalmente la soglia salvata per sito. La variante esplorativa `rank_mtgflow_top_percent` usa invece il ranking globale della coordinata firmata `(score - soglia_salvata) / IQR_train`, senza troncamento a zero. Questa regola non coincide con la coordinata `(score - Q3_train) / IQR_train` usata nelle analisi di sensibilità.
- Gli estremi meteorologici sono definiti tramite flag climatologici e lato alto/basso rispetto alle bande della stessa riga. Le analisi degli errori intersecano tali flag con le anomalie STGAN alla stessa località e al timestamp target.
- Le categorie possono sovrapporsi; le anomalie non tipizzate non vengono riclassificate come normali. Le bande di RMSE fra siti sono quartili descrittivi, non intervalli di confidenza.
- La sensibilità meteorologica varia la soglia decisionale senza addestrare nuovamente il detector. Le motivazioni scientifiche della scelta delle soglie non sono dedotte dal codice.

## 5. Elementi nuovi introdotti

- **Da integrare nella nota:** `ConvGRUCell`, `ConvGRU`, `kernel_size`, storia recente configurabile, azzeramento degli stati mancanti, distinzione kernel/patch e relativo effetto sulla capacità.
- **Da integrare nel protocollo riproducibile:** `gan_train_step`, `EpochShuffleSampler`, batch di inferenza separato, `ScoreStore` e ranking esterno con normalizzazione globale preservata.
- **Da integrare nelle analisi comparative:** `build_detector_spatial_frequency`, `rank_mtgflow_top_percent`, `aggregate_detector_region`, `load_directional_meteo_flags`, `build_stgan_meteo_errors`, `attach_meteo_conditions` e `meteo_sensitivity_sweep`.
- **Sostituito:** `SpatialEncoder` CNN senza ricorrenza; il caricamento di quella vecchia architettura è rifiutato dal loader finale.
- **Conservati:** LSTM del trend, calendario, convenzioni BCE, peso di ricostruzione predefinito, score globale e target meteorologico.
- **Da non attribuire a questo step:** la rimozione del GAT nel previsore appartiene al branch fratello `ablation/no-gat`, staccato a `28e532c1`. Non è un aggiornamento incorporato automaticamente nel tip CNN.

## 6. Informazioni utili per la tesi

### Metodi

- Distinguere tre stati: STGAN su grafo, primo adattamento CNN e versione finale ConvGRU.
- Specificare separatamente patch, kernel, numero di strati/canali, finestra recente e trend.
- Descrivere maschere sia negli input sia negli stati ricorrenti e nelle loss.
- Esplicitare l'ablazione del mixing convoluzionale, evitando di chiamare il kernel 1×1 un modello completamente privo di informazione spaziale.
- Riportare il numero di parametri per le configurazioni realmente confrontate, senza presumere parità.
- Separare normalizzazione delle feature sul training e normalizzazione degli score sul test.
- Documentare ordine del campionamento, top-K e gestione dei pareggi.
- Separare decisioni native, ranking dopo esclusioni, filtro diurno e variante MTGFlow a budget globale.
- Documentare denominatori, allineamento località–timestamp–orizzonte, sottogruppi sovrapposti e significato delle bande descrittive.

### Related Work

- GAN per anomaly detection spazio-temporale.
- Ricorrenza con gate convoluzionali e confronto con operatori su grafo.
- LSTM per contesto temporale lungo.
- Receptive field, padding e supporto spaziale osservato.
- Ablazioni architetturali e controllo della capacità parametrica.
- Ranking di anomalie, selezione top-K e soglie basate su quantili/IQR.
- Valutazione condizionata a eventi meteorologici e confronti senza etichette indipendenti.

## 7. Aspetti da chiarire

- Quali run 1×1/3×3/5×5 siano state avviate e completate, con quali seed, patch, epoche e revisione: **non verificabile dal solo codice**.
- Se i risultati da confrontare appartengano tutti alla versione ConvGRU o includano la precedente CNN senza ricorrenza.
- Se i confronti mantengano costanti dati, patch, finestre e ordine dei campioni; il codice consente variazioni e non documenta da solo l'esecuzione effettiva.
- Quale protocollo di decisione venga scelto come principale rispetto alle varianti esplorative.
- Se e perché controllare il numero di parametri nelle ablation: la decisione metodologica non è stabilita dal nome del kernel.
- Le motivazioni di rimozioni di helper e riorganizzazioni non sono ricavabili dai messaggi generici dei commit.
- Il codice contiene test di equivalenza, ma in questa analisi non sono stati eseguiti; la loro presenza non viene presentata come prova di esito positivo su questo ambiente.
- Le sei note con revisione coincidente non risultano superate da nuovi commit disponibili, ma non sono state riscritte né rivalidate integralmente.
- Velocità, consumo di memoria, qualità del detector, significatività e superiorità di una variante richiedono evidenze sperimentali escluse dal controllo.

## 8. Sintesi

- La nota CNN esistente è superata: il tip usa ConvGRU con trend LSTM.
- Le ablation 1×1 e 5×5 sono implementate come kernel indipendenti dalla patch.
- Sampling, scoring a blocchi e ranking richiedono precisazioni di riproducibilità.
- Mancano nella nota anche nuove analisi geografiche e meteorologiche.
- Il completamento e gli esiti delle run rimangono non verificati.

# Branch: experiment/stgan-era5-cube

## 1. Scopo dello step

- Funzione principale: applicare il detector ConvGRU–LSTM a dati ERA5 e trasformare gli score geografici in cluster ed eventi collegati nel tempo.
- Posizione: estensione a un diverso dominio meteorologico e a un algoritmo di estrazione degli eventi; non è una nuova versione del previsore fotovoltaico.
- Revisione analizzata: `98958b04`, 18 settembre 2026.
- Predecessore effettivo: `experiment/stgan-cnn` a `4410e3e9`, genitore diretto. Confronto `4410e3e9..98958b04`.
- Nella cartella delle note consultata non è presente una nota dedicata a questo branch. È lo step recente interamente mancante rispetto all'inventario.
- Differenza: nuovo contratto di 15 feature, cadenza di tre ore, griglia angolare, preparazione dei cubi e postprocessing morfologico con grafo temporale degli eventi. Il modello neurale di base rimane ereditato.

## 2. File di codice rilevanti

Percorsi riferiti a `98958b04`.

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/era5/features.py` | Contratto delle feature | Definisce dieci variabili single-level e cinque a 850 hPa, alias e ordine condivisi. |
| `scripts/download_era5.py` | Acquisizione verificabile | Richieste mensili, selezione temporale/spaziale, verifica di variabili e coordinate, mantenimento dei file originali e ricevute di integrità. |
| `physiq_pv/era5/data.py` | Preparazione | Aggiunge `month_files`, `monthly_blocks`, `prepare_era5` e `load_prepared`; allinea i due prodotti e costruisce split su disco. |
| `physiq_pv/anomaly_detection/stgan/data.py` | Interfaccia comune | Generalizza il contenitore in `AlignedCubes`, mantenendo l'alias PVGIS. |
| `physiq_pv/anomaly_detection/stgan/grid.py` | Geometria | Consente passo angolare esplicito e audit kNN opzionale. |
| `physiq_pv/anomaly_detection/stgan/pipeline.py` | Training/scoring riusabili | Introduce cadenza configurabile, nome del dominio e griglia angolare, mantenendo loss e architettura. |
| `scripts/run_era5_stgan.py` | Fasi distinte | Espone `prepare`, `train`, `events`; configura trend di 56 passi e cadenza di tre ore. |
| `physiq_pv/era5/cube.py` | Score geografici e cluster | Introduce `CubeGrid`, mappatura degli score, percentile, morfologia e componenti connesse. |
| `physiq_pv/era5/events.py` | Collegamenti temporali | Introduce `EventConfig`, `link_clusters`, `process_events` e traiettorie; gestisce separazioni e fusioni. |
| `tests/test_era5_cube.py`, `tests/test_download_era5.py`, `tests/test_stgan_cnn.py` | Controlli del contratto | Test presenti nel delta; nessun test è stato eseguito per questa analisi. |

Le funzioni di visualizzazione non sono considerate nuove componenti del detector.

## 3. Dati e preprocessing

- Due prodotti: `reanalysis-era5-single-levels` e `reanalysis-era5-pressure-levels`; quest'ultimo richiesto soltanto a 850 hPa.
- Le dieci feature single-level sono: temperatura a 2 m, componenti u/v del vento a 10 m, pressione al livello medio del mare, precipitazione totale, copertura nuvolosa totale, radiazione solare discendente alla superficie, umidità volumetrica del primo strato del suolo, velocità d'attrito e altezza dello strato limite.
- Le cinque feature a 850 hPa sono: componenti u/v del vento, velocità verticale, temperatura e umidità specifica.
- Default del downloader: dal 1980 all'ultimo anno completo comune ai due prodotti; otto istanti UTC al giorno, ogni tre ore; griglia 0.5° × 0.5°; area `[60,-15,20,50]` in ordine nord, ovest, sud, est.
- Il significato di `latest` è una regola di disponibilità, non un anno fissato nel codice; non è stata interrogata la disponibilità remota durante questo controllo.
- La selezione ogni tre ore non produce somme o medie triorarie. Il downloader dichiara e conserva precipitazione e radiazione accumulate sull'ora precedente ciascun tempo selezionato; non applica conversioni locali delle unità.
- Il reticolo dell'area predefinita contiene `81 × 131` posizioni. Il numero di celle usate dal modello può essere inferiore.
- I file dei due prodotti sono combinati per tempi validi e coordinate, controllando la presenza di tutte le feature e il solo livello 850 hPa. Longitudini riportate a `[-180,180)`, latitudini ordinate da nord a sud.
- I blocchi locali hanno forma `[tempo,81,131,15]`; i cubi del modello `[T,N_valide,15]`.
- Default della preparazione: training 1980–2004; test dal 2005 all'ultimo anno completo consecutivo presente localmente.
- `static_mask` seleziona le celle con tutte le feature finite nel primo frame di training e richiede che questa maschera resti identica nei frame successivi, incluso il test. Una variazione genera errore; non viene imputata.
- Non è una maschera terra/mare fornita a priori: il criterio è la completezza delle 15 feature. Le celle marine possono essere escluse per l'umidità del suolo.
- La normalizzazione delle feature rimane quella del detector, stimata sul training. Target: valori meteorologici correnti della patch, non produzione PV.
- Recente predefinito: `[B,1,15,P,P]`; trend: `[B,56,15]`; 56 passi di tre ore corrispondono a 168 ore. Restano maschera `[B,1,P,P]` e calendario `[B,31]`.

## 4. Modello o algoritmo

### Detector

- ConvGRU–LSTM, funzioni di perdita e definizione degli score sono ereditati da `4410e3e9`; il file del modello non cambia nel delta.
- `fit_and_score_stgan` verifica ora lunghezza e regolarità temporale secondo `timestep_hours`, invece di imporre un'ora.
- Il comando ERA5 usa `timestep_hours=3`, `trend_steps=56`, `EPSG:4326`, passo angolare 0.5° e audit kNN disabilitato.
- I kernel ConvGRU 1/3/5 sono configurabili anche qui; default della patch ereditato pari a 3.
- Download, preparazione, training e analisi degli eventi sono fasi distinte. Non si deduce dal codice che una di esse sia già stata completata.

### Dagli score agli eventi

1. `CubeGrid` ricostruisce gli score `[T,N_valide]` sulla griglia `[T,H,W]`; celle escluse rappresentate da NaN.
2. Soglia assoluta opzionale oppure percentile, globale per default e per frame se richiesto.
3. Il confronto è **strettamente maggiore** della soglia. Con pareggi, il numero selezionato non coincide necessariamente con il top-K esatto usato negli export del detector.
4. Apertura e chiusura binarie per frame, rispettando la maschera valida; default un'iterazione ciascuna con elemento strutturante quadrato 3×3.
5. Componenti connesse a otto vicini e filtro minimo sul numero di celle.
6. Calcolo di area sferica per cella, area del cluster e centroide pesato per area.
7. Collegamenti fra cluster di frame consecutivi tramite sovrapposizione, sovrapposizione dopo dilatazione, distanza dei centroidi o unione dei criteri.
8. Default: sovrapposizione dilatata, dilatazione di una cella e soglia 0.1. Il rapporto usa l'intersezione divisa per il minimo delle cardinalità dei due supporti, non l'intersezione sull'unione.
9. Gli eventi sono componenti del grafo temporale dei cluster: una separazione conserva l'identità dell'evento; una fusione unifica retrospettivamente le identità precedenti.
10. I salti temporali interrompono i collegamenti. La durata esportata è `fine - inizio + passo_temporale`; viene conservato anche il solo tempo trascorso.

Il parametro `--kernel-size` ha due significati a seconda del sottocomando: in `train` è il kernel dei gate ConvGRU; in `events` è l'elemento strutturante della morfologia. Una ablation dell'uno non è una ablation dell'altro.

L'apertura/chiusura può rimuovere o aggiungere celle alla maschera binaria iniziale. Di conseguenza i cluster finali non rappresentano semplicemente l'elenco immutato degli score sopra soglia.

## 5. Elementi nuovi introdotti

- Contratto condiviso delle 15 feature e adapter dei file mensili dei due prodotti.
- Preparazione con selezione statica delle celle complete e split temporale esplicito.
- Generalizzazione della cadenza temporale e della griglia senza sostituire il modello neurale.
- `CubeGrid`, `disk_percentile`, `morphology`, `spatial_clusters`, `EventConfig`, `link_clusters` e `process_events`.
- Definizione di un evento tramite componenti del grafo dei cluster, con gestione di split/merge.
- Politiche esplicite per soglia, celle mancanti, area, durata e discontinuità temporali.
- Nessuna nuova loss e nessuna nuova testa di previsione fotovoltaica.

## 6. Informazioni utili per la tesi

### Metodi

- Motivare separatamente il passaggio di dominio da PVGIS a ERA5 e la scelta delle 15 variabili.
- Specificare tempi selezionati, griglia, area, livello di pressione e significato degli accumuli.
- Descrivere l'allineamento dei due prodotti e la maschera statica delle celle complete.
- Definire lo split 1980–2004 / 2005–latest come default implementato, non come esperimento già concluso.
- Esplicitare l'equivalenza di durata del trend: 168 passi orari contro 56 passi triorari.
- Separare score continuo, soglia percentile, maschera morfologica, cluster spaziale ed evento temporale.
- Distinguere kernel neurale e kernel morfologico.
- Formalizzare rapporti di sovrapposizione, fusioni retrospettive, aree e durate.
- Precisare che l'algoritmo è retrospettivo: normalizzazione globale degli score e fusioni degli eventi utilizzano informazioni del periodo analizzato.

### Related Work

- Rianalisi meteorologica e differenza fra variabili istantanee e accumulate.
- Anomaly detection multivariata spazio-temporale.
- CNN/ConvGRU su griglie geografiche.
- Morfologia matematica binaria e componenti connesse.
- Tracciamento di oggetti con split e merge.
- Aree sferiche e centroidi geografici pesati.
- Definizioni di durata e persistenza degli eventi meteorologici.

## 7. Aspetti da chiarire

- Copertura e completezza effettiva dei file scaricati; nessuna verifica dei dati è stata eseguita.
- Numero e distribuzione geografica delle celle escluse e validità dell'ipotesi di maschera costante sull'intero periodo.
- Motivazione scientifica dello split, delle 15 feature e dei parametri morfologici: non è ricostruibile dai default.
- Effetto di una griglia in gradi sulla dimensione fisica della patch al variare della latitudine; il codice pesa le aree dei cluster, ma non rende metricamente uniforme il kernel neurale.
- Scelta fra soglia globale, per frame o assoluta e relazione con il top-K degli export PVGIS.
- Opportunità di unificare retrospettivamente eventi che si fondono; non è una scelta neutra rispetto al conteggio degli eventi.
- Sensibilità dei risultati a morfologia, collegamenti e durata minima: il codice parametrizza l'algoritmo, ma non fornisce evidenze di validazione scientifica.
- Il campo `reference_hyperparameters` nella pipeline ereditata controlla alcuni iperparametri, ma non certifica l'equivalenza scientifica fra domini, feature e griglie.
- Non sono attestati training ERA5 completati, eventi reali individuati o risultati quantitativi.
- Non è stato esaminato il contenuto dei capitoli LaTeX: il controllo riguarda la copertura delle note di branch, non l'aggiornamento dell'intero manoscritto.

## 8. Sintesi

- ERA5 è uno step nuovo non coperto dalle note preesistenti.
- Il detector eredita ConvGRU–LSTM e usa 15 feature su griglia 0.5° ogni tre ore.
- Preparazione con maschera statica e trend di 168 ore.
- Score trasformati in cluster ed eventi con morfologia e collegamenti temporali.
- Implementazione documentata; esecuzione e validazione sperimentale non accertate.
