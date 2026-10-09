# Branch: feat/sde-net-paper-faithful

## 1. Scopo dello step

- Consolidare il previsore fotovoltaico con encoder temporale, attenzione sul grafo e blocco SDE, separando dispersione delle traiettorie e rumore della testa di regressione.
- Posizione nel percorso: previsione probabilistica di base, dalla quale si separano successivamente la previsione diretta multiorizzonte e le analisi delle anomalie.
- Revisione analizzata: `7998a75b`, corrispondente ai riferimenti locale e `origin` disponibili il 10 settembre 2026.
- Predecessore di confronto: stato `91753344`, antecedente al commit `4dc591d8` che introduce `sde_net.py`; è il punto di separazione dalla linea oggi denominata `feat/sde-net-true-minimal`. La punta attuale di quel branch non è il predecessore di questo. Git consente di verificare l'ascendenza dei commit, non il nome originario assegnato al ramo al momento della creazione.
- La selezione dei branch riguarda il periodo 10 agosto–10 settembre 2026; questa nota documenta lo step complessivo al suo stato finale. Un confronto aggiuntivo `90075d68..7998a75b` distingue gli aggiornamenti recenti dal nucleo architetturale precedente.
- Differenza principale: dal blocco con drift dipendente dal tempo, diffusione per nodo e uscita PV puntuale, a drift vettoriale, diffusione scalare per esempio e testa gaussiana eteroschedastica. Le modifiche recenti riguardano soprattutto la rivalutazione puntuale STGAN e la rimozione di alcuni helper di confronto tra run.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/model/sde_net.py` | Dinamica stocastica | Introduce `VectorDrift`, `ScalarDiffusion`, `PaperSDEBlock`, riferimento `YearMSDSDENet` e loss della diffusione. |
| `physiq_pv/model/st_gnn.py` | Previsore PV | Integra il nuovo blocco SDE, testa PV media/deviazione standard e prior geografico additivo nei logit GAT. |
| `physiq_pv/model/bilstm_encoder.py` | Encoder temporale ereditato | Chiarisce che il percorso attivo usa una BiLSTM con pooling temporale, anche se il flag storico si chiama `use_patchtst`. |
| `physiq_pv/model/graph_builder.py` | Grafo geografico | Costruisce archi geografici con pesi gaussiani e self-loop. |
| `physiq_pv/data/pvgis_irradiance.py` | Irradianza di ingresso | Ricostruisce POA effettiva da componenti diretta e diffusa inclinate. |
| `physiq_pv/data/pvgis_dataset.py` | Feature, split e finestre | Allineamento dei siti, statistiche di normalizzazione, validazione separata, target e filtro opzionale delle finestre con eventi. |
| `physiq_pv/data/pvgis_labels.py` | Etichette di analisi | Gestione delle categorie e dei gruppi associati alle osservazioni. |
| `physiq_pv/training/losses.py` | Obiettivo di regressione | NLL gaussiana elementare; MSE della testa ausiliaria. |
| `physiq_pv/training/train_loop.py` | Ottimizzazione | Passi distinti per previsione e diffusione, perturbazioni pseudo-OOD, scelta del modello sulla validazione. |
| `physiq_pv/training/uncertainty.py` | Inferenza probabilistica | Traiettorie browniane, decomposizione della varianza e quantili della miscela gaussiana. |
| `physiq_pv/experiments/pvgis_stgnn_runner.py` | Configurazione del protocollo | Collega costruzione dei dati, modello, training, inferenza e parametri di selezione. |
| `physiq_pv/experiments/sde_pipeline.py` | Coordinamento e rivalutazione | Costruzione dei comandi e rietichettatura delle previsioni tramite detector. |
| `physiq_pv/reporting/pointwise_detector_posthoc.py` | Analisi puntuale | Join previsione/etichetta e filtro opzionale degli azzeramenti solari regionali isolati. |
| `physiq_pv/reporting/daytime_bin_anomaly_report.py`, `posthoc_outputs.py`, `run_metrics.py` | Metriche e stratificazioni | Errori e intervalli per categorie e fasce di produzione; statistiche dei boxplot sull'insieme delle righe ammissibili. |
| `physiq_pv/reporting/anomaly_driver.py`, `anomaly_extremes.py` | Diagnostica delle anomalie | Attribuzione tramite contributi del detector, segno dello scostamento e aggregazione in episodi. |
| `physiq_pv/reporting/event_onset.py`, `hourly_event_response.py` | Risposta durante gli eventi | Riferimenti locali per ora del giorno, profili dall'insorgenza e confronto con persistenza. |
| `physiq_pv/reporting/mtgflow_spatiotemporal.py` | Analisi spaziale | Aggregazione degli score e raggruppamenti geografici. |

## 3. Dati e preprocessing

- Input: serie PVGIS orarie sincronizzate per località, coordinate geografiche e, quando richiesto, etichette esterne del detector.
- Il set completo dichiarato nel codice contiene: temperatura a 2 m, POA, vento a 10 m, seno e coseno dell'elevazione solare, indice di cielo sereno sul piano inclinato, sua deviazione standard mobile su tre ore, differenza temporale POA, componenti diretta e diffusa inclinate, storia della produzione PV.
- `pv_lag_pvgis` è il valore PV presente nella finestra storica: la causalità deriva dal taglio della finestra prima del target.
- Target principale: `pv_power_output` al timestamp futuro; target ausiliario opzionale: `kt_poa`.
- Normalizzazione: z-score globale per temperatura, POA, vento e differenza POA; scala PV per sito dal percentile 99 dei valori positivi diurni del fit, con fallback mediano. Altri canali conservano la trasformazione fisica esplicita.
- `build_datasets` separa un anno di validazione, per default l'ultimo degli anni forniti, prima di stimare la normalizzazione. Lo split test è fornito separatamente.
- Il filtro `train_normal_only` elimina finestre con eventi nel contesto delle feature, nella sequenza o nel target. Le feature derivate richiedono anche due ore precedenti la finestra esplicita.
- Forme: dati `[T,N,F]`; batch di ingresso `[B,N,L,F]`; target e uscite PV `[B,N]`; stato latente `[B,N,D]`.
- Nella rivalutazione puntuale, il percorso NetCDF identifica azzeramenti simultanei regionali di diretta, diffusa, altezza solare e produzione, racchiusi fra osservazioni a irradianza regionale positiva. Il percorso manifest controlla invece la sola POA. Entrambi escludono anche l'osservazione immediatamente successiva: i due controlli non leggono esattamente le stesse variabili.

## 4. Modello o algoritmo

- Flusso: BiLSTM condivisa tra siti → proiezione → GAT geografica → integrazione SDE → testa PV e testa ausiliaria opzionale.
- L'encoder temporale è una BiLSTM a due strati nel costruttore del modello; la dimensione e il pooling sono configurabili.
- Nei logit GAT il prior geografico entra come `edge_prior_strength * log(edge_weight)`; segue una softmax sugli archi entranti e l'aggregazione dei messaggi.
- `VectorDrift` applica Linear e ReLU. `ScalarDiffusion` media lo stato iniziale sui nodi e produce uno scalare sigmoidale per esempio, di forma `[B,1]`.
- Euler–Maruyama: `x_next = x + f(x)*dt + sigma*g(x0)*sqrt(dt)*epsilon`, con rumore indipendente nelle coordinate latenti. La scala è calcolata sullo stato iniziale e mantenuta lungo la traiettoria; default del blocco: quattro passi su un intervallo latente di durata quattro.
- Questo tempo di integrazione è un parametro del modello, distinto dalle ore dell'orizzonte di previsione.
- Testa PV: media `softplus` e deviazione standard `softplus + 1e-3`. L'obiettivo elementare è `log(sigma²) + (y-mu)²/sigma²`, equivalente alla NLL gaussiana a meno di fattore e costante.
- Testa ausiliaria: indice POA limitato tramite sigmoide e `kt_poa_max`; MSE pesata quando abilitata.
- Training: un ottimizzatore SGD aggiorna encoder, GAT, drift e teste; un secondo SGD aggiorna la diffusione mediante BCE, con ID=0 e pseudo-OOD=1. I pseudo-OOD sono `x + noise_std * randn_like(x)`; il secondo passo usa rappresentazioni staccate dall'encoder.
- Sono previsti warm-up della scala SDE, decadimento del learning rate della previsione, clipping dei suoi gradienti ed early stopping su metrica di validazione.
- Inferenza: ogni traiettoria restituisce una gaussiana. Media predittiva = media delle medie; varianza totale = varianza delle medie + media delle varianze delle componenti.
- L'intervallo principale usa quantili a code uguali della miscela gaussiana, invertiti numericamente. La banda media ± z·deviazione totale è conservata come approssimazione diagnostica.
- Il percorso deterministico integra il solo drift: non è in generale la media esatta delle traiettorie stocastiche.
- Le analisi successive uniscono le previsioni alle etichette osservate senza riaddestrare il previsore. Il top-K STGAN può essere ricalcolato dopo le esclusioni ammissibili.

## 5. Elementi nuovi introdotti

- Componenti SDE separati, diffusione globale per esempio e testa gaussiana.
- Inferenza mediante miscela di gaussiane e separazione delle due componenti della dispersione.
- Filtro esplicito delle finestre di training e validazione, protocolli di eventi regionali e rivalutazione puntuale dei detector.
- Analisi degli errori per produzione, driver, intervallo temporale e tempo dall'insorgenza.
- Nel delta recente, `build_horizon_comparison_figures` viene rimosso da questo ramo. La previsione diretta multiorizzonte appartiene al ramo dedicato, non a questa implementazione del modello.
- `YearMSDSDENet` è una classe di riferimento presente nel codice; la sua presenza non dimostra che sia stata usata negli esperimenti PV.

## 6. Informazioni utili per la tesi

### Metodi

- Split temporali, statistiche del fit e causalità delle finestre.
- BiLSTM, GAT con prior geografico e diffusione condivisa per esempio.
- Dinamica latente, loss separate e costruzione pseudo-OOD.
- Testa gaussiana, miscela predittiva, decomposizione della varianza e selezione sulla validazione.
- Distinzione fra etichette puntuali, eventi regionali e gruppi degli errori.
- Differenza fra intervalli predittivi e distribuzione degli errori assoluti.

### Related Work

- Equazioni differenziali stocastiche neurali e discretizzazione Euler–Maruyama.
- SDE-Net e discriminazione ID/pseudo-OOD.
- BiLSTM, pooling temporale e Graph Attention Networks.
- Regressione gaussiana eteroschedastica e legge della varianza totale.
- Intervalli da miscele gaussiane, calibrazione e ampiezza degli intervalli.
- Persistenza per serie fotovoltaiche e rilevamento di eventi spazio-temporali.

## 7. Aspetti da chiarire

- I valori effettivamente usati in ciascuna esecuzione, il numero di siti, i periodi realmente disponibili e le prestazioni non sono ricavabili dai soli default.
- L'allineamento a un articolo dichiarato nei commenti non è una verifica bibliografica indipendente.
- `build_detector_event_protocol` riceve gli anni della mappa storica che include anche l'anno di validazione; la sua calibrazione regionale va descritta separatamente dal fit della normalizzazione, che esclude tale anno.
- Il riferimento usato da `attach_anomaly_sign` è costruito sulle osservazioni presenti nel file di previsioni: è una diagnostica retrospettiva, non una feature causale del modello.
- Il codice non dimostra che gli episodi selezionati corrispondano a eventi meteorologici certificati, né che le due componenti di incertezza siano calibrate.
- La validità delle esclusioni solari, la confrontabilità dei due percorsi di rilevamento e i criteri finali di selezione degli eventi richiedono verifica sui dati, esclusa da questa analisi.

## 8. Sintesi

- Previsore PV BiLSTM–GAT con blocco SDE scalare per esempio.
- Regressione eteroschedastica e training separato della diffusione.
- Intervalli ottenuti dalla miscela delle gaussiane lungo le traiettorie.
- Analisi puntuali e regionali degli errori, senza nuovi risultati sperimentali.
