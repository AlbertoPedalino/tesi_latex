# Branch: feat/sde-direct-multihorizon-t1-t6

## 1. Scopo dello step

- Estendere il previsore SDE da un solo orizzonte a più uscite future prodotte nello stesso passaggio.
- Posizione: estensione della previsione probabilistica, precedente alle analisi comparative delle anomalie.
- Revisione analizzata: `657a3fa8`, riferimenti locale e `origin` concordi nella copia Git consultata il 10 settembre 2026.
- Predecessore effettivo: `feat/sde-net-paper-faithful` nello stato `90075d68`, genitore del primo commit multiorizzonte `7e51c560`. Il confronto è `90075d68..657a3fa8`; la punta successiva `7998a75b` del ramo precedente non è usata come base.
- Differenza: una testa produce media e scala gaussiana per ogni orizzonte richiesto, mantenendo encoder e dinamica SDE condivisi. Non è un rollout autoregressivo.
- Periodo di selezione: 10 agosto–10 settembre 2026. Analisi statica dei soli sorgenti Python.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/data/pvgis_dataset.py` | Finestre supervisionate | Introduce tuple ordinate di orizzonti, target `[N,H]`, timestamp di emissione e filtro su tutti i target richiesti. |
| `physiq_pv/model/st_gnn.py` | Teste di uscita | Testa PV con `2*H` uscite e testa ausiliaria con `H` uscite. |
| `physiq_pv/training/train_loop.py` | Loss e validazione | Estende maschere e riscalamento fisico alla dimensione orizzonte. |
| `physiq_pv/training/uncertainty.py` | Inferenza e coordinate | Conserva la dimensione orizzonte durante le traiettorie e l'esportazione. |
| `physiq_pv/experiments/pvgis_stgnn_runner.py` | Configurazione | Aggiunge `--forecast-horizons`, controlla unicità e ordine e propaga la scelta. |
| `physiq_pv/experiments/sde_pipeline.py` | Coordinamento | Propaga gli orizzonti diretti e distingue i relativi percorsi di esecuzione. |
| `physiq_pv/reporting/posthoc_outputs.py` | Analisi multiorizzonte | Introduce `build_direct_multihorizon_posthoc` e dettagli per gli orizzonti selezionati. |
| `physiq_pv/reporting/pointwise_detector_posthoc.py` | Join con detector | Consente più previsioni, distinte per orizzonte, per la stessa località e lo stesso target. |
| `physiq_pv/reporting/daytime_bin_anomaly_report.py`, `anomaly_driver.py`, `event_onset.py` | Analisi degli errori | Aggiungono gestione/selezione dell'orizzonte nei percorsi di analisi. |
| `physiq_pv/reporting/mtgflow_spatiotemporal.py` | Aggregazione spaziale | Introduce errori per località/giorno e cluster/giorno filtrando prima l'orizzonte. |

## 3. Dati e preprocessing

- Stesse feature PVGIS del predecessore: meteorologia, geometria solare, indici e variazioni dell'irradianza e produzione storica.
- L'ingresso resta `[B,N,L,F]`. Per `H>1`, target PV, target ausiliari e uscite assumono forma `[B,N,H]`; con un solo orizzonte resta l'interfaccia `[B,N]`.
- Per una finestra che comincia all'indice `i`, l'emissione è `i+L-1`; il target dell'orizzonte `h` è `i+L+h-1`.
- Il numero di finestre per una serie di lunghezza `T` diventa `T-L-max(h)+1`.
- Nessun valore futuro viene reinserito nell'ingresso per generare l'orizzonte successivo.
- Se è attivo il training su finestre normali, è sufficiente un evento in uno dei target richiesti per escludere tutta la finestra. Non vengono automaticamente esaminati orizzonti intermedi non richiesti.
- La normalizzazione rimane quella del predecessore; la scala per sito viene estesa con una dimensione singleton sugli orizzonti.
- Gli identificatori di una previsione sono località, timestamp target e orizzonte; viene inoltre conservato `issue_timestamp`.

## 4. Modello o algoritmo

- Encoder BiLSTM, proiezione, GAT e blocco SDE rimangono condivisi.
- La testa PV produce coppie `(mu_h,sigma_h)`, con media non negativa e scala positiva. La testa ausiliaria, se attiva, produce un indice POA per orizzonte.
- Non è introdotta una matrice di covarianza appresa fra orizzonti. Le uscite condividono la rappresentazione e, durante un passaggio stocastico, la traiettoria latente.
- Training: NLL gaussiana elementare e MSE ausiliaria sono aggregate anche sugli orizzonti. La diffusione continua a essere aggiornata con il protocollo ID/pseudo-OOD del predecessore.
- Inferenza: campioni delle teste di forma `[M,B,N,H]`; decomposizione della varianza e quantili della miscela calcolati per ciascuna uscita.
- `_prediction_coordinates` appiattisce coordinate nell'ordine batch–nodo–orizzonte, coerente con i tensori.
- I confronti normale/anomalo di `build_direct_multihorizon_posthoc` usano l'etichetta puntuale associata al target.
- Le aggregazioni giornaliere e geografiche accumulano somme degli errori assoluti e quadratici e conteggi; il RMSE aggregato è la radice della media dei quadrati, non la media dei RMSE dei sottogruppi.

## 5. Elementi nuovi introdotti

- `_normalise_forecast_horizons`, parser `_forecast_horizons`, `_physical_scale` e `_prediction_coordinates`.
- Dimensione orizzonte nei dataset e nelle teste del modello.
- `build_direct_multihorizon_posthoc`, `aggregate_daily_forecast_errors` e `aggregate_forecast_error_clusters`.
- Join molti-a-uno fra previsioni multiorizzonte ed etichette del detector.
- Il percorso a singolo orizzonte resta disponibile. La costante di coordinamento elenca 1–6 ore, ma il parametro CLI, se omesso, continua a usare il singolo `--horizon`.

## 6. Informazioni utili per la tesi

### Metodi

- Previsione diretta multi-output e condivisione del backbone.
- Allineamento emissione/target e gestione dei diversi orizzonti.
- Regola di esclusione di una finestra con almeno un target anomalo.
- NLL per orizzonte e distribuzioni predittive marginali.
- Metriche separate per orizzonte, categoria e produzione.

### Related Work

- Strategie dirette, multi-output e ricorsive per la previsione multiorizzonte.
- Apprendimento con rappresentazioni condivise.
- Previsione probabilistica marginale e dipendenza fra orizzonti.
- Aggregazione corretta di MAE e RMSE.

## 7. Aspetti da chiarire

- Gli orizzonti effettivamente attivati in una determinata esecuzione e i relativi iperparametri.
- L'equivalenza fra i campioni di confronto: la richiesta di più target riduce le finestre disponibili e può rafforzare il filtro degli eventi.
- L'eventuale confronto con modelli separati per orizzonte richiede dati sperimentali non consultati.
- Non è possibile dedurre dal codice un miglioramento a sei ore, una migliore calibrazione o un vantaggio computazionale.
- La dipendenza congiunta fra orizzonti non è valutata da una NLL gaussiana multivariata esplicita.

## 8. Sintesi

- Un solo previsore produce più orizzonti futuri direttamente.
- Dataset, teste gaussiane e inferenza mantengono la dimensione orizzonte.
- Etichette puntuali e metriche sono allineate a ogni timestamp target.
- Nessuna nuova prestazione è dedotta dalla sola implementazione.
