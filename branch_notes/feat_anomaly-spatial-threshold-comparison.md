# Branch: feat/anomaly-spatial-threshold-comparison

## 1. Scopo dello step

- Confrontare la distribuzione spazio-temporale delle decisioni MTGFlow/STGAN e gli errori del previsore al variare della soglia o della condizione di input/target.
- Posizione: analisi successive a training e inferenza, sulla linea del previsore multiorizzonte.
- Revisione analizzata: `6356f1f9`, riferimenti locale e `origin` concordi nella copia consultata il 10 settembre 2026.
- Predecessore effettivo: `feat/sde-direct-multihorizon-t1-t6` a `657a3fa8`; il primo commit successivo è `0a0444f1`. Confronto `657a3fa8..6356f1f9`.
- Il ramo incorpora anche aggregazioni dei flag analoghe alla linea `feat/sde-stgan-events-t6`; quest'ultima non è il suo predecessore Git diretto.
- Differenza principale: aggiunta di protocolli di confronto e rappresentazioni degli errori, senza modificare il previsore neurale.
- Periodo di selezione: 10 agosto–10 settembre 2026; nessun risultato sperimentale consultato.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/reporting/detector_threshold_sensitivity.py` | Sensibilità alla soglia | Coordinate di decisione MTGFlow/STGAN, join con errori, metriche per soglia e fasce MAE. |
| `physiq_pv/reporting/detector_spatial_comparison.py` | Confronto geografico | Selezione del vicinato di riferimento e aggregazioni per località, tempo e cluster. |
| `physiq_pv/reporting/input_target_cases.py` | Quattro casi input/target | Etichette STGAN globali, classificazione delle finestre, statistiche per bin e serie locale dello score. |
| `physiq_pv/reporting/stgan_regional.py` | Rappresentazione regionale | Quote orarie dei siti anomali e quote giornaliere delle ore anomale per sito. |
| `physiq_pv/reporting/pointwise_detector_posthoc.py` | Rivalutazione puntuale | Supporto multiorizzonte, esclusioni di record solari regionali isolati e ricalcolo del top-K. |
| `physiq_pv/reporting/anomaly_extremes.py` | Giorni ed episodi | Aggregazione dei flag, filtri diurni, raggruppamento di episodi ed etichette temporali. |
| `physiq_pv/reporting/mtgflow_spatiotemporal.py` | Statistiche geografiche | Estende i controlli sulle statistiche necessarie alle soglie e alle aggregazioni. |
| `physiq_pv/reporting/posthoc_outputs.py` | Figure degli errori | Adatta i raggruppamenti e i percorsi di analisi alle nuove stratificazioni. |
| `physiq_pv/reporting/daytime_bin_anomaly_report.py` | Statistiche ereditate | Fornisce `_boxplot_stats`, usata per quartili e baffi esatti degli errori assoluti. |

Gli script Python di impacchettamento e trasferimento delle immagini non modificano il metodo e sono esclusi dalla descrizione.

## 3. Dati e preprocessing

- Input: previsioni già disponibili con osservazione, previsione e orizzonte; score/decisioni dei detector; coordinate dei siti; eventuali statistiche di soglia e sorgenti PVGIS per le esclusioni.
- Le righe di previsione si distinguono per località, timestamp target e orizzonte. Le decisioni del detector si distinguono per località e timestamp.
- Errori: `abs_error=abs(y_pred-y_true)` e `squared_error=(y_pred-y_true)^2`.
- Le analisi per produzione selezionano osservazioni diurne e usano un picco di riferimento per assegnare i bin percentuali. Il picco e le date effettivamente forniti non sono dedotti da risultati.
- Le decisioni STGAN possono essere ricostruite globalmente dopo l'esclusione dei timestamp non ammissibili e prima di selezionare località, giorni o bin.
- La funzione dei quattro casi conserva anche le ore notturne per descrivere l'input.
- Input di default: 24 ore della stessa località, da `issue_timestamp-23h` a `issue_timestamp`, estremi inclusi. Almeno un flag anomalo rende anomala la finestra.
- Il target è normale/anomalo secondo l'etichetta STGAN dell'osservazione futura. Non è la classificazione della previsione numerica.
- Finestre incomplete e target senza score vengono esclusi con motivazioni distinte; non sono assimilati ai normali.
- Nessun nuovo tensore di training: i nuovi algoritmi operano su tabelle.

## 4. Modello o algoritmo

### Sensibilità alle soglie

- MTGFlow: per sito, soglia `Q3+k*IQR` con statistiche di riferimento del training. La coordinata `(score-Q3)/IQR` permette di selezionare gli anomali con coordinata maggiore o uguale a `k`.
- Il caricatore verifica che le statistiche ricostruiscano la soglia salvata al coefficiente di riferimento; non deduce i quartili dalla sola soglia.
- STGAN: coordinata basata sul ranking globale. Nel percorso che ricalcola il ranking, l'ordinamento stabile conserva i pareggi ed è costruito per riprodurre il conteggio top-K con arrotondamento per eccesso.
- A ogni soglia si dividono le stesse previsioni disponibili nei due gruppi, quindi si calcolano `MAE=sum(abs_error)/n` e `RMSE=sqrt(sum(squared_error)/n)`.
- Il valore di una curva è una metrica aggregata delle righe del gruppo, per orizzonte ed eventuale bin; non è un valore giornaliero.
- `plot_mae_dispersion` mantiene la curva MAE e aggiunge fascia scura Q1–Q3 e fascia chiara fra i baffi di Tukey degli errori assoluti.
- I baffi sono gli estremi osservati entro `Q1-1.5*IQR` e `Q3+1.5*IQR`. Non rappresentano necessariamente minimo e massimo complessivi.
- Gli outlier contribuiscono al MAE, anche se non sono mostrati singolarmente. La fascia non è centrata artificialmente sul MAE.
- Queste sono bande di dispersione degli errori, non intervalli di confidenza del MAE né intervalli predittivi del modello. Il RMSE resta una metrica aggregata senza un nuovo boxplot o una banda di questo tipo.

### Quattro casi e distribuzione regionale

- `classify_input_target_cases` costruisce normale→normale, normale→anomalo, anomalo→normale e anomalo→anomalo, controllando la coerenza fra emissione, target e orizzonte.
- Il campo input riguarda la sequenza esplicita della stessa località: non include tutte le altre località del grafo e non estende automaticamente il controllo alle ore aggiuntive delle feature derivate.
- `summarize_input_target_cases` restituisce MAE, RMSE e statistiche degli errori assoluti per caso, bin e orizzonte. Gruppi vuoti restano NaN.
- `plot_input_target_cases` mostra boxplot degli errori assoluti con media evidenziata e barre del RMSE aggregato.
- `plot_stgan_score_timeline` usa tempo sull'asse x e anomaly score sull'asse y; in assenza di un sito esplicito sceglie quello con più flag anomali. Questa è una selezione descrittiva dei casi più segnalati, non un campionamento rappresentativo.
- La serie locale inserisce NaN nelle ore mancanti e collega solo i tratti con dati disponibili.
- `aggregate_stgan_region`: quota oraria = siti anomali / siti con score valido; quota sito-giorno = ore anomale / ore con score valido.
- La rappresentazione regionale non applica una nuova soglia: aggrega le etichette puntuali ricevute. Le celle senza osservazioni restano mancanti.
- Il confronto geografico dispone di un'altra regola di riferimento: sito più vicino al centro medio delle coordinate, poi vicini geografici; questa regola è indipendente dagli score.

## 5. Elementi nuovi introdotti

- Moduli dedicati a sensibilità delle soglie, confronto spaziale, quattro casi e panorama regionale.
- Quantili degli errori assoluti associati ai gruppi definiti dalla soglia.
- Ricalcolo globale delle etichette STGAN sul sottoinsieme ammissibile.
- Controlli espliciti di copertura temporale e motivi di esclusione.
- Integrazione di analisi puntuali, regionali e per produzione senza riaddestramento.

## 6. Informazioni utili per la tesi

### Metodi

- Unità statistica località–timestamp–orizzonte e conteggi dei gruppi.
- Differenza fra soglia IQR per sito e ranking top-K globale.
- Calcolo aggregato delle metriche e definizione esatta delle bande.
- Quattro casi come condizionamento sulle osservazioni, non matrice di confusione del previsore.
- Copertura dei dati, ore escluse e ambito della finestra di input.
- Denominatori delle quote regionali e criteri di selezione dei siti mostrati.

### Related Work

- Soglie IQR, ranking globale e selezione a budget di anomalie.
- Boxplot di Tukey, quantili e dispersione rispetto a incertezza della stima.
- Errori di previsione condizionati agli eventi.
- Aggregazione spazio-temporale e dipendenza fra osservazioni.
- Analisi per fasce di produzione fotovoltaica.

## 7. Aspetti da chiarire

- La griglia concreta di soglie, gli eventi scelti, i picchi di riferimento e le figure effettivamente prodotte dipendono dai chiamanti esclusi dall'analisi.
- L'uguaglianza del numero di anomalie non rende equivalenti i criteri MTGFlow e STGAN.
- La sovrapposizione delle fasce non è un test di significatività e non dimostra equivalenza fra gruppi.
- Per stimare intervalli di confidenza servirebbe un protocollo ulteriore che consideri la dipendenza temporale e spaziale; non è implementato da queste bande.
- Gli errori del previsore rispetto alle categorie del detector non misurano direttamente l'accuratezza del detector rispetto a etichette indipendenti.
- Il filtraggio di azzeramenti solari via NetCDF e via manifest usa variabili diverse; non va dichiarato equivalente senza verifica.
- La classificazione della sola località centrale non rappresenta necessariamente lo stato dell'intero ingresso spaziale del previsore.

## 8. Sintesi

- Confronto degli errori per soglia, produzione e stato input/target.
- Fasce MAE basate sulla distribuzione degli errori assoluti.
- Serie locali e quote regionali dalle decisioni STGAN.
- Nessun nuovo training e nessuna conclusione sperimentale ricavata dal codice.
