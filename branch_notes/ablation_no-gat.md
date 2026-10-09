# Branch: ablation/no-gat

## 1. Scopo dello step

- Rimuovere il message passing GAT dal previsore fotovoltaico SDE.
- Posizione: ablazione del backbone di previsione nella linea che già contiene la variante del detector STGAN-CNN.
- Revisione analizzata: `6bda0521`, riferimenti locale e `origin` concordi nella copia consultata il 10 settembre 2026.
- Predecessore effettivo: `experiment/stgan-cnn` a `28e532c1`, genitore diretto. Confronto `28e532c1..6bda0521`.
- Differenza: lo stato iniziale della SDE deriva direttamente dalla proiezione dell'encoder temporale, senza aggregazione sugli archi.
- L'ablazione riguarda il previsore `STGNN`, non il detector `STGAN`.
- Periodo di selezione: 10 agosto–10 settembre 2026; analisi statica senza esperimenti.

## 2. File di codice rilevanti

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/model/st_gnn.py` | Previsore | Rimuove `GATLayer`, lista degli strati GAT, relativi parametri e ciclo di message passing in `encode`. |
| `physiq_pv/data/pvgis_dataset.py` | Costruzione del modello | `make_model` smette di passare teste/layer GAT, flag spaziale e forza del prior. |
| `physiq_pv/experiments/pvgis_stgnn_runner.py` | Configurazione | Registra zero strati GAT; mantiene il parametro CLI del prior come valore senza effetto sul modello. |
| `physiq_pv/model/graph_builder.py` | Grafo ereditato | La modifica è esplicativa: il grafo continua a essere costruito, ma non viene consumato dal previsore. |
| `physiq_pv/model/sde_net.py` | Diffusione ereditata | Invariato; letto per verificare il pooling sui nodi ancora presente in `ScalarDiffusion`. |
| `physiq_pv/training/train_loop.py`, `physiq_pv/training/uncertainty.py` | Training/inferenza ereditati | Invariati; mantengono obiettivi e inferenza probabilistica del predecessore. |

## 3. Dati e preprocessing

- Feature, target, normalizzazione, split temporali e costruzione delle finestre restano quelli del predecessore.
- Il batch continua ad avere forma `[B,N,L,F]`: il codice non trasforma il dataset in una raccolta di modelli separati per sito.
- Le uscite gaussiane mantengono `[B,N]` oppure `[B,N,H]` se sono richiesti più orizzonti.
- Coordinate, archi e pesi geografici continuano a essere prodotti e passati lungo le interfacce compatibili.
- Nessuna nuova feature e nessuna nuova etichetta sono introdotte dall'ablazione.
- Il detector CNN eventualmente disponibile nel ramo rimane un componente ereditato distinto.

## 4. Modello o algoritmo

- Percorso precedente: BiLSTM per sito → proiezione → GAT → SDE → teste.
- Percorso nuovo: BiLSTM per sito → proiezione → SDE → teste.
- `encode` restituisce direttamente lo stato `[B,N,D]` dopo la proiezione; `edge_index` ed `edge_weight` non sono utilizzati.
- La dimensione latente conserva il nome storico `gat_dim`, anche se non identifica più uno strato GAT.
- La rimozione elimina parametri del modello; il delta non introduce un blocco sostitutivo per mantenere identico il numero di parametri.
- **Resta una dipendenza comune fra siti nella parte stocastica**: `ScalarDiffusion._example_summary` media lo stato iniziale sui nodi, poi calcola un unico coefficiente di diffusione per esempio. In modalità stocastica, il contenuto di altri siti può quindi modificare la scala del rumore di una località.
- È corretto descrivere l'encoder come privo di message passing sul grafo; non è corretto estendere automaticamente questa affermazione alla completa indipendenza delle distribuzioni predittive tra località.
- Il drift e le teste operano sulle rappresentazioni per sito. Il percorso deterministico senza rumore non usa la scala di diffusione per aggiornare lo stato.
- Training invariato: NLL gaussiana della produzione, eventuale MSE ausiliaria, ottimizzazione separata della diffusione tramite pseudo-OOD.
- Inferenza invariata: traiettorie browniane, componenti della varianza e intervalli dalla miscela gaussiana.
- L'assenza di consumo degli archi non elimina il costo della costruzione del grafo dalla pipeline.

## 5. Elementi nuovi introdotti

- Non viene aggiunta una nuova classe neurale: l'ablazione modifica direttamente `STGNN`.
- Rimossi `GATLayer`, costruzione degli strati e propagazione dei messaggi.
- Rimossi dal costruttore i parametri relativi a teste, numero di strati, abilitazione GAT e forza del prior.
- Mantenute le firme compatibili che ricevono archi/pesi e l'opzione CLI del prior senza effetto.
- Nessuna modifica al training o all'architettura del detector STGAN-CNN.

## 6. Informazioni utili per la tesi

### Metodi

- Ablazione del message passing geografico nel previsore.
- Componenti conservate e diminuzione della capacità parametrica.
- Distinzione fra assenza di GAT e diffusione ancora condivisa tra siti.
- Interfacce del grafo mantenute e costo di preprocessing non rimosso.
- Protocollo di training/inferenza identico a quello ereditato.

### Related Work

- Graph Attention Networks e apprendimento spazio-temporale.
- Ablazioni architetturali e controllo della capacità del modello.
- Pooling globale e dipendenza tra uscite.
- Diffusione neurale scalare condivisa.

## 7. Aspetti da chiarire

- Il commento del modello che parla di località previste indipendentemente è più forte di quanto garantisca il forward stocastico, a causa del pooling nella diffusione.
- Se l'obiettivo è eliminare ogni influenza tra siti, questo delta non lo realizza completamente.
- Il costo effettivamente risparmiato e la variazione esatta dei parametri richiedono il conteggio relativo alla configurazione scelta; non sono risultati deducibili dal nome del branch.
- L'effetto sulle prestazioni può riflettere sia la rimozione del message passing sia la riduzione dei parametri.
- Numero di epoche, configurazioni eseguite e risultati non sono stati consultati.
- Non sono stati eseguiti esperimenti né modifiche correttive.

## 8. Sintesi

- Rimozione della GAT dal previsore SDE.
- Encoder temporale, teste e protocollo probabilistico conservati.
- Grafo ancora costruito, senza utilizzo nel message passing.
- Diffusione globale ancora dipendente dall'insieme delle località.
