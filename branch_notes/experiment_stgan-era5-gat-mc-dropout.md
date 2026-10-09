# Branch: experiment/stgan-era5-gat-mc-dropout

## 1. Scopo dello step

- Funzione principale: sostituire, nel generatore dello STGAN su ERA5, l'encoder spaziale convoluzionale su patch con un Graph Attention Network sparso a due layer definito sul grafo dell'intera griglia. Il discriminatore resta quello locale a patch.
- Posizione: ramo parallelo a `experiment/stgan-era5-mc-dropout`, che mantiene la ConvGRU nel generatore. I due rami condividono il protocollo MC dropout e si distinguono per l'encoder spaziale del generatore.
- Revisione analizzata: `351ba0c5`, 8 ottobre 2026 (riferimento locale). `origin` è avanti di un commit, `ee10748c`, che modifica solo `.gitignore`.
- Predecessore effettivo: commit `1264aa50` del 22 settembre 2026 ("finalize STGAN ERA5 MC dropout pipeline"), punto di biforcazione sul ramo CNN. Confronto `1264aa50..351ba0c5`, 41 commit.
- Differenza rispetto allo step precedente: nel predecessore un campione è una cella target con la sua patch 3×3 e il generatore usa una ConvGRU; qui un campione è un timestamp con tutti i nodi e il generatore usa attenzione su grafo.
- Perimetro di questa nota: le sole parti specifiche del GAT. Score MC in modalità `paper`, split con validation, monitoraggio, riferimenti PCA/MMD, codifica temporale ciclica, normalizzazione stagionale, z-score climatologico, BF16 e ripresa da checkpoint sono presenti anche qui e sono descritti in `experiment_stgan-era5-mc-dropout.md`.
- Non presente in questo branch: la variante CNN `full_grid` (`full_grid.py` è assente).
- Esclusi: notebook, sweep, documentazione, log, risultati e lo script di benchmark. Nessun esperimento è stato eseguito per questa nota.

## 2. File di codice rilevanti

Percorsi riferiti a `351ba0c5`.

| Percorso | Responsabilità | Modifiche principali |
|---|---|---|
| `physiq_pv/anomaly_detection/stgan/graph.py` | Grafo e attenzione | Nuovo: `grid_edge_index`, `SparseGATLayer`, `TwoLayerGAT`. |
| `physiq_pv/anomaly_detection/stgan/gat.py` | Modello | Nuovo: `STGANGATGenerator` e `STGANGAT`, con adattatori che estraggono le patch per il discriminatore. |
| `physiq_pv/anomaly_detection/stgan/graph_data.py` | Dataset | Nuovo: `STGANGraphDataset`, un campione per timestamp con tutti i nodi. |
| `physiq_pv/anomaly_detection/stgan/training.py` | Passo di ottimizzazione | Aggiunge `graph_gan_train_step`: stessi obiettivi, discriminatore valutato a blocchi di centri. |
| `physiq_pv/anomaly_detection/stgan/pipeline.py` | Orchestrazione | Ramo `graph_mode`: costruzione degli archi, scelta di dataset e modello, classe di checkpoint `STGAN_GAT`. |
| `physiq_pv/anomaly_detection/stgan/config.py` | Configurazione | `spatial_encoder`, `gat_hidden_dim`, `gat_heads`, `gat_layers`, `discriminator_chunk_size`, `trend_chunk_size`; `STGANGATConfig`. |
| `physiq_pv/anomaly_detection/stgan/model.py` | Discriminatore e parti comuni | Il discriminatore a patch è riusato senza modifiche architetturali. |
| `scripts/run_era5_stgan.py` | Comandi | `--spatial-encoder {gat,convgru}` con default `gat`; opzioni del GAT; batch di default pari a un timestamp. |

`scripts/benchmark_stgan_gat.py` misura tempi e non introduce metodo.

## 3. Dati e preprocessing

- Input, feature, split e normalizzazioni sono quelli del ramo CNN.
- Nodi: le località valide della griglia, nell'ordine originale. Le celle senza località non sono nodi.
- Archi: per ogni nodo, un arco entrante da ciascuno degli otto vicini immediati presenti sulla griglia, più un auto-anello. Archi diretti, memorizzati come indice sparso `[2,E]`. Nessun avvolgimento ai bordi e nessuna matrice di adiacenza densa.
- I buchi della griglia restano buchi: un vicino assente non genera un arco.
- Tensori per `B` timestamp, `N` nodi, `F = 15` feature, `T = recent_steps`, `S = 56` passi di trend:
  - recente `[B,T,N,F]`;
  - trend `[B,N,S,F]`;
  - maschera `[B,N,1]`, tutta a uno;
  - calendario `[B,31]` (`onehot`) oppure `[B,N,4]` (`cyclic`);
  - osservato `[B,N,F]`;
  - indici di timestamp e di nodo `[B,N]`.
- Le patch 3×3 esistono solo per il discriminatore e sono ricavate dagli indici dei vicini di ogni nodo; una cella assente vale zero e ha maschera 0.

## 4. Modello o algoritmo

### Layer di attenzione

- `SparseGATLayer`: proiezione lineare senza bias delle feature di ogni nodo in `heads × out` valori.
- Punteggio di attenzione additivo per arco: somma di un termine della sorgente e di uno della destinazione, seguita da LeakyReLU con pendenza 0.2.
- Normalizzazione softmax sugli archi entranti di ciascuna destinazione, calcolata solo sugli `E` archi.
- Aggregazione dei messaggi pesati sulla destinazione; le teste sono concatenate oppure mediate, poi si aggiunge un bias.
- Sotto BF16 attenzione e somme sparse restano in FP32.

### Encoder spaziale

- `TwoLayerGAT`: primo layer con teste concatenate (default 16 feature per testa, 4 teste) e attivazione ELU; secondo layer con teste mediate e uscita di dimensione pari ai canali dell'encoder (default 32).
- Il codice impone esattamente due layer.
- Con due layer su vicinato a otto, ogni nodo riceve informazione fino a due celle di distanza. Il raggio non cresce con il numero di passi recenti, perché l'aggregazione temporale è puntuale.

### Generatore

1. Il GAT è applicato a ogni passo recente su tutti i nodi: `[B·T,N,F] → [B,T,N,C]`.
2. Una GRU a gate puntuali (ConvGRU con kernel 1) aggrega i passi recenti per nodo: `[B,N,C]`. È applicata anche con un solo passo recente.
3. La LSTM di trend legge una sequenza per nodo. È eseguita a blocchi di nodi (`trend_chunk_size`, default 256) con gradient checkpointing; le equazioni non cambiano.
4. Il calendario è proiettato e, se condiviso, ripetuto su tutti i nodi.
5. Tre dropout nello stesso ordine del modello CNN (spaziale, temporale, fusione), poi proiezione 1×1 con `tanh`: `[B,N,F]`.

La mescolanza spaziale è affidata al solo GAT; la parte temporale recente non contiene convoluzioni spaziali.

### Discriminatore

- È il discriminatore a patch del ramo CNN, senza dropout.
- Per ogni nodo si estrae la patch 3×3 della storia recente, dell'osservazione e della ricostruzione globale di G; il discriminatore restituisce un output per nodo.
- I centri sono elaborati a blocchi di `discriminator_chunk_size`. Il default della configurazione GAT e del comando è 10611, cioè l'intera griglia ERA5 di un timestamp in un blocco.
- Il codice dichiara che la dimensione del blocco influisce solo su memoria e velocità.

### Funzioni di perdita e training

- Obiettivi invariati: D con `0.5 · [BCE(D(reale), 0) + BCE(D(generato), 1)]`; G con `w · ricostruzione + BCE(D(generato), 0)`, `w = 500` di default.
- La ricostruzione è la media, sui centri, dell'errore quadratico medio sulla patch di ciascun centro. Poiché le patch sono estratte dalla ricostruzione globale, l'errore di un nodo entra attraverso tutte le patch che lo contengono.
- Le loss dei blocchi sono pesate per la frazione di centri del blocco: la somma è la media su tutti i centri.
- Passo di G: i gradienti rispetto alla ricostruzione sono accumulati blocco per blocco su una copia staccata della predizione, poi propagati una sola volta attraverso il generatore.
- Un passo di ottimizzazione per rete e per batch; con batch 1 un passo vede tutti i nodi di un timestamp. Gli step per epoca sono `ceil(T / batch)`, con `T` timestamp utili.
- Campionamento: timestamp rimescolati, ciascuno con tutti i nodi.
- Tassi distinti per G e D, rapporto di aggiornamento D:G, BF16 e ripresa da checkpoint valgono come nel ramo CNN.

### Score

- Stesse definizioni per coppia tempo–nodo: `sG` errore quadratico medio sulla patch del nodo, `sD = D(reale) − D(generato)`, errori per feature del nodo.
- Campionamento MC come nel ramo CNN: encoder di G, storia di D e `D(reale)` calcolati una volta, dropout e parte finale ripetuti per ogni campione.
- Fusione `paper` e deviazione standard MC invariate.

## 5. Elementi nuovi introdotti

- Classi: `SparseGATLayer`, `TwoLayerGAT`, `STGANGATGenerator`, `STGANGAT`, `STGANGraphDataset`, `STGANGATConfig`.
- Funzioni e metodi: `grid_edge_index`, `graph_gan_train_step`, `encode_trend`, `patch_plan`, `gather_patch`, `patch_inputs`, `patch_batches`.
- Comportamenti modificati: unità del campione da cella con patch a timestamp con tutti i nodi; batch di default da 256 patch a 1 timestamp; il comando ERA5 usa `gat` come encoder di default.
- Componenti sostituiti: la ConvGRU 3×3 del generatore è sostituita da GAT a due layer più GRU puntuale. La ConvGRU 3×3 resta nel discriminatore.
- Non modificati: LSTM di trend, proiezione del calendario, dropout, proiezione di uscita, architettura del discriminatore, target avversari, formula dello score.

## 6. Informazioni utili per la tesi

### Metodi

- Costruzione del grafo: nodi, vicinato a otto, auto-anelli, archi diretti, trattamento dei bordi e dei buchi.
- Equazioni del layer di attenzione: proiezione, punteggio additivo, LeakyReLU, softmax sugli archi entranti, combinazione delle teste.
- Architettura a due layer con dimensioni e attivazioni; campo recettivo risultante.
- Flusso del generatore: GAT per passo, GRU puntuale nel tempo, LSTM per nodo, fusione, uscita.
- Asimmetria fra le due reti: generatore globale su grafo, discriminatore locale su patch 3×3.
- Riduzione della loss di ricostruzione tramite le patch e sua differenza rispetto a una media diretta sui nodi.
- Unità del batch e numero di step di ottimizzazione per epoca rispetto al modello a patch.
- Elaborazione a blocchi di D e della LSTM come accorgimento di memoria che non cambia gli obiettivi.

### Related Work

- Graph Attention Network: attenzione additiva multi-testa.
- Message passing su grafi regolari derivati da griglie.
- GRU e loro uso come aggregatore temporale per nodo.
- Reti spazio-temporali su grafo per dati meteorologici.
- STGAN originale con unità ricorrenti a convoluzione su grafo, di cui questa variante sostituisce l'encoder spaziale.
- Discriminatori locali a patch.
- Gradient checkpointing.

## 7. Aspetti da chiarire

- Valore di `recent_steps` da riportare: il default del comando è 1, il parametro è configurabile.
- Motivazione dei due layer fissi, di 4 teste e di 16 feature per testa.
- Motivazione della scelta di mantenere il discriminatore a patch invece di un discriminatore su grafo.
- Confrontabilità con il ramo CNN a patch: cambiano insieme encoder, unità del campione e numero di step per epoca.
- Invarianza rispetto alla dimensione del blocco di D e della LSTM: affermata nei commenti e coperta da test presenti nel branch, non eseguiti per questa nota.
- Effetto di una griglia in gradi sulla distanza fisica fra nodi vicini al variare della latitudine: gli archi non hanno pesi geometrici.
- Assenza di feature sugli archi e di codifica posizionale dei nodi: è un fatto del codice, la motivazione non è ricavabile.
- Esiti: nessun training completato, score o evento reale è attestato dal codice.

## 8. Sintesi

- Il generatore passa da ConvGRU su patch a GAT sparso a due layer sul grafo globale della griglia, seguito da una GRU puntuale nel tempo.
- Un campione è un timestamp con tutti i nodi; un passo di ottimizzazione vede l'intero campo.
- Il discriminatore resta locale 3×3 ed è valutato su ogni nodo a blocchi.
- Loss, target, score MC e fusione `paper` sono quelli del ramo CNN.
- Monitoraggio, validation e opzioni di input sono condivisi e descritti nella nota CNN.
