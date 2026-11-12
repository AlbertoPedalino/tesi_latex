"""Every panel of the multi-panel figures is its own image with its own whole caption, "Figura 6.16a: ...": no letters
(a), (b) and no common caption. The figure number is stepped by \\phantomcaption; the list of figures gets one entry
per image, 6.16a, 6.16b, ...

The captions of the split figures live here, not in the chapters: edit them below and run
    python -I strumenti/figure_separate.py
which rewrites the figure blocks in the chapters (running it twice changes nothing). The panels are cut by
figures/pannelli.py from the figure scripts."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
DRAWN_CM = 14.0
NE = 'Linee di costa da Natural Earth.'


def width_cm(stem):
    data = (FIG / f'{stem}.pdf').read_bytes()
    box = re.search(rb'/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', data)
    return (float(box.group(3)) - float(box.group(1))) / 72 * 2.54


def first_sentence(text):
    """The first sentence, for the list of figures: up to the first full stop outside math that is followed by a
    space and a capital letter (so not "classe 4. i" nor "6.16a")."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == '$':
            depth ^= 1
        elif ch == '.' and not depth and (i + 1 == len(text) or (text[i + 1] == ' ' and text[i + 2:i + 3].isupper())):
            return text[:i + 1]
    return text


def block(stem, label, rows, captions, natural=False, shrink=1.0):
    """Images side by side aligned at their bottom edge, so that their captions start at the same height."""
    centimetres = {letter: width_cm(f'{stem}_{letter}') for row in rows if row != '|' for letter in row}
    widths = {letter: cm / DRAWN_CM for letter, cm in centimetres.items()}
    scale = shrink * min(1.0, min(0.97 / sum(widths[l] for l in row) for row in rows if row != '|'))
    lines = ['\\begin{figure}[htbp]', '    \\centering', '    \\phantomcaption', f'    \\label{{{label}}}']
    for number, row in enumerate(rows):
        if row == '|':  # the rest on the next float, as the same figure
            lines += ['\\end{figure}', '', '\\begin{figure}[htbp]', '    \\ContinuedFloat', '    \\centering']
            continue
        parts = []
        for letter in row:
            width = f'{centimetres[letter]:.2f}cm' if natural else f'{scale * widths[letter]:.3f}\\textwidth'
            text = captions[letter]
            parts.append(f'    \\begin{{subfigure}}[t]{{{width}}}\n'
                         f'        \\centering\n'
                         f'        \\includegraphics[width=\\linewidth]{{{stem}_{letter}}}\n'
                         f'        \\caption[{first_sentence(text)}]{{{text}}}\n'
                         f'        \\label{{{label}-{letter}}}\n'
                         f'    \\end{{subfigure}}')
        lines.append('\n    \\hfill\n'.join(parts) if len(row) > 1 else parts[0])
        if number < len(rows) - 1 and rows[number + 1] != '|':
            lines.append('    \\par\\bigskip')
    return '\n'.join(lines)


def replace_figure(text, label, new):
    at = text.index('\\label{' + label + '}')
    start = text.rindex('\\begin{figure}', 0, at)
    end = text.index('\\end{figure}', at)
    after = end + len('\\end{figure}')
    following = text[after:after + 120].lstrip()
    if following.startswith('\\begin{figure}') and '\\ContinuedFloat' in following[:60]:  # its continued float
        end = text.index('\\end{figure}', text.index('\\ContinuedFloat', after))
    comments = [l for l in text[start:end].splitlines() if l.lstrip().startswith('%')]
    return text[:start] + new + ('\n' + '\n'.join(comments) if comments else '') + '\n' + text[end:]


def ref(label, letter):
    return f'Figura~\\ref{{{label}-{letter}}}'


def case(label, variable, the_variable, moment, cluster, core, cell):
    """Short captions: four of them share one page."""
    return {
        'a': f"{core} Scarto standardizzato {the_variable} {moment}; in arancione il nucleo del cluster ({cluster}), il cerchio è la "
             f"cella della {ref(label, 'c')}. {NE}",
        'b': f"$G_{{\\mathrm{{n}}}}$ nello stesso istante della {ref(label, 'a')}, come percentile, con lo stesso nucleo "
             f"e la stessa cella. {NE}",
        'c': f"{cell} In alto il percentile di ogni score nella cella (gli score leggono tutte le variabili), con le "
             f"baseline come nella Figura~\\ref{{fig:ris-eventi-profilo-a}}; in basso lo scarto standardizzato {the_variable} da "
             f"sola. In azzurro gli istanti in cui la cella fa parte del cluster.",
        'd': f"Il cluster dal nucleo verso l'esterno, nell'istante delle mappe: percentile mediano di ogni score "
             f"secondo le celle dal bordo del cluster, con le baseline come nella "
             f"Figura~\\ref{{fig:ris-eventi-profilo-a}}; fuori, solo le celle più vicine a questo nucleo che ad "
             f"altre segnalazioni.",
    }


L = {'cicli': 'fig:era5-cicli', 'luogo': 'fig:ris-g-luogo', 'temp': 'fig:ris-d-termini-temporale',
     'mappa': 'fig:ris-d-termini-mappa', 'prof': 'fig:ris-eventi-profilo', 'tipi': 'fig:ris-eventi-profilo-tipi',
     'drop': 'fig:ris-dropout', 'dmap': 'fig:ris-dropout-mappe'}
FIGURES = {
    'chapters/dati.tex': [
        ('era5_cicli', L['cicli'], ['ab'], {
            'a': r"Lo stesso mese ha livelli e variabilità diversi da una cella all'altra. Ciclo stagionale della "
                 r"temperatura a due metri e della copertura nuvolosa in tre celle: Europa settentrionale ($55^\circ$N, "
                 r"$10^\circ$E), Pianura Padana ($45^\circ$N, $9^\circ$E) e Sahara ($25^\circ$N, $10^\circ$E); media "
                 r"per mese, con una fascia a una deviazione standard dei valori triorari del mese. La temperatura è in "
                 r"gradi Celsius (in kelvin nei dati). Anni 1980--1999.",
            'b': r"Il ciclo giornaliero cambia con il luogo e con la stagione. Media per ora UTC in gennaio "
                 r"(tratteggio) e in luglio (linea continua), per le stesse variabili e celle della "
                 + ref(L['cicli'], 'a') + r". Anni 1980--1999."}, True),
    ],
    'chapters/risultati.tex': [
        ('era5_g_luogo', L['luogo'], ['a', 'b'], {
            'a': r"Senza normalizzazione le segnalazioni di G si concentrano a sud e in estate. Percentuale dei valori "
                 r"segnalati da G, per fascia di latitudine e mese, sugli anni di validazione, con la soglia al "
                 r"95-esimo percentile dello score sugli stessi anni. La scala dei colori non è lineare: il bianco "
                 r"corrisponde al 5\%, il blu a percentuali minori, il rosso a percentuali maggiori.",
            'b': r"Con la normalizzazione climatologica le segnalazioni sono distribuite in modo quasi uniforme. Come "
                 r"la " + ref(L['luogo'], 'a') + r", per $G_{\mathrm{n}}$: G con la normalizzazione climatologica per "
                 r"cella, ora e giorno dell'anno (finestra di $\pm15$ giorni)."}),
        ('era5_d_termini_correlazione_temporale', L['temp'], ['ab'], {
            'a': r"In ogni luogo il giudizio del discriminatore sull'osservazione scende quando G sale: tutte le celle "
                 r"sono blu, con correlazioni intorno a $-0.7$. Correlazione temporale di Spearman, in ogni cella, fra G e "
                 r"$D_{\mathrm{reale}}$, sugli anni di validazione, su una scala da $-1$ a $1$. " + NE,
            'b': r"Lo stesso vale per il giudizio sulla ricostruzione. Correlazione temporale di Spearman, in ogni "
                 r"cella, fra G e $D_{\mathrm{generato}}$, come nella " + ref(L['temp'], 'a') + '. ' + NE}),
        ('era5_d_termini_mappa', L['mappa'], ['ab'], {
            'a': r"G, l'errore di ricostruzione del generatore, il 9 agosto 2003 alle 12~UTC, l'istante con la media di G più alta "
                 r"degli anni di validazione, come percentile fra le celle di quell'istante. La linea arancione "
                 r"racchiude l'1\% delle celle con G più alto. " + NE,
            'b': r"Dove G è più alto, D è fra i valori più bassi. D, la componente del discriminatore, nello stesso "
                 r"istante e con la stessa linea della " + ref(L['mappa'], 'a') + r"; dentro la linea il percentile "
                 r"mediano è 6. " + NE}),
        ('era5_eventi_profilo', L['prof'], ['ab'], {
            'a': r"$G_{\mathrm{n}}$ cala più lentamente di quasi tutte le baseline. Ogni metodo sui propri cluster: "
                 r"percentile mediano dello score secondo le celle di distanza dal bordo del cluster più vicino, sugli "
                 r"anni di test, con la soglia di ciascun metodo al 95-esimo percentile del proprio score sugli anni di "
                 r"validazione; i sei modelli shallow sono la fascia grigia, dal più basso al più alto.",
            'b': r"Sui cluster di $G_{\mathrm{n}}$ tutte le baseline calano allo stesso modo di $G_{\mathrm{n}}$. Come "
                 r"la " + ref(L['prof'], 'a') + r", ma con tutti i metodi misurati sui cluster di $G_{\mathrm{n}}$."}),
        ('era5_eventi_profilo_tipi', L['tipi'], ['ab', '|', 'c'], {
            'a': r"$G_{\mathrm{n}}$ cala allo stesso modo per tutti i tipi di cluster. Percentile mediano di "
                 r"$G_{\mathrm{n}}$ secondo le celle di distanza dal bordo del cluster, per ogni tipo di cluster, sugli "
                 r"anni di test. In colore, con i nomi nella Figura~\ref{fig:ris-eventi-profilo-tipi-b}, i tre tipi principali, quelli con più valori segnalati nei cluster medi ed "
                 r"enormi della Figura~\ref{fig:ris-eventi-dimensione} (velocità verticale, precipitazione e copertura "
                 r"nuvolosa), e la temperatura a due metri, citata nel testo; gli altri tipi sono in grigio. Ogni punto è la "
                 r"mediana dei valori a quella distanza in tutti i cluster del tipo, di ogni dimensione; i punti con "
                 r"meno di 1000 valori non sono tracciati, perché la mediana sarebbe poco affidabile.",
            'b': r"Le variabili, invece, calano in modo diverso: alcune tornano ai livelli di fondo entro due celle fuori "
                 r"dal bordo del cluster, come la precipitazione, altre restano a lungo lontane dal normale, come la "
                 r"temperatura a due metri. Scarto standardizzato mediano, in valore assoluto, della variabile che dà il nome al tipo, in deviazioni "
                 r"standard (0 è il valore normale), con gli stessi tipi e colori della "
                 + ref(L['tipi'], 'a') + r"; l'asse è limitato a 3.5.",
            'c': r"$G_{\mathrm{n}}$ non è il solo a calare allo stesso modo per tutti i tipi di cluster: è fra i "
                 r"metodi più uniformi, con la variazione in tre ore, HBOS e COPOD, mentre la distanza dal normale, MCD "
                 r"e PCA cambiano di 20 su 100 o più da un tipo all'altro. Una riga per metodo, ciascuno sui "
                 r"propri cluster. Ogni pallino è un tipo di cluster, con i colori della " + ref(L['tipi'], 'a') + r", "
                 r"e sta al percentile mediano dello score nei valori a due celle fuori dal bordo: il percentile dice "
                 r"dove cade lo score fra i valori degli anni di validazione (50 è il valore tipico, 95 la soglia), e la mediana "
                 r"è presa su tutti i valori a quella distanza. Il segmento va dal tipo più basso al più alto; il numero "
                 r"a destra è la differenza fra i due percentili, su una scala da 0 a 100. Anni di test, con la soglia di ciascun "
                 r"metodo al 95-esimo percentile del proprio score sugli anni di validazione; non sono considerati i tipi con "
                 r"meno di 1000 valori a quella distanza."}),
        ('era5_dropout', L['drop'], ['abc'], {
            'a': r"In proporzione a G la dispersione del dropout è quasi costante, e cala solo sui valori di G più "
                 r"alti. Dispersione media divisa per G medio, per percentile di G su "
                 r"tutti i valori, sugli anni di test.",
            'b': r"Attraverso i cluster il rapporto è quasi costante, con un minimo poco dentro il bordo. Mediana del "
                 r"rapporto fra dispersione e G, secondo le celle di distanza dal bordo del cluster, con la soglia "
                 r"di $G_{\mathrm{n}}$ al 95-esimo percentile, sugli anni di test.",
            'c': r"A parità di G, più dispersione va con dati più lontani dal normale, fino alla penultima classe. I "
                 r"valori segnalati da $G_{\mathrm{n}}$, divisi a parità di G in cinque classi con lo stesso numero "
                 r"di valori, ordinate per rapporto (1 = estrazioni più concordi, 5 = più discordi); per ogni classe, la percentuale oltre la soglia della "
                 r"distanza dal normale. Anni di test."}),
        ('era5_dropout_mappe', L['dmap'], ['ab'], {
            'a': r"La dispersione del dropout è più alta a sud e lungo il margine del dominio. Dispersione "
                 r"di G fra le 20 estrazioni in ogni cella, in media sugli anni di test. " + NE,
            'b': r"Il rapporto fra dispersione e G varia poco nel dominio. Rapporto medio in ogni cella, "
                 r"sugli anni di test. " + NE}),
        ('era5_caso_00', 'fig:ris-caso-velocita', ['ab', 'c', '|', 'd'], case(
            'fig:ris-caso-velocita', 'velocità verticale', 'della velocità verticale', 'il 23 novembre 2017 alle 00~UTC',
            "63 ore, dal 21 al 24 novembre 2017, con il massimo in questo istante",
            "Il nucleo è una fascia stretta in cui la velocità verticale è per lo più sotto il normale.",
            r"Nella cella cerchiata $G_{\mathrm{n}}$ resta oltre la soglia, salvo una breve pausa, per circa un giorno e "
            r"mezzo, e la distanza dal normale sale e scende insieme a $G_{\mathrm{n}}$.")),
        ('era5_caso_01', 'fig:ris-caso-precipitazione', ['ab', 'c', '|', 'd'], case(
            'fig:ris-caso-precipitazione', 'precipitazione', 'della precipitazione', 'il 19 maggio 2006 alle 03~UTC',
            "72 ore, dal 16 al 19 maggio 2006, finito poche ore dopo il massimo",
            "Il nucleo è diviso in più parti, in cui la precipitazione è quasi ovunque sopra il normale.",
            r"Nella cella cerchiata la distanza dal normale è alta già nei giorni prima, e $G_{\mathrm{n}}$ supera la "
            r"soglia solo quando arriva la precipitazione.")),
        ('era5_caso_02', 'fig:ris-caso-nuvole', ['ab', 'c', '|', 'd'], case(
            'fig:ris-caso-nuvole', 'copertura nuvolosa', 'della copertura nuvolosa', 'il 15 giugno 2024 alle 12~UTC',
            "66 ore, dal 14 al 17 giugno 2024, con il massimo in questo istante",
            "Il nucleo è una fascia stretta sull'Africa nord-occidentale in cui la copertura nuvolosa è quasi ovunque "
            "sopra il normale.",
            r"Nella cella cerchiata $G_{\mathrm{n}}$ e la distanza dal normale salgono e scendono insieme, e "
            r"$G_{\mathrm{n}}$ resta oltre la soglia per dodici ore.")),
    ],
}

for name, figures in FIGURES.items():
    path = ROOT / name
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    for stem, label, rows, captions, *natural in figures:
        shrink = 1.0  # four images and four captions on one page
        text = replace_figure(text, label, block(stem, label, rows, captions, bool(natural and natural[0]), shrink))
    if name == 'chapters/risultati.tex':
        start = text.index('\\paragraph{Come leggere le figure.}')
        end = text.index('La Tabella~\\ref{tab:ris-casi}', start)
        text = text[:start] + (
            "\\paragraph{Come leggere le figure.} Ogni esempio ha quattro immagini. Le prime due sono le mappe "
            "dello scarto standardizzato della variabile del tipo e di $G_{\\mathrm{n}}$, con il nucleo del cluster racchiuso dalla "
            "linea arancione; la terza segue nel tempo la cella cerchiata sulle mappe, cioè la cella del nucleo con "
            "$G_{\\mathrm{n}}$ più alto; la quarta è il profilo del cluster dal nucleo verso l'esterno, come nella "
            "Figura~\\ref{fig:ris-eventi-profilo-a}. ") + text[end:]
    path.write_text(text, encoding='utf-8', newline='\n')

tesi = ROOT / 'tesi.tex'
t = tesi.read_text(encoding='utf-8').replace('\r\n', '\n')
old = '\\captionsetup[sub]{font=small}\n'
new = ('% Images of one figure, each with its own caption "Figura 6.16a: ..." and its own entry in the list of figures.\n'
       '\\DeclareCaptionLabelFormat{sottofigura}{\\figurename~\\thefigure#2}\n'
       '\\captionsetup[sub]{font=small, labelformat=sottofigura, labelsep=colon, list=true, listformat=subsimple}\n'
       '\\makeatletter\\let\\l@subfigure\\l@figure\\makeatother\n')
if old in t:
    t = t.replace(old, new)
tesi.write_text(t, encoding='utf-8', newline='\n')
print('ok')
