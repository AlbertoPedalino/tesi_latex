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
        elif ch == '.' and not depth and (i + 1 == len(text) or (text[i + 1] == ' ' and (text[i + 2:i + 3].isupper() or text[i + 2:i + 3] == '$'))):
            return text[:i + 1]
    return text


def block(stem, label, rows, captions, names=None):
    """Two images in a row share one scale, so that their text has the same size, and fill 0.96 of the text width
    (about 0.48 each), aligned at their bottom edge so that their captions start at the same height; an image alone
    keeps the size at which it is drawn. The figure scripts draw every image of a
    pair at about half the text width, so that the text inside the figures has the same size everywhere."""
    names = names or {}
    image = lambda letter: names.get(letter, f'{stem}_{letter}')
    lines = ['\\begin{figure}[htbp]', '    \\centering', '    \\phantomcaption', f'    \\label{{{label}}}']
    for number, row in enumerate(rows):
        if row == '|':  # the rest on the next float, as the same figure
            lines += ['\\end{figure}', '', '\\begin{figure}[htbp]', '    \\ContinuedFloat', '    \\centering']
            continue
        assert len(row) <= 2, 'at most two images in a row'
        parts = []
        for letter in row:
            fraction = min(0.97, width_cm(image(letter)) / DRAWN_CM)
            if len(row) == 2:  # one scale for the pair: the same size of text, about 0.48 each
                fraction = width_cm(image(letter)) / DRAWN_CM * min(1.1, 0.96 * DRAWN_CM / sum(
                    width_cm(image(other)) for other in row))
            text = captions[letter]
            parts.append(f'    \\begin{{subfigure}}[t]{{{fraction:.3f}\\textwidth}}\n'
                         f'        \\centering\n'
                         f'        \\includegraphics[width=\\linewidth]{{{image(letter)}}}\n'
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


def case(label, the_variable, moment, place, core, cell, note=''):
    """The three images of an example; the durations and the zones are in the table of the examples."""
    return {
        'a': f"Scarto standardizzato {the_variable} {moment}, con il nucleo del cluster in arancione e la cella seguita "
             f"nel cerchio. {core} {NE}",
        'b': f"$G_{{\\mathrm{{n}}}}$ come percentile {moment}, con il nucleo del cluster in arancione e la cella seguita "
             f"nel cerchio. {NE}",
        'c': f"Percentile di ogni score nella cella seguita ({place}), in alto, e scarto standardizzato {the_variable}"
             f"{note}, in basso; in azzurro gli istanti in cui la cella è nel cluster. {cell}",
    }


PROFILE = "percentile mediano di ogni score secondo la distanza in celle dal bordo del cluster"


L = {'cicli': 'fig:era5-cicli', 'luogo': 'fig:ris-g-luogo', 'temp': 'fig:ris-d-termini-temporale',
     'mappa': 'fig:ris-d-termini-mappa', 'prof': 'fig:ris-eventi-profilo', 'tipi': 'fig:ris-eventi-profilo-tipi',
     'drop': 'fig:ris-dropout', 'dmap': 'fig:ris-dropout-mappe'}
SHALLOW = r"la fascia grigia va dal più basso al più alto dei sei modelli shallow"
FIGURES = {
    'chapters/dati.tex': [
        ('era5_cicli', L['cicli'], ['ab'], {
            'a': r"Ciclo stagionale della temperatura a due metri e della copertura nuvolosa in tre celle: media per "
                 r"mese, con una fascia a una deviazione standard. Anni 1980--1999. Livello e ampiezza del ciclo "
                 r"cambiano da una cella all'altra.",
            'b': r"Ciclo giornaliero della temperatura a due metri e della copertura nuvolosa in Europa settentrionale "
                 r"(blu), Pianura Padana (arancio) e Sahara (verde): media per ora UTC in gennaio (tratteggio) e in "
                 r"luglio (linea continua). Anni 1980--1999. Il ciclo cambia con il luogo e con la stagione."}),
    ],
    'chapters/risultati.tex': [
        ('era5_g_luogo', L['luogo'], ['a', 'b'], {
            'a': r"Percentuale dei valori segnalati da G, senza normalizzazione climatologica, per fascia di latitudine e mese, sugli "
                 r"anni di validazione; il bianco corrisponde al 5\%. Le segnalazioni si concentrano a sud e in estate.",
            'b': r"Percentuale dei valori segnalati da $G_{\mathrm{n}}$, con la normalizzazione climatologica, per fascia "
                 r"di latitudine e mese, sugli anni di validazione; il bianco corrisponde al 5\%. Le segnalazioni sono "
                 r"distribuite in modo quasi uniforme."}),
        ('era5_d_termini_correlazione_temporale', L['temp'], ['ab'], {
            'a': r"Correlazione temporale di Spearman fra G e $D_{\mathrm{reale}}$, il giudizio del discriminatore "
                 r"sull'osservazione, in ogni cella, sugli anni di validazione. È negativa in tutte le celle, intorno a "
                 r"$-0.7$. " + NE,
            'b': r"Correlazione temporale di Spearman fra G e $D_{\mathrm{generato}}$, il giudizio del discriminatore "
                 r"sulla ricostruzione, in ogni cella, sugli anni di validazione. È negativa in tutte le celle, intorno "
                 r"a $-0.7$. " + NE}),
        ('era5_d_termini_mappa', L['mappa'], ['ab'], {
            'a': r"G il 9 agosto 2003 alle 12~UTC, l'istante con la media di G più alta degli anni di validazione, come "
                 r"percentile fra le celle dell'istante; la linea arancione racchiude l'1\% delle celle con G più "
                 r"alto. " + NE,
            'b': r"D il 9 agosto 2003 alle 12~UTC, come percentile fra le celle dell'istante; la linea arancione "
                 r"racchiude l'1\% delle celle con G più alto. Dentro la linea D è fra i valori più bassi. " + NE}),
        ('era5_eventi_profilo', L['prof'], ['ab'], {
            'a': r"Percentile mediano dello score secondo la distanza in celle dal bordo del cluster più vicino, per "
                 r"ogni metodo sui propri cluster, sugli anni di test; " + SHALLOW + r". $G_{\mathrm{n}}$ cala più "
                 r"lentamente di quasi tutte le baseline.",
            'b': r"Percentile mediano dello score secondo la distanza in celle dal bordo del cluster più vicino, per "
                 r"tutti i metodi sui cluster di $G_{\mathrm{n}}$, sugli anni di test; " + SHALLOW + r". Tutte le "
                 r"baseline calano allo stesso modo di $G_{\mathrm{n}}$."}),
        ('era5_eventi_profilo_tipi', L['tipi'], ['ab', '|', 'c'], {
            'a': r"Percentile mediano di $G_{\mathrm{n}}$ secondo la distanza in celle dal bordo del cluster, per tipo "
                 r"di cluster, sugli anni di test: velocità verticale (rosa), precipitazione (viola), copertura "
                 r"nuvolosa (grigio scuro), temperatura a due metri (ocra), altri tipi in grigio chiaro. Le curve "
                 r"quasi coincidono.",
            'b': r"Scarto standardizzato mediano, in valore assoluto, della variabile del tipo, secondo la distanza in "
                 r"celle dal bordo del cluster, per ogni tipo di cluster, sugli anni di test. Alcune variabili tornano "
                 r"ai livelli di fondo entro due celle, altre restano a lungo lontane dal normale.",
            'c': r"Percentile mediano di ogni score a due celle fuori dal bordo dei propri cluster, un pallino per tipo "
                 r"di cluster, sugli anni di test; il segmento va dal tipo più basso al più alto e il numero a destra è "
                 r"la loro differenza. $G_{\mathrm{n}}$ è fra i metodi più uniformi."}),
        ('era5_concordate_moto', 'fig:ris-concordate-moto', ['ab'], {
            'a': r"Spostamento lungo il moto delle segnalazioni di ogni metodo rispetto ai cluster concordati brevi, "
                 r"secondo la velocità del cluster: mediana sui cluster, positiva davanti al cluster; " + SHALLOW +
                 r". $G_{\mathrm{n}}$ sta un po' dietro, di più nei cluster veloci.",
            'b': r"Spostamento lungo il moto delle segnalazioni di ogni metodo rispetto ai cluster concordati lunghi, "
                 r"secondo la velocità del cluster: mediana sui cluster, positiva davanti al cluster; " + SHALLOW +
                 r". $G_{\mathrm{n}}$ sta davanti, più delle baseline tranne che nei 48 cluster più veloci."}),
        ('era5_dropout', L['drop'], ['ab', 'c'], {
            'a': r"Dispersione media divisa per G medio, per fascia di percentile di G, sugli anni di test. In "
                 r"proporzione a G la dispersione è quasi costante, e cala solo sui valori di G più alti.",
            'b': r"Mediana del rapporto fra dispersione e G secondo la distanza in celle dal bordo del cluster di "
                 r"$G_{\mathrm{n}}$, sugli anni di test. Il rapporto è quasi costante, con un minimo poco dentro il "
                 r"bordo.",
            'c': r"Valori segnalati da $G_{\mathrm{n}}$ divisi a parità di G in cinque classi, dalle estrazioni più "
                 r"concordi (1) alle più discordi (5), e percentuale di ciascuna classe oltre la soglia della distanza "
                 r"dal normale. Anni di test. Fino alla classe 4, più dispersione va con dati più lontani dal "
                 r"normale."}),
        ('era5_dropout_mappe', L['dmap'], ['ab'], {
            'a': r"Dispersione di G fra le 20 estrazioni in ogni cella, in media sugli anni di test. È più alta a sud e "
                 r"lungo il margine del dominio. " + NE,
            'b': r"Rapporto medio fra dispersione e G in ogni cella, sugli anni di test. Varia poco nel dominio. " + NE}),
        ('era5_caso_00', 'fig:ris-caso-velocita', ['ab', 'c'], case(
            'fig:ris-caso-velocita', 'della velocità verticale', 'il 23 novembre 2017 alle 00~UTC',
            r"$58.5^\circ$N, $8.5^\circ$E",
            "Il nucleo è una fascia stretta in cui la velocità verticale è per lo più sotto il normale.",
            r"$G_{\mathrm{n}}$ resta oltre la soglia per circa un giorno e mezzo, e la distanza dal normale sale e "
            r"scende con esso.", note=" (negativo se l'aria sale)")),
        ('era5_caso_01', 'fig:ris-caso-precipitazione', ['ab', 'c'], case(
            'fig:ris-caso-precipitazione', 'della precipitazione', 'il 19 maggio 2006 alle 03~UTC',
            r"$56.5^\circ$N, $23^\circ$E",
            "Il nucleo è diviso in più parti, in cui la precipitazione è quasi ovunque sopra il normale.",
            r"La distanza dal normale è alta già nei giorni prima; $G_{\mathrm{n}}$ supera la soglia solo con la "
            r"precipitazione.")),
        ('era5_caso_02', 'fig:ris-caso-nuvole', ['ab', 'c'], case(
            'fig:ris-caso-nuvole', 'della copertura nuvolosa', 'il 15 giugno 2024 alle 12~UTC',
            r"$23.5^\circ$N, $13^\circ$W",
            "Il nucleo è una fascia stretta sull'Africa nord-occidentale in cui la copertura nuvolosa è quasi ovunque "
            "sopra il normale.",
            r"$G_{\mathrm{n}}$ e la distanza dal normale salgono e scendono insieme.")),
        ('era5_caso', 'fig:ris-casi-profili', ['ab', 'c'], {
            'a': r"Velocità verticale, 23 novembre 2017 alle 00~UTC: " + PROFILE + r". $G_{\mathrm{n}}$ cala "
                 r"gradualmente.",
            'b': r"Precipitazione, 19 maggio 2006 alle 03~UTC: " + PROFILE + r". $G_{\mathrm{n}}$ cala gradualmente.",
            'c': r"Copertura nuvolosa, 15 giugno 2024 alle 12~UTC: " + PROFILE + r". La distanza dal normale cala più in "
                 r"fretta di $G_{\mathrm{n}}$."},
         {'a': 'era5_caso_00_d', 'b': 'era5_caso_01_d', 'c': 'era5_caso_02_d'}),
    ],
}

for name, figures in FIGURES.items():
    path = ROOT / name
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    for stem, label, rows, captions, *names in figures:
        text = replace_figure(text, label, block(stem, label, rows, captions, names[0] if names else None))
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
