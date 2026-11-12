"""Check the terms of the thesis against its closed glossary.

Run from anywhere:
    python strumenti/controlla_lessico.py
It reads every text a reader sees: the running text of the chapters included by tesi.tex (comments skipped), the
captions and tables they contain, the words inside every figure they include (pdftotext, which comes with TeX Live)
and inside the TikZ schemes, and the summary in riassunto/. Every rule below is a variant that the glossary does not
allow; each hit is printed with file, line and context. No output under a rule means the thesis follows it.

The glossary (one term per concept; the rules forbid the variants found while writing):
- G: "errore di ricostruzione del generatore" (in the formulas of Ch. 4 $\\bar{s}_{G,tn}$); G_n: "score del
  rilevatore", $G_{\\mathrm{n}}(t,n)$ in the formulas; D: "componente del discriminatore"; D_reale and D_generato:
  the two "giudizi" of the discriminator; G and D together: "componenti"; their sum: "score combinato".
- The output of the generator: "ricostruzione". The draws of the Monte Carlo dropout: "estrazioni"; their standard
  deviation: "dispersione"; dispersione divided by G: "rapporto". n_D:n_G: "aggiornamenti D:G".
- "valore" = cell-instant pair ("coppia cella--istante"); "valore segnalato", "segnalazione"; "cluster (di
  segnalazioni)"; zones "nucleo", "contorno", "zona lontana"; "bordo del cluster"; "margine del dominio".
- "scarto standardizzato" (Section 4.3); "percentuale"; "anni di addestramento / di validazione / di test";
  "storia ripulita"; "anomalia meteorologica" only for real phenomena; short names of the variables after Tab. 3.1.
Words of the article (STGAN's "campo generato", "errore di previsione") may appear where the article is described.
"""
import re
import subprocess
import sys
from pathlib import Path

TESI = Path(__file__).resolve().parents[1]
# (pattern, what is wrong, files where the pattern is allowed)
RULES = [
    (r'\\zeta|ζ', 'zeta: lo score del rilevatore è G_n', ()),
    (r'(?<!in )\bquot[ae]\b(?! \d)', '"quota": si scrive "percentuale" (salvo "in quota", altitudine)', ()),
    (r'punti? percentil', '"punti percentili": scrivere i due percentili', ()),
    (r'\bscart[oi]\b(?! standardizzat)(?! stand\.)', 'scarto senza "standardizzato"', ()),
    (r'valor[ei] standardizzat|variabilità abituale', 'vecchio nome dello scarto standardizzato', ()),
    (r'(?<!ricostruzione )\berrore del generatore', 'G si chiama "errore di ricostruzione del generatore"', ()),
    (r'componente di ricostruzione|score di ricostruzione', 'nome di G', ()),
    (r'giudizio del discriminatore(?! sull)(?! sulla)', '"giudizio" è D_reale o D_generato; D è la "componente"',
     ('stato_arte.tex',)),
    (r'leggono il punto|ogni cella da sola', '"metodi che leggono una cella in un istante"', ()),
    (r'osservazioni segnalate', '"valori segnalati"', ()),
    (r'\btraining\b', '"addestramento"', ()),
    (r'\bsulla validazione\b|\bdella validazione\b|\b[Ss]ul test\b', '"sugli anni di validazione / di test"', ()),
    (r'periodo di addestramento', '"anni di addestramento"', ('stato_arte.tex',)),
    (r'\bi due score\b(?= (salgono|sono quasi))', 'G e D sono "componenti"', ()),
    (r'rapporto \$?n_D|rapporto D:G|\blr (gen|discr)\b|agg\. discr', 'n_D:n_G sono "aggiornamenti D:G"', ()),
    (r'peso della ricostruzione', '"peso di ricostruzione"', ()),
    (r'campo stimato', 'l\'uscita del generatore è la "ricostruzione"', ()),
    (r'score dell\'articolo|metodo originale', '"score combinato"', ()),
    (r'score di STGAN(?! come)', '"score combinato" fuori dal Cap. 2', ('stato_arte.tex',)),
    (r'media climatologica', 'nel Cap. 3: "normale per mese e ora"', ()),
    (r'deviazion[ei] standard delle estrazioni', '"dispersione"', ()),
    (r'istante[-–]+cella|coppi[ae] di cella e istante', '"coppia cella–istante"', ()),
    (r'bordo (interno )?del nucleo|\bsuo bordo\b', '"bordo del cluster"', ()),
    (r'a 2\s*~?m\b|a 10\s*~?m\b', '"a due metri", "a dieci metri"', ()),
    (r'umidità del primo strato|precipitazione totale|copertura nuvolosa totale|altezza dello strato limite',
     'nomi lunghi delle variabili dopo la Tab. 3.1', ('dati.tex',)),
    (r'A parità di errore|percentile dell\'errore|Secondo l\'errore', 'nella 6.8 G si chiama G', ()),
]
RULES = [(re.compile(p, re.I), what, allowed) for p, what, allowed in RULES]


def visible_lines(path):
    for number, line in enumerate(path.read_text(encoding='utf-8').split('\n'), 1):
        if not line.lstrip().startswith('%'):
            yield number, re.sub(r'(?<!\\)%.*', '', line)


def sources():
    chapters = re.findall(r'^\\include\{(chapters/[^}]+)\}', (TESI / 'tesi.tex').read_text(encoding='utf-8'), re.M)
    graphics = set()
    for name in chapters:
        path = TESI / f'{name}.tex'
        for number, line in visible_lines(path):
            graphics |= set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}', line))
            yield path.name, number, line
            for scheme in re.findall(r'\\input\{([^}]*)\}', line):
                tikz = TESI / (scheme if scheme.endswith('.tex') else scheme + '.tex')
                for n, l in visible_lines(tikz):
                    yield tikz.name, n, l
    for g in sorted(graphics):
        pdf = TESI / 'figures' / (g if g.endswith('.pdf') else g + '.pdf')
        if pdf.exists():
            text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True).stdout
            yield pdf.name, 0, ' '.join(text.decode('utf-8', 'replace').split())
    summary = TESI / 'riassunto' / 'riassunto_tesi_it.tex'
    inside = False
    for number, line in visible_lines(summary):
        inside = inside or '\\begin{document}' in line
        if inside:
            yield summary.name, number, line


def main():
    hits = 0
    for regex, what, allowed in RULES:
        found = [(name, number, line, m) for name, number, line in sources() if name not in allowed
                 for m in regex.finditer(line)]
        if found:
            print(f'\n### {what} ({len(found)})')
            for name, number, line, m in found:
                print(f'  {name}:{number}: ...{line[max(0, m.start() - 70):m.end() + 50]}...')
        hits += len(found)
    print(f'\n{hits} varianti fuori glossario.' if hits else 'Nessuna variante fuori glossario.')
    return 1 if hits else 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
