# ICML AIWILD main-text figures as ASCII plots

This is a standalone, text-only companion to `docs/icml_aiwild_main_figure_tables.md`. It converts every active main-text figure in `overleaf/icml_aiwild_template/icml_aiwild_2026.pdf` into Markdown-safe ASCII and keeps exact numeric tables immediately below the plots. It does not modify the manuscript or the original table document.

## Rendering guide

- Every plot is inside a fenced monospaced block; no image embedding is used.
- Numeric y-resolution is stated or evident from the labeled ticks.
- Dense Elo plots assign distinct, order-preserving columns to distinct observed Elo values. Local x-spacing is collision-resolved when 1–2 Elo differences cannot be represented literally.
- `:` denotes an uncertainty interval where the active figure uses one; exact `mean ± SEM` values appear in the table below the plot.
- G1 = Item Allocation; G2 = Diplomatic Treaty; G3 = Co-funding.

## Figure 1 — Negotiation episode workflow

**Old NeurIPS relationship:** exact old Figure 1; the number is unchanged.

```
+----------------------+       +-------------------------+
| PRIVATE PREFERENCES  |       | PUBLIC NEGOTIATION LOOP |
| Agent 1: Stone 43    |       +-------------------------+
|          Apple 33    |                    |
+----------+-----------+                    v
           +------------------------> [1. START]
                                           |
                                           v
                                    [2. DISCUSSION]
                                     share / conceal
                                     preferences
                                           |
                                           v
                                    [3. PROPOSALS]
                                     allocation +
                                     stated reasoning
                                           |
                                           v
                                    [4. VOTING]
                                  accept / reject
                                      /       \
                               unanimous       rejected
                                  |               |
                                  v               v
                            [FINAL OUTCOME] [5. REFLECTION]
                              utilities       diagnose round
                                                 |
                                                 +----> back to
                                                       discussion
```

| Stage | Measured state | Example from the figure |
|:--|:--|:--|
| Initialization | Private utility vector | Stone 43; Apple 33; Jewel 5 |
| Discussion | Natural-language transcript | Agents reveal or frame preferences |
| Proposal | Structured allocation + rationale | Agent 1 requests Apple and Stone |
| Voting | Per-agent accept/reject | Agent 1 accepts; Agent 2 rejects |
| Reflection | Private diagnosis | Agent 1 identifies a promising non-overlap |
| Termination | Allocation, utilities, consensus, rounds | Unanimous acceptance |

**Reading:** private preferences feed a public negotiation cycle; rejection causes reflection and another round, while unanimous acceptance terminates.

## Figure 2 — Bilateral capability scaling against GPT-5-nano

**Old NeurIPS relationship:** no exact counterpart; closest are old main-text Figures 2–3.

### Panel (a): adversary payoff

`o` = model mean; `:` = ±SEM; `.` = fitted trend; `*` = overlap.

```
              G1 Item Allocation                              G2 Diplomatic Treaty                                G3 Co-funding
 90 |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |   100 |                                        |    50 |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                    :  o|       |                                    :   |
    |                                        |       |                              ::::: :.**|       |                                :   :   |
 80 |                                :    : :|       |                  :   :      :o::**** : |       |                :           : : :   :   |
    |                               ::::  :::|    90 |               :: ::  :::    ::***:o::o |    40 |                :           : : ::::o:::|
    |                      :       ::::: ::::|       |               :o:o: ::::  ::***:o:: :: |       |                :        : :: ::o:::::**|
    |                      :       ::o:: :o::|       |             : ::::o::oo:: **: : : : o  |       |                o:   :   : :o o::*****::|
    |              :      ::: :    :o:o:::***|       |           : ::o:: ::::***.oo:   :   :  |       |                ::   ::: : :: :o**oo:ooo|
    |              :    : :o: : : :o::****:::|       |           : o:: o *****:: ::    :   :  |       |                :: : ::::o o: ***::: :::|
 70 |              :  : : ::: : :::::**::o:::|    80 |           : :::.*. o:  :::::        :  |       |                :: : o:::: :***::::: :::|
    |              :  : : o:o : :::** :::::::|       |       :   o :*..:  ::   o:             |    30 |                :o : :o:::.**:::  :   ::|
    |              o  : : ::: o o***:  :o:  :|       |       :   :.**  :  :    ::             |       |              :: ::o ::**..  :          |
    |              : :: : : : :.**o     ::   |       |       :   *. :          :o             |       |              :: :::.***o    o          |
    |              :::o o : :**.:::     :    |       |       o...              ::             |       |              :o ***. ::: :  :          |
    |              :::: : ...*: :::     :    |    70 |     ..*.                 :             |       |  :           o*..o:   :: :  :          |
    |           : ::::***.   :: :::          |       |   ... :                  :             |    20 |  :           **  :     : :  :          |
 60 |           : : :**::    :   ::          |       |  .                                     |       |  :         ..::  :       o             |
    |           : :.*::::    :               |       |                                        |       |  o       .*. :   :       :             |
    |           : *.:: :     o               |       |                                        |       |  :    :...:  :           :             |
    |  :    :   :.* :: : :   :               |    60 |                                        |       |  :   .*.  :              :             |
    |  :    :  .* o :: o :   :               |       |                                        |       |  : ...o   o                            |
    |  :    :.. : :    : :   : :             |       |                                        |    10 |  ..   :   :        :                   |
 50 |  :   .*   : :    : :   : :             |       |                                        |       |       :   :        :                   |
    |  o... :   : :    : :   : :             |       |  :                                     |       |       :   : :      o                   |
    |  *.   o   : :    : o     :             |    50 |  :                                     |       |             :      :                   |
    |  :    :     :      :     o             |       |  :                                     |       |             :                          |
    |  :    :            :     :             |       |  :                                     |     0 |             o                          |
    |  :    :            :     :             |       |  o                                     |       |             :                          |
    |       :            :     :             |       |  :                                     |       |             :                          |
 40 |       :            :     :             |    40 |  :                                     |       |             :                          |
    |                          :             |       |  :                                     |       |                                        |
    |                                        |       |  :                                     |       |                                        |
    |                                        |       |                                        |   -10 |                                        |
    +----------------------------------------+       +----------------------------------------+       +----------------------------------------+
     1100           1300                 1500         1100           1300                 1500         1100           1300                 1500
             Adversary Elo (ordered)                          Adversary Elo (ordered)                          Adversary Elo (ordered)
```

| Game | Model means | Payoff / 100 Elo | 95% slope CI | Typical point | R² |
|:--|:--|:--|:--|:--|:--|
| G1 Item Allocation | 30 | +6.75 | [4.56, 8.95] | mean ± 5.72 | 0.56 |
| G2 Diplomatic Treaty | 30 | +6.76 | [4.63, 8.89] | mean ± 4.28 | 0.58 |
| G3 Co-funding | 30 | +7.40 | [4.92, 9.89] | mean ± 5.19 | 0.55 |

### Panel (b): baseline payoff at competition endpoints

`C` = maximally cooperative EWM; `x` = maximally competitive EWM; `*` = overlap. All 30 distinct Elo values occupy distinct columns; local x-spacing is collision-resolved.

```
              G1 Item Allocation                              G2 Diplomatic Treaty                                G3 Co-funding
105 |                                        |   105 |                                        |    65 |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
100 |                                    CCCC|   100 |                                  CCCCCC|       |                                        |
    |                                 CCC    |       |                              CCCC      |    60 |                                        |
    |                                C       |       |                            CC          |       |                                        |
    |                               C        |       |                     CCCCCCC            |       |                                        |
    |                              C         |       |                    C                   |       |                                        |
 90 |                        CC   C          |    90 |                   C                    |       |                                        |
    |               C C     C C  C           |       |                 CC                     |       |                                        |
    |              C CC    C  C  C           |       |                C                       |       |                                        |
    |              C CC C  C   CC            |       |               C                        |       |                                        |
    |             C    CC C    C             |       |              C                         |       |                                        |
    |            C     C CC    C             |       |              C       x                 |    50 |                                        |
 80 |            C     C C                   |    80 |              C x   xx x                |       |                                        |
    |           C        C                   |       |             C  x  x   x                |       |                                        |
    |        CCC                             |       |             C x xx     x               |       |                                        |
    |  CCCCCC                                |       |            C x         x               |       |                                        |
    |                                        |       |            C x         x               |       |                                        |
 70 |                                        |    70 |            Cx          x xx            |       |                                        |
    |                                        |       |           C x           xx x           |       |                                        |
    |                                        |       |          C x            x  xxxx        |    40 |                                        |
    |                                        |       |        CC x                   x   x    |       |                                       C|
    |                                        |       |       C  x                     xxxx    |       |                                  C  CC |
    |                                        |       |      C  x                      x   x  x|       |                                 C CC   |
 60 |             xxx                        |    60 |    CC   x                          xxx |       |                             C CC   C   |
    |            x   x                       |       |   C    x                             x |       |                            C C         |
    |            x    xx                     |       |  C    x                                |       |                        C  C            |
    |            x      xx                   |       |      x                                 |       |                        C  C            |
    |           x         x    xx            |       |     x                                  |       |                C     CC CC             |
 50 |          x           x   x x           |    50 |    x                                   |    30 |                C CC  C   C             |
    |         x            x   x  xx         |       |    x                                   |       |              CC CCC C                  |
    |        x             x xx     x       x|       |   x                                    |       |  CC      CCCC   C  C                   |
    |      xx               x x      x    xx |       |  x                                     |       |    CC   C          C                   |
    |    xx                          x xxx   |       |                                        |       |      CCC                               |
 40 |  xx                            x x     |    40 |                                        |       |                                        |
    |                                 x      |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |    20 |                                        |
    |                                        |       |                                        |       |                                        |
 30 |                                        |    30 |                                        |       |                       x                |
    |                                        |       |                                        |       |                       x                |
    |                                        |       |                                        |       |                       xx  x            |
    |                                        |       |                                        |       |                       xx xx            |
    |                                        |       |                                        |       |           x           xx x xx          |
 20 |                                        |    20 |                                        |       |           xx      x   xxx  x x         |
    |                                        |       |                                        |    10 |          x  x     xx x  x    x         |
    |                                        |       |                                        |       |          x  x  x  xx x       x         |
    |                                        |       |                                        |       |          x   x x x x x        x        |
    |                                        |       |                                        |       |         x    xx xx  xx         x       |
    |                                        |       |                                        |       |         x        x   x          x      |
 10 |                                        |    10 |                                        |       |        x         x   x           x   x |
    |                                        |       |                                        |       |        x                         xx  xx|
    |                                        |       |                                        |       |        x                           xx  |
    |                                        |       |                                        |       |       x                            xx  |
    |                                        |       |                                        |     0 |  xxxxxx                                |
  0 |                                        |     0 |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
 -5 |                                        |    -5 |                                        |    -5 |                                        |
    +----------------------------------------+       +----------------------------------------+       +----------------------------------------+
     1100           1300                 1500         1100           1300                 1500         1100           1300                 1500
             Adversary Elo (ordered)                          Adversary Elo (ordered)                          Adversary Elo (ordered)
```

| Game | Adversary Elo band | Max-cooperative payoff | Max-competitive payoff |
|:--|:--|:--|:--|
| G1 | Q1 (1110–1302) | 87.5 ± 5.0 | 59.7 ± 6.6 |
| G1 | Q2 (1317–1358) | 85.5 ± 8.1 | 45.7 ± 3.9 |
| G1 | Q3 (1363–1448) | 91.9 ± 6.2 | 47.1 ± 3.9 |
| G1 | Q4 (1468–1504) | 100.0 ± 0.0 | 44.5 ± 2.7 |
| G2 | Q1 (1110–1302) | 88.8 ± 2.6 | 79.9 ± 6.2 |
| G2 | Q2 (1317–1358) | 94.6 ± 1.7 | 76.3 ± 4.7 |
| G2 | Q3 (1363–1448) | 97.3 ± 0.8 | 62.2 ± 4.1 |
| G2 | Q4 (1468–1504) | 99.2 ± 0.3 | 60.9 ± 4.1 |
| G3 | Q1 (1110–1302) | 28.2 ± 2.3 | 9.2 ± 5.1 |
| G3 | Q2 (1317–1358) | 32.6 ± 2.7 | 12.0 ± 7.1 |
| G3 | Q3 (1363–1448) | 34.8 ± 1.9 | 8.2 ± 4.2 |
| G3 | Q4 (1468–1504) | 39.6 ± 1.7 | 1.7 ± 2.6 |

**Reading:** adversary payoff rises in every game. Cooperation also lifts the baseline, whereas competition produces a flat-to-declining baseline curve.

## Figure 3 — Utility relative to fair share

**Old NeurIPS relationship:** no exact counterpart; closest are old Figure 5 (main text) and Figure 29 (appendix).

### Bilateral panel

`o` = model residual; `.` = fitted trend; `-` = zero/fair-share line.

```
              G1 Item Allocation                              G2 Diplomatic Treaty                                G3 Co-funding
 15 |                                        |    15 |                                        |    15 |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
 10 |                                        |    10 |                                        |    10 |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                 o      |       |                                        |       |                                        |
    |                                    oo..|       |                                        |       |                                        |
  5 |                                  .... o|     5 |                                        |     5 |  o                                     |
    |                   o   o        ..o   o |       |                                        |       |                              o         |
    |                               .o       |       |                                     ...|       |                            o     ooo   |
    |                            o...   o    |       |                                   .o.oo|       |                       o o      oo  ..o.|
  0 |-----------------o-------o-***oo--------|     0 |-------------------------------o-**-----|     0 |----------------o-------------*******o-o|
    |                  o   o   ..            |       |                              o.o oo    |       |       o         o     ....o..          |
    |                     o  ...             |       |                             o..        |       |               ......o..                |
    |                o    ...                |       |                        o    ..      o  |       |           .....   oo   o               |
    |              o   ...     o             |       |                  o        oo.          |       |    .......    o  o       o  o          |
 -5 |                ..                      |    -5 |                   o  oo  ..     o      |    -5 |  ...                 o                 |
    |               ..          o            |       |                o       .o.             |       |                                        |
    |             ..o    o                   |       |             o o      ...               |       |           o  o                         |
    |            ..          o               |       |           o     o ..o                  |       |             o                          |
-10 |           ..o                          |   -10 |                 ..                     |   -10 |                                        |
    |         ...                            |       |                ..        o             |       |                                        |
    |  o    ... o                            |       |               .                        |       |                                        |
    |     ...                                |       |             ..     o                   |       |                                        |
    |   ...                                  |       |            ..                          |       |                                        |
-15 |  ..                                    |   -15 |       o   ..                           |   -15 |                                        |
    |                                        |       |          ..  o                         |       |                                        |
    |                                        |       |        ..                              |       |                                        |
    |                                        |       |       ..                               |       |                                        |
-20 |                                        |   -20 |     ..                                 |   -20 |                                        |
    |                                        |       |   ...                                  |       |                                        |
    |                                        |       |  ..                                    |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
-25 |       o                                |   -25 |                                        |   -25 |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
-30 |                                        |   -30 |  o                                     |   -30 |                                        |
    +----------------------------------------+       +----------------------------------------+       +----------------------------------------+
     1100           1300                 1500         1100           1300                 1500         1100           1300                 1500
             Reference Elo (ordered)                          Reference Elo (ordered)                          Reference Elo (ordered)
```

### Multi-agent panel

`H` = heterogeneous focal agent; `A` = inserted adversary; `B` = baseline-agent mean; `O` = homogeneous control; `-` = zero/fair-share line.

```
    |                                                       |
    |                                                       |
  4 |                                                       |
    |                                                       |
    |                                                       |
    |                                                       |
    |                                                 A     |
  2 |                                               AA      |
    |                                             AA    H   |
    |                                           AA     HH H |
    |                                         AA      H  H  |
    |                                      AAA       H   H  |
  0 |------------------------------------AA----------H------|
    |                                 AAA HHHH       H      |
    |                               AA   H    HHHH  H       |
    |                          H   A     H        HH        |
 -2 |                          H  A HHH  H                  |
    |                          H A H   HH                   |
    |                         HH A H                        |
    |                         HHA H                         |
    |                        H AH H                         |
 -4 |                  H     HA HH                          |
    |                  H   HHA  HH                          |
    |             HH  HH   HA   H                           |
    |            H  HHHH   *                       BBBB     |
    |            H     H  *                  BBBBBB         |
 -6 |           H      H AH            BBBBBB               |
    |       HHHH       H AH      BBBBBB                     |
    |     HH            *HBBBBBBB                           |
    |    *BBBBBBBBBBBBB***                                  |
    |              AAAA HH O                                |
 -8 |        AAAAAA     HH                                  |
    |    AAAA           HH                                  |
    |                   H                                   |
    |                   H                                   |
-10 |                                                       |
    |                                                       |
    |                                                       |
    |                                                       |
    |                                                       |
-12 |                                                       |
    +-------------------------------------------------------+
       1240                       1389                  1504
                     Reference Elo (ordered)
```

| Series | Elo span | Fitted low-Elo residual | Fitted high-Elo residual | Δ / 100 Elo | Zero crossing |
|:--|:--|:--|:--|:--|:--|
| Bilateral — G1 | 1110–1504 | -15.69 | +6.07 | +5.52 | 1394 |
| Bilateral — G2 | 1110–1504 | -22.59 | +2.79 | +6.44 | 1461 |
| Bilateral — G3 | 1110–1504 | -5.06 | +1.24 | +1.60 | 1426 |
| Multi-agent — heterogeneous focal | 1240–1504 | -7.49 | +1.03 | +3.23 | 1472 |
| Multi-agent — inserted adversary | 1240–1484 | -9.51 | +2.55 | +4.94 | 1432 |
| Multi-agent — baseline mean | 1240–1484 | -7.61 | -5.31 | +0.94 | No crossing in range |
| Multi-agent — GPT-5-nano control | 1337 | -7.93 | -7.93 | — | — |

**Reading:** focal-agent trends rise through zero around Elo 1400–1470. Baseline agents improve only mildly and remain below fair share.

## Figure 4 — Strategic behavior in bilateral play

**Old NeurIPS relationship:** no counterpart; this systematic six-category analysis is new.

### Payoff association

`o` marks the correlation; `|` is zero.

```
                            Payoff correlation (Spearman rho)
                           ---------------------------------------+--------------------
Trade / compromise        +0.17                                        |=================o
Emotional persuasion      +0.13                                        |=============o
Logical persuasion        +0.11                                        |==========o
Pressure                  -0.05                                   o====|
Self-interest / exploitation -0.27             o==========================|
Formalization             -0.34       o================================|
                           -0.4              -0.2               +0.0               +0.2
```

### Behavior frequency versus capability

Each `o` curve is the same centered five-model smooth used by the figure after its one-outlier-per-category removal.

```
              Trade / compromise                            Emotional persuasion                            Logical persuasion
 2.2 |                                      |   0.8 |                                      |     2 |                                      |
     |                                      |       |                                      |       |                                      |
     |                                      |       |                                      |       |                                      |
     |                                      |       |                                      |       |                                      |
     |                                o     |       |                                      |       |                                      |
1.65 |                              oo ooooo|   0.6 |                                      |   1.5 |                                      |
     |                           o o        |       |               o              o       |       |                              o       |
     |                        o o o         |       |              o o            o o      |       |                             o o     o|
     |             oo o      o o            |       |             o   o          o   o     |       |                            o   ooooo |
     |            o  oo   o o               |       |            o     o       oo    o     |       |                           oo         |
 1.1 |     oooo  oo    o  oo                |   0.4 |           o      o     oo       o    |     1 |                          o           |
     |    o    oo       oo                  |       |         oo        o  oo          o   |       |                          o           |
     |  oo                                  |       |    ooooo           oo             ooo|       |                          o           |
     |                                      |       |  oo                                  |       |                         o            |
     |                                      |       |                                      |       |                   ooo oo             |
0.55 |                                      |   0.2 |                                      |   0.5 |            ooo  oo   o               |
     |                                      |       |                                      |       |  oooooooooo   oo                     |
     |                                      |       |                                      |       |                                      |
     |                                      |       |                                      |       |                                      |
     |                                      |       |                                      |       |                                      |
   0 |                                      |     0 |                                      |     0 |                                      |
     +--------------------------------------+       +--------------------------------------+       +--------------------------------------+
      1110               1350           1504         1110                1350          1504         1110               1350           1504
             Adversary Elo (ordered)                        Adversary Elo (ordered)                        Adversary Elo (ordered)

                   Pressure                             Self-interest / exploitation                           Formalization
 2.2 |                                      |   0.8 |                                      |     0.3 |                                      |
     |                                      |       |                                      |         |                                      |
     |                                      |       |                                      |         |                                      |
     |                                      |       |                                      |         |                                      |
     |                                      |       |                                      |         |                                      |
1.65 |                             o     ooo|   0.6 |                                      |   0.225 |                                      |
     |                            o ooooo   |       |                      o               |         |                                      |
     |                           o          |       |                     oo               |         |                                      |
     |                          o           |       |              o     o  o              |         |                               o o  o |
     |                         o            |       |             oo    o    oo   ooo oo   |         |              o              oo o oo o|
 1.1 |                        o             |   0.4 |     oooooooo  oo o       ooo   o  ooo|    0.15 |             oo o           o     o   |
     |                        o             |       |    o            o                    |         |            o  oo          o          |
     |                      oo              |       |  oo                                  |         |    ooooo  o     oo o     o           |
     |                     o                |       |                                      |         |  oo     oo        o oo  oo           |
     |                    o                 |       |                                      |         |                       oo             |
0.55 |                  ooo                 |   0.2 |                                      |   0.075 |                                      |
     |              o   o                   |       |                                      |         |                                      |
     |  oooooooooooo ooo                    |       |                                      |         |                                      |
     |                                      |       |                                      |         |                                      |
     |                                      |       |                                      |         |                                      |
   0 |                                      |     0 |                                      |       0 |                                      |
     +--------------------------------------+       +--------------------------------------+         +--------------------------------------+
      1110                1350          1504         1110                1350          1504           1110               1350           1504
             Adversary Elo (ordered)                        Adversary Elo (ordered)                          Adversary Elo (ordered)
```

| Behavior | Payoff Spearman ρ | Payoff p | Δ events / rollout / 100 Elo |
|:--|:--|:--|:--|
| Trade / compromise | +0.17 | 2.1e-14 | +0.22 |
| Emotional persuasion | +0.13 | 4.3e-09 | +0.01 |
| Logical persuasion | +0.11 | 2e-06 | +0.32 |
| Pressure | -0.05 | 0.027 | +0.49 |
| Self-interest / exploitation | -0.27 | 8.7e-34 | +0.01 |
| Formalization | -0.34 | 9.5e-53 | +0.02 |

**Reading:** trade/compromise and logical persuasion both rise with capability and predict payoff. Pressure rises fastest but is slightly payoff-negative.

## Figure 5 — Test-time compute payoff

**Old NeurIPS relationship:** no exact counterpart; closest is old appendix Figure 18.

`o` = mean payoff; `:` = ±SEM; `.` connects effort levels.

```
              GPT-5                          Claude Sonnet 4.6                      Gemini 3 Flash
    |                            |       |                            |       |                            |
    |                            |       |                            |       |                            |
 80 |                            |    80 |                            |    80 |                            |
    |                            |       |                         :  |       |                            |
    |                            |       |  :       :      :       :  |       |                            |
    |                            |       |  :       :      :       :  |       |          :                 |
    |                 :       :  |       |  :       :      :       :  |       |          :                 |
 75 |                 :       :  |    75 |  :       :      :       :  |    75 |          :      :       :  |
    |  :              :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
    |  :       :      :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
    |  :       :      :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
    |  :       :      :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
 70 |  :       :      :       :  |    70 |  :       :     .*.......*  |    70 |  :       :      :       :  |
    |  :       :      :       :  |       |  :       : .... :       :  |       |  :       :      :       :  |
    |  :       :      :       :  |       |  *.......*.     :       :  |       |  :     ..*.     :       :  |
    |  :       :      :       :  |       |  :       :      :       :  |       |  :  ...  : ..   :       :  |
    |  :       :      :   ....*  |       |  :       :      :       :  |       |  *..     :   .. :   ....*  |
 65 |  :       :     .*...    :  |    65 |  :       :      :       :  |    65 |  :       :     .*...    :  |
    |  :       :   .. :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
    |  :       : ..   :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
    |  *.......*.     :       :  |       |  :       :      :       :  |       |  :       :      :       :  |
    |  :       :      :       :  |       |  :       :      :          |       |  :       :      :       :  |
 60 |  :       :      :       :  |    60 |  :       :                 |    60 |  :       :      :       :  |
    |  :       :      :       :  |       |  :       :                 |       |  :       :      :       :  |
    |  :       :      :       :  |       |          :                 |       |  :       :      :       :  |
    |  :       :      :       :  |       |                            |       |                 :       :  |
    |  :       :      :       :  |       |                            |       |                 :       :  |
 55 |  :       :      :       :  |    55 |                            |    55 |                 :          |
    |  :       :      :          |       |                            |       |                            |
    |  :       :                 |       |                            |       |                            |
    |  :       :                 |       |                            |       |                            |
    |  :       :                 |       |                            |       |                            |
 50 |          :                 |    50 |                            |    50 |                            |
    |                            |       |                            |       |                            |
    |                            |       |                            |       |                            |
    +----------------------------+       +----------------------------+       +----------------------------+
      min     low    med    high           low     med   high     max           min     low    med    high
           Requested effort                     Requested effort                     Requested effort
```

| Model | Requested effort | Mean payoff ± SEM | Observed tokens / call | Game cells |
|:--|:--|:--|:--|:--|
| GPT-5 | Minimal | 62.5 ± 11.4 | 0 | 9 |
| GPT-5 | Low | 61.6 ± 11.2 | 395 | 9 |
| GPT-5 | Medium | 65.3 ± 11.2 | 1,494 | 9 |
| GPT-5 | High | 65.7 ± 10.3 | 1,342 | 9 |
| Claude Sonnet 4.6 | Low | 68.2 ± 9.6 | 1,490 | 9 |
| Claude Sonnet 4.6 | Medium | 68.0 ± 9.8 | 1,712 | 9 |
| Claude Sonnet 4.6 | High | 69.5 ± 8.6 | 1,479 | 9 |
| Claude Sonnet 4.6 | Max | 70.4 ± 8.5 | 1,228 | 9 |
| Gemini 3 Flash | Minimal | 65.7 ± 8.1 | 241 | 9 |
| Gemini 3 Flash | Low | 67.6 ± 9.2 | 357 | 9 |
| Gemini 3 Flash | Medium | 65.1 ± 9.8 | 1,677 | 9 |
| Gemini 3 Flash | High | 65.6 ± 9.6 | 1,950 | 9 |

**Reading:** all effort-level means lie inside broad, strongly overlapping SEM intervals despite large token-count changes.

## Figure 6 — Test-time compute and strategic behavior

**Old NeurIPS relationship:** no counterpart; this TTC behavior analysis is new.

`G` = GPT-5; `M` = Gemini 3 Flash; `*` = overlap. Each point averages 18 rollouts.

```
           Trade / compromise                         Emotional persuasion                        Logical persuasion
4.8 |                                  |      1 |                                  |   4.8 |                                  |
    |                                  |        |                                  |       |                                  |
    |          GGGG                    |        |                                  |       |                   MMMMM          |
    |  MMMGGGGG    GG              GG  |        |                                  |       |               MMMM     MMMMM     |
    |  GGGMMMMM      GG        GGGG    |        |                              GG  |       |           MMMM              MMM  |
    |          MMMMMMMM**MMM*GG        |        |  MMMMMMMMMMMMMMM          GGG    |       |          M                       |
3.6 |                    GGG MMMMM     |   0.75 |                 MMMMMM****MMMMM  |   3.6 |       MMM                    GG  |
    |                             MMM  |        |                   GGGG           |       |      M                     GG    |
    |                                  |        |               GGGG               |       |   MMM                   GGG      |
    |                                  |        |          GGGGG                   |       |  *GGGGGGGGGGGGGG      GG         |
    |                                  |        |     GGGGG                        |       |                 GGGGGG           |
2.4 |                                  |    0.5 |  GGG                             |   2.4 |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
1.2 |                                  |   0.25 |                                  |   1.2 |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
    |                                  |        |                                  |       |                                  |
  0 |                                  |      0 |                                  |     0 |                                  |
    +----------------------------------+        +----------------------------------+       +----------------------------------+
      min       low      med      high            min       low      med      high           min       low      med      high
              Requested effort                            Requested effort                           Requested effort

                Pressure                          Self-interest / exploitation                      Formalization
  4 |                                  |      3 |                                  |   1.6 |                                  |
    |                                  |        |                                  |       |                                  |
    |           GGGGGGGGGG***MMM       |        |  MMM                             |       |                                  |
    |         GG         M   GGG**MMM  |        |     MMMMM                        |       |                               M  |
    |      GGG         MM         GGG  |        |          MMM                 GG  |       |                              M   |
    |    GG           M                |        |             MM              G M  |   1.2 |                            MM    |
  3 |  GG            M                 |   2.25 |            GG M          GG*MM   |       |                           M      |
    |              MM                  |        |          GG  GG*M       G M      |       |                          M       |
    |             M                    |        |         G       G*G  GG*MM       |       |                         M        |
    |  MMMMMMMMMMM                     |        |        G          M*G M          |       |  G                    MM         |
    |                                  |        |      GG             MM           |       |   G                  M           |
  2 |                                  |    1.5 |     G                            |   0.8 |    G                M            |
    |                                  |        |   GG                             |       |     G             MM             |
    |                                  |        |  G                               |       |      G           M               |
    |                                  |        |                                  |       |       G        MM                |
    |                                  |        |                                  |       |       G       M              GG  |
  1 |                                  |   0.75 |                                  |       |        G    MM             GG    |
    |                                  |        |                                  |   0.4 |  MMMMMMM*MMM            GGG      |
    |                                  |        |                                  |       |          G            GG         |
    |                                  |        |                                  |       |           G     GGGGGG           |
    |                                  |        |                                  |       |            GGGGG                 |
    |                                  |        |                                  |       |                                  |
  0 |                                  |      0 |                                  |     0 |                                  |
    +----------------------------------+        +----------------------------------+       +----------------------------------+
      min       low      med      high            min       low      med      high           min       low      med      high
              Requested effort                            Requested effort                           Requested effort
```

#### GPT-5

| Behavior | Minimal | Low | Medium | High | Minimal → high |
|:--|:--|:--|:--|:--|:--|
| Trade / compromise | 3.83 | 4.39 | 3.50 | 4.06 | +0.22 |
| Emotional persuasion | 0.50 | 0.61 | 0.67 | 0.83 | +0.33 |
| Logical persuasion | 2.78 | 2.78 | 2.72 | 3.56 | +0.78 |
| Pressure | 2.89 | 3.61 | 3.56 | 3.22 | +0.33 |
| Self-interest / exploitation | 1.28 | 2.22 | 1.83 | 2.44 | +1.17 |
| Formalization | 0.94 | 0.11 | 0.22 | 0.50 | -0.44 |

#### Gemini 3 Flash

| Behavior | Minimal | Low | Medium | High | Minimal → high |
|:--|:--|:--|:--|:--|:--|
| Trade / compromise | 4.06 | 3.67 | 3.61 | 3.28 | -0.78 |
| Emotional persuasion | 0.78 | 0.78 | 0.72 | 0.72 | -0.06 |
| Logical persuasion | 2.94 | 3.83 | 4.28 | 4.00 | +1.06 |
| Pressure | 2.28 | 2.33 | 3.61 | 3.39 | +1.11 |
| Self-interest / exploitation | 2.67 | 2.39 | 1.61 | 2.33 | -0.33 |
| Formalization | 0.33 | 0.39 | 0.78 | 1.39 | +1.06 |

**Reading:** reasoning effort changes the behavioral mix, but it simultaneously strengthens payoff-positive and payoff-negative behaviors.

## Figure 7 — Heterogeneous Game 1 payoff scaling

**Old NeurIPS relationship:** no exact standalone counterpart; its Game 1 panel appeared within old main-text Figure 7.

`o` = model-level mean; `.` = fitted trend. Every distinct model Elo gets a distinct, order-preserving column.

```
                     N=2                                              N=4                                              N=6
100 |                                        |    75 |                                        |    65 |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                 o      |       |                                        |       |                                        |
    |                                        |       |                                    o   |       |                                        |
    |                                        |       |                                        |       |                                        |
 88 |                                        |    64 |                                        |    57 |                       o                |
    |                                        |       |                                        |       |                                   o    |
    |                                        |       |                             oo       o |       |                  o                     |
    |                                      o |       |                    o               ....|       |                o                      o|
    |                                        |       |                  o      o    .......  o|       |                                        |
    |                   o         oo         |       |         o             o ......      o  |       |                                        |
 76 |                                  o ....|    53 |   o         o   o o .....        oo    |    49 |                                 o   o .|
    |                               .....o   |       |                ......     o     o      |       |                             o......... |
    |                       o o  ....        |       |          ......o                       |       |     o              o  ........   o   o |
    |                o        ....           |       |     ......                             |       |             o  ...o....   o            |
    |     o                ...          o o  |       |   ...                                  |       |        ....o...                    o   |
    |                 ...o.                 o|       |     o o                                |       |   ......o       o                      |
 64 |            o ....                      |    42 |                                        |    41 |       o                                |
    |          .....            o            |       |                                        |       |                                        |
    |   o  .o...  o   o                      |       |              o                         |       |              o                         |
    |   ....                                 |       |                                        |       |   o                                    |
    |   .                                    |       |                                        |       |                              o         |
    |         o                              |       |                                        |       |                         o              |
 52 |              o                         |    31 |                                        |    33 |                                        |
    |                                        |       |            o                           |       |                                        |
    |                  o                     |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
    |                                        |       |                                        |       |                                        |
 40 |                                        |    20 |                                        |    25 |                                        |
    +----------------------------------------+       +----------------------------------------+       +----------------------------------------+
      1240                1389           1504          1240                1389           1504          1240                1389           1504
               Arena Elo (ordered)                              Arena Elo (ordered)                              Arena Elo (ordered)

                     N=8                                              N=10
 60 |                                        |    55 |                                        |
    |                                        |       |                                        |
    |                                        |       |                                        |
    |                                        |       |                                        |
    |                                        |       |                                        |
    |                                        |       |                                        |
 53 |                                        |    48 |                o                       |
    |                                        |       |                                    o   |
    |                 o                      |       |                                        |
    |                             o          |       |            o                       ....|
    |                                 ooo    |       |                       o      o ...o o o|
    |                                        |       |                             ...  o   o |
 46 |                                        |    41 |                         o.o..   o      |
    |                o  o               .....|       |                   o   ....             |
    |         o        o           o..... oo |       |             o       ...     o          |
    |                        ......          |       |                 ....                   |
    |                    .....               |       |               ..oo                     |
    |            oo .....o               o   |       |           ....     o                   |
 39 |         ......                        o|    34 |       o...                             |
    |    .o....             o o              |       |     ....                               |
    |   ..                                   |       |   ...                                  |
    |                                        |       |              o                         |
    |       o                   o            |       |         o                              |
    |                                        |       |     o                                  |
 32 |                                        |    27 |                                        |
    |   o          o                         |       |   o                                    |
    |                                        |       |                                        |
    |                                        |       |                                        |
    |                                        |       |                                        |
    |                                        |       |                                        |
 25 |                                        |    20 |                                        |
    +----------------------------------------+       +----------------------------------------+
      1240                1389           1504          1240                1389           1504
               Arena Elo (ordered)                              Arena Elo (ordered)
```

| Group size | Models | Mean observations / model | Payoff / 100 Elo | R² |
|:--|:--|:--|:--|:--|
| N=2 | 24 | 8.3 | +7.49 | 0.34 |
| N=4 | 24 | 16.7 | +4.99 | 0.28 |
| N=6 | 24 | 25.0 | +2.42 | 0.11 |
| N=8 | 24 | 33.3 | +3.06 | 0.20 |
| N=10 | 24 | 41.7 | +5.24 | 0.52 |

**Reading:** the fitted trend is positive for every tested group size, although its strength and explanatory power vary with N.

## Figure 8 — Heterogeneous versus homogeneous-control Gini

**Old NeurIPS relationship:** no counterpart; the 325-run random monoculture control is new.

### Panel (a): aggregate comparison

`[---o---]` is mean ± SEM.

```
Heterogeneous rosters    0.162 ± 0.006 |                          [-----o-----]                     |
Homogeneous monocultures 0.156 ± 0.011 |               [----------o---------]                       |
                                        0.13         0.15           0.16          0.17          0.19
```

| Roster condition | Runs | Corrected Gini ± SEM |
|:--|:--|:--|
| Heterogeneous random rosters | 1,300 | 0.162 ± 0.006 |
| Homogeneous monocultures | 325 | 0.156 ± 0.011 |

### Panel (b): monoculture capability

`1`, `2`, and `3` are G1–G3 model means; `:` is ±SEM; `.` is the run-level fit; `-` is the heterogeneous mean and `=` bounds its ±SEM envelope.

```
    |                                                                      |
    |                                                                      |
0.5 |           :                                                          |
    |           :                                                          |
    |           :                                                          |
    |           :                                                          |
    |           1                                                          |
    |           :                                                          |
    |           :                                                          |
    |           :                                                          |
0.4 |           :                                                          |
    |      :                                                               |
    |      :                                                               |
    |      :                                                               |
    |      :                                                               |
    |      :                                                               |
    |      3                                                               |
    |      :                                                               |
0.3 |      :                         :                                     |
    |      :                         :                                     |
    |      *.                        :                                     |
    |      :.....                    :                                     |
    |           ....                 3                                     |
    |               ....             :                                     |
    |                  .....    :    :             :                       |
    |                      .... :    :             :              :        |
0.2 |                         ..*..  :             :              :        |
    |                           1 .....            3              1        |
    |      ==========================================================      |
    |---------------------------*---------****-----*--*------**---*--------|
    |      ==========================================================      |
    |                                            .....1      13            |
    |                                                .*...   ::            |
    |                                                 :  ....:: :          |
0.1 |                                                       ..*.:          |
    |                                                           *...       |
    |                                                           :   .      |
    |                 :                                         :          |
    |                 2           2                                 :      |
    |                 :           :                            :    2      |
    |                                                          2    :      |
    |                                                                      |
0.0 |                                 2                                    |
    |                                                                      |
    |                                                                      |
    +----------------------------------------------------------------------+
           1250       1300       1350       1400       1450       1500
                        Monoculture model Elo (ordered)
```

| Game | Monoculture model | Elo | Runs | Corrected Gini ± SEM |
|:--|:--|:--|:--|:--|
| G1 | Claude 3 Haiku | 1260 | 25 | 0.448 ± 0.052 |
| G1 | GPT-5 nano | 1337 | 25 | 0.193 ± 0.027 |
| G1 | Qwen3 Max | 1435 | 25 | 0.138 ± 0.022 |
| G1 | Opus 4.5 | 1468 | 25 | 0.141 ± 0.023 |
| G1 | Gemini 3.1 Pro | 1494 | 25 | 0.187 ± 0.031 |
| G2 | Nova Pro | 1290 | 20 | 0.052 ± 0.011 |
| G2 | GPT-4o | 1345 | 20 | 0.046 ± 0.008 |
| G2 | o3-mini | 1363 | 20 | 0.003 ± 0.002 |
| G2 | GPT-5.2 Chat | 1478 | 20 | 0.027 ± 0.006 |
| G2 | Opus 4.6 | 1499 | 20 | 0.037 ± 0.010 |
| G3 | Nova Micro | 1240 | 20 | 0.325 ± 0.068 |
| G3 | DeepSeek V3 | 1358 | 20 | 0.252 ± 0.053 |
| G3 | DeepSeek R1 | 1422 | 20 | 0.190 ± 0.033 |
| G3 | Opus 4.5 Think | 1474 | 20 | 0.134 ± 0.034 |
| G3 | GPT-5.4 High | 1484 | 20 | 0.090 ± 0.028 |

**Reading:** aggregate heterogeneous and homogeneous inequality are nearly tied; within monocultures, capability is the stronger gradient.

## Figure 9 — Homogeneous-adversary inequality and role payoff

**Old NeurIPS relationship:** no exact counterpart; closest are old main-text Figure 8 and appendix Figure 27.

Left: `o` = baseline-only Gini mean and `:` = ±SEM. Right: `A` = adversary payoff and `B` = mean baseline-agent payoff.

```
                Baseline-only Gini                                     Role payoff
     |                                          |     60 |                                          |
     |                                          |        |                                          |
     |                                          |        |                                          |
0.22 |                                          |        |                                          |
     |                                          |        |                                       A  |
     |                                          |   57.5 |                                      A   |
     |                                          |        |                                    AA    |
     |                                          |        |                                   A      |
     |                                          |        |                                  A       |
0.20 |                                          |        |                                 A        |
     |  :                                       |     55 |                                A         |
     |  :                                       |        |                               A          |
     |  :                                       |        |                             AA           |
     |  oo                                      |        |                            A             |
     |  : oo        :                           |        |                          AA              |
0.18 |  :   oo      :                           |   52.5 |                       AAA                |
     |  :     ooo   :                           |        |                   AAAA                   |
     |  :        oo :                           |        |                AAA                       |
     |             oo                           |        |              AA                          |
     |              :oo                         |        |             A                            |
     |              :  oo                       |     50 |            A                             |
     |              :    o                      |        |           A                              |
0.16 |              :     oo     :              |        |           A                              |
     |                      o    :           :  |        |          A                           BB  |
     |                       oo  :           :  |        |         A                         BBB    |
     |                         oo:           :  |   47.5 |        A                       BBB       |
     |                           ooooooo     :  |        |       A                     BBB          |
     |                           :      oooooo  |        |      A                 BBBBB             |
0.14 |                           :           :  |        |  BBB*BBB         BBBBBB                  |
     |                           :           :  |        |     A   BBBBBBBBB                        |
     |                           :           :  |     45 |    A                                     |
     |                                       :  |        |   A                                      |
     |                                          |        |  A                                       |
     |                                          |        |                                          |
0.12 |                                          |        |                                          |
     |                                          |   42.5 |                                          |
     |                                          |        |                                          |
     |                                          |        |                                          |
     |                                          |        |                                          |
     |                                          |        |                                          |
0.10 |                                          |     40 |                                          |
     +------------------------------------------+        +------------------------------------------+
       Q1          Q2           Q3          Q4             Q1          Q2           Q3          Q4
                Adversary Elo quartile                              Adversary Elo quartile
```

| Adversary Elo quartile | Runs | Baseline-only Gini ± SEM | Adversary payoff | Mean baseline payoff | Adversary gap |
|:--|:--|:--|:--|:--|:--|
| Q1 1240-1317 | 520 | 0.186 ± 0.010 | 44.2 | 45.8 | -1.5 |
| Q2 1389-1389 | 260 | 0.172 ± 0.014 | 51.0 | 45.6 | +5.4 |
| Q3 1448-1448 | 260 | 0.144 ± 0.013 | 52.9 | 46.4 | +6.6 |
| Q4 1484-1484 | 260 | 0.142 ± 0.013 | 58.0 | 48.3 | +9.6 |

**Reading:** the baseline fleet becomes internally more equal while the inserted adversary's payoff advantage grows from -1.5 to +9.6.

## Source scope

| Figure | Active numerical scope |
|:--|:--|
| 1 | Manual conceptual workflow |
| 2 | 1,500 primary bilateral runs; 30 adversary models |
| 3 | 1,500 bilateral + 1,300 heterogeneous + 1,300 homogeneous-adversary + 130 control runs |
| 4 | 1,920 accepted qualitative rollouts; 1,891 payoff-valid |
| 5 | 216 TTC runs; 18 per family-effort cell |
| 6 | 144 displayed GPT-5/Gemini TTC rollouts |
| 7 | 1,300 canonical heterogeneous multi-agent runs; G1 shown |
| 8 | 1,300 heterogeneous + 325 monoculture-control runs |
| 9 | 1,300 canonical homogeneous-adversary runs |
