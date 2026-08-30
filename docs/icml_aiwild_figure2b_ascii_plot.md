# ICML AIWILD Figure 2(b) — text-only plot

This is a standalone ASCII rendering of Figure 2(b) from the most recent
compiled ICML AIWILD manuscript. It is separate from the table-based figure
translation and does not modify the paper.

`C` traces the maximally cooperative curve, `x` traces the maximally
competitive curve, and `*` marks an overlap. The curves use the same
exponentially weighted means as the active figure (`alpha = 0.24`).

The three panels are arranged in one row and three columns. Each plotting area
is 60 character rows tall and 40 characters wide, so the panels remain taller
than they are wide while providing substantially more horizontal resolution.
All 30 observed Elo values are distinct and occupy distinct character columns.
Their ordering is exact; because some models differ by only 1–2 Elo, local
spacing is collision-resolved rather than perfectly linear. The vertical
resolution is unchanged at 1.83 payoff units per row in G1–G2 and 1.17 payoff
units per row in G3. The y-ranges match the active figure: -5 to 105 in G1–G2
and -5 to 65 in G3.

```
              G1 Item Allocation                              G2 Diplomatic Treaty                                G3 Co-funding
               Baseline payoff                                  Baseline payoff                                  Baseline payoff
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
     1100           1300                 1500        1100           1300                 1500        1100           1300                 1500
         Adversary Elo (order-preserving)                Adversary Elo (order-preserving)                Adversary Elo (order-preserving)

C = maximally cooperative EWM mean
x = maximally competitive EWM mean
* = overlap
```

The main visual contrast is preserved: cooperative baseline payoff rises with
adversary capability, while competitive baseline payoff is flat-to-declining,
especially in G2 and G3.
