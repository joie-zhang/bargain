# Root import cross-check

- Parsed the 284 positive-inventory Python files and followed their local imports without executing them.
- The resulting closure has 290 Python files and no parse errors.
- Found six additional support files required by imported analysis or maintained visualization code.
- Manually checked the four import sites that establish these links.
- The companion JSON records exact source lines and import edges.
- This checks static imports, not every dynamic path, historical source revision or possible manual command.
- Conditional imports remain protected without claiming that every paper run executes them.
- The six files were added to the consolidated keep inventory.
- No candidate overlaps these additions.

The added files are two analysis-package files, the paper-figure package initializer, and three shared analysis loaders.
Their complete paths and reasons are in the companion JSON.
