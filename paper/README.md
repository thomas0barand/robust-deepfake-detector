# LaTeX Compilation in VS Code

This folder uses the IEEE conference template (`IEEEtran`).

## Windows

1. Install a TeX distribution (recommended: MiKTeX).
2. (If using `latexmk`) install Perl, then confirm:
   - `perl -v`
   - `latexmk -v`
3. Install the VS Code extension **LaTeX Workshop** (James Yu).
4. Install the IEEE class package in your TeX distribution:
   - package name: `ieeetran`
5. Verify your compiler/tool paths:
   - `where pdflatex`
   - `where latexmk`
   - `kpsewhich IEEEtran.cls`
6. In VS Code, open `paper/main.tex` and build with `Ctrl+Alt+B`.

Notes:
- If `spawn latexmk ENOENT` appears, `latexmk` is not installed or not on PATH.
- If `IEEEtran.cls` is not found, install `ieeetran` and verify with `kpsewhich IEEEtran.cls`.

## macOS

1. Install **MacTeX** (full distribution) or BasicTeX + required packages.
2. Install VS Code extension **LaTeX Workshop** (James Yu).
3. Verify tools in a new terminal:
   - `pdflatex --version`
   - `latexmk -v`
   - `kpsewhich IEEEtran.cls`
4. If `IEEEtran.cls` is missing, install it:
   - `sudo tlmgr install IEEEtran`
5. Open `paper/main.tex` and build with `Cmd+Option+B`.

## Build output

Generated LaTeX artifacts (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, etc.) should not be committed.
Keep source files (`.tex`, `.bib`, figures) under version control.