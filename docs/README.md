# The documentation site

Built with [MyST](https://mystmd.org) from:

- the pages in this folder (`index.md`, `installation.md`, `guide/`);
- the tutorials in `examples/tutorials/`, converted to notebooks and executed
  (`docs/tutorials` is a symlink to that folder);
- the READMEs of the applications in `examples/`;
- the API reference, written from the package's docstrings by `generate_api_pages.py`.

The table of contents is `myst.yml`. CI builds the site on every pull request and
publishes it to GitHub Pages from `main` (`.github/workflows/docs.yml`).

## Building locally

Inside the DOLFINx container, with the `docs` extra installed
(`python3 -m pip install -e ".[docs]"`):

```bash
# 1. The tutorials, as notebooks next to the scripts (gitignored)
for script in examples/tutorials/[0-9]*.py; do
    jupytext --to ipynb --set-kernel python3 "${script}"
done

# 2. The API reference
python3 docs/generate_api_pages.py docs/api

# 3. The site; --execute runs the tutorials, one at a time (about 4 minutes)
cd docs
myst build --html --execute --execute-parallel 1
python3 -m http.server -d _build/html    # or: myst start --execute
```

MyST bootstraps its own Node.js on first use, or uses one already installed. Two
problems you may meet:

- the build stops with `ECONNREFUSED 127.0.0.1`: set
  `NODE_OPTIONS=--dns-result-order=ipv4first`;
- `Jupyter server did not start`, as root (in the DOLFINx container): allow it with a
  Jupyter config, as CI does:

  ```bash
  export JUPYTER_CONFIG_DIR=/tmp/jupyter-config
  mkdir -p $JUPYTER_CONFIG_DIR
  echo "c.ServerApp.allow_root = True" > $JUPYTER_CONFIG_DIR/jupyter_server_config.py
  ```

Figures appear in the pages only with an inline matplotlib backend:
`export MPLBACKEND=module://matplotlib_inline.backend_inline`.

A new module needs a line in the API reference section of `myst.yml`, and a new tutorial
a line in the tutorials section. CI checks that both are listed.
