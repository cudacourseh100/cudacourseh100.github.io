# CUDA Programming for NVIDIA H100s Site

Static site and GitHub Pages bundle for **CUDA Programming for NVIDIA H100s** by **Prateek Shukla**.

## Local Preview

1. Build the GitHub Pages bundle:

   `node scripts/build-pages.mjs`

2. Serve the generated `docs/` folder:

   `python3 -m http.server 4173 --directory docs`

3. Open:

   `http://127.0.0.1:4173`

## GitHub Pages

The repo includes a workflow at `.github/workflows/pages.yml` that builds the site from source and deploys it to GitHub Pages on pushes to `main` or `master`.

The workflow runs:

`node scripts/build-pages.mjs`

You can still rebuild locally before pushing if you want to preview the exact Pages bundle:

`node scripts/build-pages.mjs`
