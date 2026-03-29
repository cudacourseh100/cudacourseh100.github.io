# SEO Indexing Checklist

Last reviewed: 2026-03-29

Use this checklist after each meaningful site update.

## Technical Checks

- [ ] Homepage returns `200 OK`
- [ ] `robots.txt` returns `200 OK`
- [ ] `sitemap.xml` returns `200 OK`
- [ ] Homepage canonical points to `https://cudacourseh100.github.io/`
- [ ] Lesson pages have unique titles and descriptions
- [ ] Lesson pages are listed in `sitemap.xml`
- [ ] No production page has `noindex`
- [ ] Social card URLs resolve correctly

## Search Console

- [ ] Site verified in Google Search Console
- [ ] Sitemap submitted
- [ ] Homepage requested for indexing
- [ ] `slides.html` requested for indexing
- [ ] Top lesson pages requested for indexing
- [ ] Coverage report checked for crawl or canonical issues

## Bing Webmaster

- [ ] Site verified in Bing Webmaster Tools
- [ ] Sitemap submitted
- [ ] Crawl diagnostics checked

## Distribution

- [ ] FreeCodeCamp course page links to the site
- [ ] GitHub repo or profile links to the site
- [ ] Social posts link to the site
- [ ] Any course launch content links to the site

## Content Expansion

- [ ] FAQ section exists or is planned
- [ ] Glossary page exists or is planned
- [ ] Internal links between lessons are present
- [ ] About / instructor page exists or is planned

## Repo Commands

```sh
python3 scripts/export-lesson-markdown.py
node scripts/build-pages.mjs
```

## Public URLs To Check

```text
https://cudacourseh100.github.io/
https://cudacourseh100.github.io/robots.txt
https://cudacourseh100.github.io/sitemap.xml
https://cudacourseh100.github.io/slides.html
https://cudacourseh100.github.io/pages/lesson-1.html
```
