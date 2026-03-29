# SEO Strategy

Last reviewed: 2026-03-29

This document is the fetchable SEO strategy for the CUDA Programming for NVIDIA H100s site.

## Current State

- Live site: `https://cudacourseh100.github.io/`
- Homepage status: `200 OK`
- `robots.txt` status: `200 OK`
- `sitemap.xml` status: `200 OK`
- Canonical tags: present on the homepage and lesson pages via the build pipeline
- Open Graph and Twitter tags: present
- JSON-LD structured data: present
- Search visibility concern: the site does not appear to be surfacing in search results yet for obvious branded queries

## What This Means

The site is crawlable.

The current issue appears to be indexation and authority, not an obvious crawler block.

That distinction matters:

- Crawlability = search engines are allowed to fetch the site.
- Indexation = search engines decide to keep pages in the index.
- Ranking = indexed pages earn visibility for queries.

Right now the technical crawl signals are in place, but public visibility still looks weak.

## Likely Reasons The Site Is Not Showing Yet

- The site is new or recently updated, so indexation may still be catching up.
- The site is on a GitHub Pages subdomain, which usually starts with low authority.
- There may be few or no strong external links pointing at the site yet.
- Search Console and Bing Webmaster submissions may not have been completed.
- Branded demand for the exact query may still be low, so search engines have little confidence or query history.

## Technical SEO Already In Place

- Canonical tags are generated in `scripts/build-pages.mjs`.
- `robots.txt` allows crawling and points to `sitemap.xml`.
- `sitemap.xml` lists the homepage, `slides.html`, and all lesson pages.
- The homepage publishes `WebSite`, `Course`, and `ItemList` structured data.
- Lesson pages publish `TechArticle` structured data.

## Technical Improvements To Keep

- Keep every lesson page as a distinct HTML URL with unique title and description.
- Keep internal links from the homepage to the lessons, slide index, and code anchors.
- Keep the sitemap current on every deployment.
- Keep the markdown lesson bundle published for agent retrieval, but do not treat it as the primary search landing surface.

## Immediate Actions Outside The Codebase

1. Verify the site in Google Search Console.
2. Submit `https://cudacourseh100.github.io/sitemap.xml` in Google Search Console.
3. Request indexing for the homepage, `slides.html`, and the lesson pages.
4. Verify the site in Bing Webmaster Tools and submit the same sitemap.
5. Add strong external links from:
   - the FreeCodeCamp course page
   - Prateek's GitHub profile or repo README
   - YouTube description if a video exists
   - X / LinkedIn posts that mention the course

## Code-Level Improvements Worth Pursuing

1. Add a dedicated FAQ section on the homepage and publish matching `FAQPage` structured data.
2. Add a glossary page with terms like `mbarrier`, `CUtensorMap`, `cp.async.bulk`, `WGMMA`, `NVSwitch`, and `PMIx`.
3. Add an instructor/about page for Prateek Shukla and link it from the homepage footer.
4. Add stronger internal links between lesson pages so search engines can follow the curriculum as a graph rather than only from the homepage.
5. Publish a custom domain when available. A serious branded domain usually performs better than a generic GitHub Pages hostname over time.

## Content Strategy

The site should rank on specific, technical, mechanism-heavy terms rather than generic learning queries.

Good target clusters:

- `CUDA Programming for NVIDIA H100s`
- `Prateek Shukla H100 course`
- `Hopper asynchronous execution`
- `H100 WGMMA`
- `H100 cp.async.bulk`
- `cuTensorMap tutorial`
- `H100 mbarrier`
- `H100 Stream-K`
- `H100 NCCL PMIx Slurm`

Avoid trying to chase broad, commodity phrases first:

- `learn CUDA`
- `CUDA course`
- `GPU programming course`

The site is stronger when it owns the advanced Hopper vocabulary.

## Measurement

Track these signals weekly:

- indexed pages in Google Search Console
- sitemap discovered vs indexed counts
- top branded queries
- impressions and clicks
- which lesson pages get indexed first
- backlinks from FreeCodeCamp, GitHub, and social posts

## Notes From The 2026-03-29 Audit

- The live homepage HTML includes canonical, robots, Open Graph, Twitter, and JSON-LD tags.
- `robots.txt` allows all crawlers and points at the sitemap.
- `sitemap.xml` is live and lists the lesson URLs.
- Branded search visibility still appears weak, which suggests discovery and authority are the next bottlenecks.
