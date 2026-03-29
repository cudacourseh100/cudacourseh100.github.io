# AI-Fetchable Lesson Markdown

These files combine the published lesson pages with full extracted slide text so an AI agent can pull searchable lesson content with `curl` or `wget`.

## Fetch Examples

```sh
curl -L https://cudacourseh100.github.io/markdown/lessons/lesson-01-introduction-to-h100s.md
wget -O lesson-06-wgmma-part-1.md https://cudacourseh100.github.io/markdown/lessons/lesson-06-wgmma-part-1.md
```

## Files

- `lessons/lesson-01-introduction-to-h100s.md` - Lesson 1: Introduction to H100s
- `lessons/lesson-02-clusters-data-types-inline-ptx-pointers.md` - Lesson 2: Clusters, Data Types, Inline PTX, Pointers
- `lessons/lesson-03-asynchronicity-and-barriers.md` - Lesson 3: Asynchronicity and Barriers
- `lessons/lesson-04-cutensormap.md` - Lesson 4: cuTensorMap
- `lessons/lesson-05-cp-async-bulk.md` - Lesson 5: `cp.async.bulk`
- `lessons/lesson-06-wgmma-part-1.md` - Lesson 6: WGMMA Part 1
- `lessons/lesson-07-wgmma-part-2.md` - Lesson 7: WGMMA Part 2
- `lessons/lesson-08-kernel-design.md` - Lesson 8: Kernel Design
- `lessons/lesson-08-1-stream-k.md` - Lesson 8.1: Stream-K
- `lessons/lesson-09-multi-gpu-part-1.md` - Lesson 9: Multi GPU Part 1
- `lessons/lesson-10-multi-gpu-part-2.md` - Lesson 10: Multi GPU Part 2

## Strategy Docs

- `seo/seo-strategy.md` - SEO strategy
- `seo/indexing-checklist.md` - SEO indexing checklist

## Regeneration

```sh
python3 scripts/export-lesson-markdown.py
```
