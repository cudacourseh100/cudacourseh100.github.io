import { cp, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const outDir = path.join(rootDir, "docs");

const rootFiles = ["index.html", "slides.html", "styles.css", "script.js", "favicon.svg"];

const slideFiles = [
  "0. CUDA Programming for NVIDIA H100 GPUs.pdf",
  "2. Clusters, Data types, inline PTX, State Spaces.pdf",
  "3. Asynchronicity and barriers.pdf",
  "4. cuTensorMap.pdf",
  "5. cp.async.bulk.pdf",
  "6. WGMMA-1.pdf",
  "7. Wgmma part 2.pdf",
  "8. Kernel Design.pdf",
  "9. Multi GPU.pdf",
  "10. Multi GPU  Part 2.pdf",
];

const codeFiles = [
  "sm90_gemm_tma_warpspecialized_pingpong.hpp",
  "sm90_tile_scheduler_stream_k.hpp",
  "fast.cu/README.md",
  "fast.cu/LICENSE",
  "fast.cu/examples/matmul/matmul_12.cuh",
];

const localAttrPattern = /\b(?:href|src)="([^"]+)"/g;
const idPattern = /\sid="([^"]+)"/g;

async function ensureDir(targetPath) {
  await mkdir(path.dirname(targetPath), { recursive: true });
}

async function copyRelative(fromRelative, toRelative = fromRelative) {
  const sourcePath = path.join(rootDir, fromRelative);
  const targetPath = path.join(outDir, toRelative);
  await ensureDir(targetPath);
  await cp(sourcePath, targetPath, { recursive: true });
}

async function fileExists(targetPath) {
  try {
    await stat(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function validateIndex() {
  const htmlPath = path.join(outDir, "index.html");
  const html = await readFile(htmlPath, "utf8");

  const ids = new Set([...html.matchAll(idPattern)].map((match) => match[1]));
  const missing = [];

  for (const match of html.matchAll(localAttrPattern)) {
    const value = match[1];

    if (
      value.startsWith("http://") ||
      value.startsWith("https://") ||
      value.startsWith("mailto:") ||
      value.startsWith("tel:")
    ) {
      continue;
    }

    if (value.startsWith("#")) {
      if (!ids.has(value.slice(1))) {
        missing.push(`Missing in-page anchor target: ${value}`);
      }
      continue;
    }

    const cleanValue = value.split("#")[0].split("?")[0];
    const targetPath = path.join(outDir, cleanValue);

    if (!(await fileExists(targetPath))) {
      missing.push(`Missing local asset: ${cleanValue}`);
    }
  }

  if (missing.length > 0) {
    throw new Error(missing.join("\n"));
  }
}

async function build() {
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  for (const file of rootFiles) {
    await copyRelative(file);
  }

  for (const file of slideFiles) {
    await copyRelative(path.join("H100-Course", "slides", file));
  }

  for (const file of codeFiles) {
    await copyRelative(path.join("files", file));
  }

  const indexHtml = await readFile(path.join(outDir, "index.html"), "utf8");
  await writeFile(path.join(outDir, "404.html"), indexHtml);
  await writeFile(path.join(outDir, ".nojekyll"), "");

  await validateIndex();

  console.log("Built GitHub Pages bundle in docs/");
}

build().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
