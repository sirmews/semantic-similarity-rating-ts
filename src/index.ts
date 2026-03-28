/**
 * # Semantic-Similarity Rating (SSR) — TypeScript
 *
 * A TypeScript port of the Python SSR library for converting LLM textual responses
 * into Likert-scale probability distributions using semantic similarity.
 *
 * ## Quick start
 *
 * ```ts
 * import { ResponseRater } from "@sirmews/semantic-similarity-rating";
 * import type { ReferenceSentence } from "@sirmews/semantic-similarity-rating";
 *
 * const refs: ReferenceSentence[] = [
 *   { id: "set1", intResponse: 1, sentence: "Strongly disagree" },
 *   { id: "set1", intResponse: 2, sentence: "Disagree" },
 *   { id: "set1", intResponse: 3, sentence: "Neutral" },
 *   { id: "set1", intResponse: 4, sentence: "Agree" },
 *   { id: "set1", intResponse: 5, sentence: "Strongly agree" },
 * ];
 *
 * const rater = await ResponseRater.create(refs);
 * const pmfs = await rater.getResponsePmfs("set1", [
 *   "I totally agree",
 *   "Not sure about this",
 * ]);
 * ```
 *
 * ## Architecture
 *
 * The library is organised into small, independently testable modules:
 *
 * | Module             | Purpose                                       | Pure? |
 * | ------------------ | --------------------------------------------- | ----- |
 * | `types`            | All type definitions and interfaces            | N/A   |
 * | `linalg`           | Plain-array linear algebra helpers             | ✅    |
 * | `compute`          | Core SSR algorithm (PMF conversion & scaling)  | ✅    |
 * | `validation`       | Reference sentence validation                  | ✅    |
 * | `embeddings`       | Embedding provider (wraps @xenova/transformers)| Async |
 * | `response-rater`   | High-level `ResponseRater` class               | Async |
 *
 * @packageDocumentation
 */

// ---------------------------------------------------------------------------
// Types (re-exported for consumers)
// ---------------------------------------------------------------------------

export type {
  Vector,
  Matrix,
  ReferenceSentence,
  ProcessedReferenceSet,
  ResponseRaterOptions,
  EmbeddingProvider,
} from "./types";

// ---------------------------------------------------------------------------
// Core algorithm (pure functions)
// ---------------------------------------------------------------------------

export { scalePmf, responseEmbeddingsToPmf } from "./compute";

// ---------------------------------------------------------------------------
// Linear algebra helpers (pure functions)
// ---------------------------------------------------------------------------

export {
  vectorNorm,
  normalizeVector,
  dot,
  argmax,
  argmin,
  sum,
  min,
  matMul,
  columnNorms,
  rowNorms,
  divideColumns,
  divideRows,
  transpose,
  matrixShape,
} from "./linalg";

// ---------------------------------------------------------------------------
// Validation (pure functions)
// ---------------------------------------------------------------------------

export { validateReferenceSentences } from "./validation";

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

export { TransformersEmbeddingProvider } from "./embeddings";

// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

export { ResponseRater } from "./response-rater";
