/**
 * @module types
 *
 * Core type definitions for the Semantic-Similarity Rating (SSR) library.
 *
 * This module defines all data structures used throughout the library.
 * Types are intentionally kept as plain TypeScript interfaces (no classes)
 * to make them easy to construct in tests and serializable across boundaries.
 *
 * ## Key concepts
 *
 * - **PMF (Probability Mass Function)**: A `number[]` where all values are ≥ 0
 *   and sum to 1.0. Each element represents the probability of one Likert scale point.
 *
 * - **Embedding**: A `number[]` representing a sentence's position in semantic space.
 *   Typically 384 dimensions when using the `all-MiniLM-L6-v2` model.
 *
 * - **Reference sentence**: A labeled sentence anchoring one point on a Likert scale
 *   (e.g., "Strongly agree" → int_response=5).
 *
 * - **Reference set**: A group of reference sentences (identified by a shared `id`)
 *   covering all points on the Likert scale (typically 1–5).
 */

// ---------------------------------------------------------------------------
// Vectors and matrices
// ---------------------------------------------------------------------------

/**
 * A one-dimensional numeric array.
 *
 * Used for embeddings, PMFs, and intermediate computation results.
 * All linear-algebra helpers in this library operate on plain `number[]`
 * so they can be tested without any external dependencies.
 */
export type Vector = number[];

/**
 * A two-dimensional numeric array stored as an array of row vectors.
 *
 * - `matrix[i]` is the i-th row vector.
 * - `matrix[i][j]` is the element at row i, column j.
 *
 * Shape is always `[rows, cols]` and can be obtained via {@link matrixShape}.
 */
export type Matrix = number[][];

// ---------------------------------------------------------------------------
// Reference data
// ---------------------------------------------------------------------------

/**
 * A single reference sentence that anchors one point on a Likert scale.
 *
 * @example
 * ```ts
 * const ref: ReferenceSentence = {
 *   id: "set1",
 *   intResponse: 3,
 *   sentence: "Neutral",
 * };
 * ```
 */
export interface ReferenceSentence {
  /** Identifier for the reference set this sentence belongs to (e.g. `"set1"`). */
  id: string;

  /**
   * The integer Likert scale point this sentence represents (1-indexed).
   * For a standard 5-point scale: 1, 2, 3, 4, or 5.
   */
  intResponse: number;

  /** The text of the reference sentence (e.g. `"Strongly agree"`). */
  sentence: string;

  /**
   * Optional pre-computed embedding vector for this sentence.
   * When provided, the library operates in **embedding mode** and skips
   * automatic embedding computation.
   */
  embedding?: Vector;
}

// ---------------------------------------------------------------------------
// Processed internal structures
// ---------------------------------------------------------------------------

/**
 * A fully validated and processed reference set, ready for PMF computation.
 *
 * This is the internal representation created after validating a group of
 * {@link ReferenceSentence} entries that share the same `id`.
 *
 * The `embeddingMatrix` is stored in **column-major** layout:
 * - Shape: `[embeddingDim, numLikertPoints]`
 * - Column `j` is the embedding for Likert point `j+1`.
 *
 * This layout matches the Python implementation and makes the
 * cosine-similarity dot product `responses @ embeddingMatrix` efficient.
 */
export interface ProcessedReferenceSet {
  /** The reference set identifier (e.g. `"set1"`). */
  id: string;

  /** Ordered reference sentences (sorted by `intResponse` ascending). */
  sentences: string[];

  /**
   * Embedding matrix in column-major layout.
   * Shape: `[embeddingDim, numLikertPoints]`.
   */
  embeddingMatrix: Matrix;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Configuration options for {@link ResponseRater} construction.
 */
export interface ResponseRaterOptions {
  /**
   * Name of the column containing pre-computed embeddings.
   * If a {@link ReferenceSentence} has a non-undefined `embedding` field,
   * the rater operates in embedding mode.
   *
   * @default "embedding"
   */
  embeddingsColumn?: string;

  /**
   * HuggingFace model identifier used for automatic embedding computation
   * in text mode. Uses `@huggingface/transformers` (v3+).
   *
   * @default "Xenova/all-MiniLM-L6-v2"
   */
  modelName?: string;
}

// ---------------------------------------------------------------------------
// Embedding provider interface
// ---------------------------------------------------------------------------

/**
 * A pluggable interface for computing sentence embeddings.
 *
 * The default implementation uses `@xenova/transformers`, but any object
 * satisfying this interface can be injected — making the library easy to
 * test with deterministic mock embeddings.
 *
 * @example
 * ```ts
 * // Deterministic mock for unit tests
 * const mockProvider: EmbeddingProvider = {
 *   async encode(texts: string[]): Promise<number[][]> {
 *     return texts.map((_, i) => Array(384).fill(i * 0.1));
 *   },
 * };
 * ```
 */
export interface EmbeddingProvider {
  /**
   * Compute embedding vectors for a batch of text strings.
   *
   * @param texts - Array of sentences to embed.
   * @returns A matrix where row `i` is the embedding for `texts[i]`.
   *          Shape: `[texts.length, embeddingDim]`.
   */
  encode(texts: string[]): Promise<Matrix>;
}
