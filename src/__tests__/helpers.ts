/**
 * Shared test utilities for the SSR TypeScript test suite.
 *
 * Provides:
 * - Deterministic pseudo-random number generation (seeded, no external deps)
 * - PMF validation helpers
 * - Test data factories
 *
 * These mirror the helpers in the Python `test_compute.py` and
 * `test_response_rater.py` files so that test logic is directly comparable.
 */

import type { Vector, Matrix, ReferenceSentence, EmbeddingProvider } from "../types";

// ---------------------------------------------------------------------------
// Constants (matching Python test constants)
// ---------------------------------------------------------------------------

export const EMBEDDING_DIM = 384;
export const EMBEDDING_DIM_SMALL = 128;
export const LIKERT_SIZE = 5;
export const SAMPLE_TEXTS = ["terrible", "poor", "neutral", "good", "excellent"];
export const TEST_RESPONSES = ["I love this product", "It's okay I guess", "Completely awful"];

// ---------------------------------------------------------------------------
// Deterministic PRNG (replaces np.random.seed)
// ---------------------------------------------------------------------------

/**
 * Simple seeded PRNG (Mulberry32) — produces deterministic floats in [0, 1).
 * This replaces `np.random.seed()` so tests are reproducible without NumPy.
 */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a normally-distributed random number using the Box-Muller transform.
 * Uses the provided uniform PRNG.
 */
function normalRandom(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1 || 1e-10)) * Math.cos(2.0 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Test data factories
// ---------------------------------------------------------------------------

/**
 * Create a deterministic matrix of response embeddings.
 * Shape: `[nResponses, nDimensions]` — each row is one response embedding.
 *
 * @param nResponses - Number of response vectors to generate.
 * @param nDimensions - Embedding dimension.
 * @param seed - PRNG seed for reproducibility.
 */
export function createTestEmbeddings(
  nResponses: number = 3,
  nDimensions: number = EMBEDDING_DIM,
  seed: number = 42
): Matrix {
  const rng = mulberry32(seed);
  const result: Matrix = [];
  for (let i = 0; i < nResponses; i++) {
    const row: Vector = [];
    for (let j = 0; j < nDimensions; j++) {
      row.push(normalRandom(rng));
    }
    result.push(row);
  }
  return result;
}

/**
 * Create a deterministic matrix of Likert reference embeddings in
 * **column-major** layout (matching `responseEmbeddingsToPmf` input).
 * Shape: `[nDimensions, nPoints]`.
 *
 * Uses a different seed offset than `createTestEmbeddings` for diversity.
 *
 * @param nDimensions - Embedding dimension.
 * @param nPoints - Number of Likert scale points.
 * @param seed - PRNG seed (offset by +1 from default for diversity).
 */
export function createLikertEmbeddings(
  nDimensions: number = EMBEDDING_DIM,
  nPoints: number = LIKERT_SIZE,
  seed: number = 43 // seed + 1 from Python
): Matrix {
  const rng = mulberry32(seed);
  const result: Matrix = [];
  for (let i = 0; i < nDimensions; i++) {
    const row: Vector = [];
    for (let j = 0; j < nPoints; j++) {
      row.push(normalRandom(rng));
    }
    result.push(row);
  }
  return result;
}

/**
 * Create a set of reference sentences for testing.
 *
 * @param includeEmbeddings - If true, attach random embeddings (embedding mode).
 * @param numSets - Number of reference sets to create.
 * @param embeddingDim - Dimension of embeddings (only used if includeEmbeddings).
 * @param seed - PRNG seed.
 */
export function createTestReferences(
  includeEmbeddings: boolean = false,
  numSets: number = 1,
  embeddingDim: number = EMBEDDING_DIM_SMALL,
  seed: number = 42
): ReferenceSentence[] {
  const rng = mulberry32(seed);
  const refs: ReferenceSentence[] = [];

  for (let setIdx = 1; setIdx <= numSets; setIdx++) {
    for (let i = 0; i < LIKERT_SIZE; i++) {
      const ref: ReferenceSentence = {
        id: `set${setIdx}`,
        intResponse: i + 1,
        sentence: SAMPLE_TEXTS[i],
      };
      if (includeEmbeddings) {
        const emb: Vector = [];
        for (let d = 0; d < embeddingDim; d++) {
          emb.push(normalRandom(rng));
        }
        ref.embedding = emb;
      }
      refs.push(ref);
    }
  }

  return refs;
}

// ---------------------------------------------------------------------------
// PMF assertion helpers
// ---------------------------------------------------------------------------

/**
 * Assert that a single vector is a valid PMF:
 * - All values ≥ 0
 * - Values sum to 1.0 (within tolerance)
 *
 * @param pmf - The PMF vector to validate.
 * @param label - Label for error messages.
 * @param atol - Absolute tolerance for sum-to-1 check.
 */
export function assertValidPmfVector(
  pmf: Vector,
  label: string = "PMF",
  atol: number = 1e-10
): void {
  const total = pmf.reduce((a, b) => a + b, 0);
  expect(Math.abs(total - 1.0)).toBeLessThan(atol);
  for (let j = 0; j < pmf.length; j++) {
    expect(pmf[j]).toBeGreaterThanOrEqual(0);
  }
}

/**
 * Assert that a matrix of PMFs is valid:
 * - Each row is a valid PMF.
 * - Optional shape checks.
 *
 * @param pmfs - Matrix of PMFs (one per row).
 * @param expectedRows - If provided, assert the number of rows.
 * @param expectedCols - If provided, assert the number of columns.
 */
export function assertValidPmfMatrix(
  pmfs: Matrix,
  expectedRows?: number,
  expectedCols: number = LIKERT_SIZE
): void {
  if (expectedRows !== undefined) {
    expect(pmfs.length).toBe(expectedRows);
  }
  for (let i = 0; i < pmfs.length; i++) {
    expect(pmfs[i].length).toBe(expectedCols);
    assertValidPmfVector(pmfs[i], `Row ${i}`);
  }
}

// ---------------------------------------------------------------------------
// Mock embedding provider
// ---------------------------------------------------------------------------

/**
 * A deterministic mock embedding provider for testing ResponseRater
 * without loading a real ML model.
 *
 * Returns reproducible embeddings based on a seeded PRNG, so tests are
 * deterministic and fast.
 */
export function createMockEmbeddingProvider(
  embeddingDim: number = EMBEDDING_DIM_SMALL,
  seed: number = 99
): EmbeddingProvider {
  return {
    async encode(texts: string[]): Promise<Matrix> {
      // Use a fresh RNG seeded with a hash of the input texts
      // so the same texts always produce the same embeddings
      let hashSeed = seed;
      for (const text of texts) {
        for (let i = 0; i < text.length; i++) {
          hashSeed = ((hashSeed << 5) - hashSeed + text.charCodeAt(i)) | 0;
        }
      }
      const rng = mulberry32(hashSeed);
      return texts.map(() => {
        const emb: Vector = [];
        for (let d = 0; d < embeddingDim; d++) {
          emb.push(normalRandom(rng));
        }
        return emb;
      });
    },
  };
}

// ---------------------------------------------------------------------------
// Numeric comparison helpers
// ---------------------------------------------------------------------------

/**
 * Check if two vectors are element-wise close (like `np.allclose`).
 */
export function allClose(
  a: Vector,
  b: Vector,
  atol: number = 1e-8
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > atol) return false;
  }
  return true;
}

/**
 * Compute the Shannon entropy of a PMF: -Σ p*ln(p).
 */
export function entropy(pmf: Vector): number {
  let h = 0;
  for (const p of pmf) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}
