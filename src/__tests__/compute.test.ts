/**
 * Tests for compute module — core SSR algorithm functions.
 *
 * Direct port of Python `tests/test_compute.py`.
 * Every test validates a mathematical property of the SSR algorithm,
 * not an implementation detail.
 */

import { scalePmf, responseEmbeddingsToPmf } from "../compute";
import { argmax } from "../linalg";
import {
  EMBEDDING_DIM,
  LIKERT_SIZE,
  createTestEmbeddings,
  createLikertEmbeddings,
  assertValidPmfVector,
  assertValidPmfMatrix,
  allClose,
  entropy,
} from "./helpers";

// ===================================================================
// scalePmf
// ===================================================================

describe("scalePmf", () => {
  // --- Port of TestScalePMF ---

  it("temperature=1 leaves PMF unchanged (identity)", () => {
    const pmf = [0.1, 0.2, 0.3, 0.4];
    const scaled = scalePmf(pmf, 1.0);
    expect(allClose(scaled, pmf)).toBe(true);
  });

  it("temperature=0 with non-uniform PMF returns one-hot at mode", () => {
    const pmf = [0.1, 0.2, 0.3, 0.4];
    const scaled = scalePmf(pmf, 0.0);
    expect(scaled[3]).toBe(1.0);
    expect(scaled[0]).toBe(0);
    expect(scaled[1]).toBe(0);
    expect(scaled[2]).toBe(0);
  });

  it("temperature=0 with uniform PMF returns same PMF", () => {
    const pmf = [0.25, 0.25, 0.25, 0.25];
    const scaled = scalePmf(pmf, 0.0);
    expect(allClose(scaled, pmf)).toBe(true);
  });

  it("near-zero temperature concentrates probability at mode", () => {
    const pmf = [0.1, 0.2, 0.3, 0.4];
    const sharp = scalePmf(pmf, 0.01);
    assertValidPmfVector(sharp);
    expect(sharp[argmax(pmf)]).toBeGreaterThan(0.99);
  });

  it("high temperature smooths toward uniform", () => {
    const pmf = [0.1, 0.2, 0.3, 0.4];
    const smooth = scalePmf(pmf, 10.0);
    assertValidPmfVector(smooth);

    // All values should be closer to 0.25 than original
    const maxDev = Math.max(...smooth.map((p) => Math.abs(p - 0.25)));
    expect(maxDev).toBeLessThan(0.1);
  });

  it("higher temperature increases entropy", () => {
    const pmf = [0.1, 0.2, 0.3, 0.4];

    const sharp = scalePmf(pmf, 0.01);
    const smooth = scalePmf(pmf, 10.0);

    assertValidPmfVector(sharp);
    assertValidPmfVector(smooth);

    expect(entropy(smooth)).toBeGreaterThan(entropy(sharp));
  });

  it("temperature is capped at maxTemp", () => {
    const pmf = [0.1, 0.6, 0.3];

    const capped = scalePmf(pmf, 100.0, 5.0);
    const expected = scalePmf(pmf, 5.0);

    expect(allClose(capped, expected)).toBe(true);
  });

  it("always produces a valid PMF for various temperatures", () => {
    const pmf = [0.05, 0.15, 0.3, 0.35, 0.15];
    for (const t of [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 100.0]) {
      const scaled = scalePmf(pmf, t);
      assertValidPmfVector(scaled, `temperature=${t}`);
    }
  });
});

// ===================================================================
// responseEmbeddingsToPmf
// ===================================================================

describe("responseEmbeddingsToPmf", () => {
  // --- Port of TestEmbeddingsToPMF ---

  it("converts embeddings to valid PMFs with correct shape", () => {
    const responseEmbs = createTestEmbeddings(3);
    const likertEmbs = createLikertEmbeddings();

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    expect(result.length).toBe(3);
    expect(result[0].length).toBe(LIKERT_SIZE);
    assertValidPmfMatrix(result);
  });

  it("identical inputs produce identical outputs (deterministic)", () => {
    const single = createTestEmbeddings(1);
    const responseEmbs = [single[0], [...single[0]]]; // Two identical rows
    const likertEmbs = createLikertEmbeddings();

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    assertValidPmfMatrix(result);
    expect(allClose(result[0], result[1])).toBe(true);
  });

  it("epsilon regularization changes the distribution", () => {
    const responseEmbs = createTestEmbeddings(2);
    const likertEmbs = createLikertEmbeddings();

    const noEps = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 0.0);
    const withEps = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 0.1);

    assertValidPmfMatrix(noEps);
    assertValidPmfMatrix(withEps);
    expect(allClose(noEps[0], withEps[0])).toBe(false);
  });

  it("epsilon makes all probabilities positive", () => {
    const responseEmbs = createTestEmbeddings(2);
    const likertEmbs = createLikertEmbeddings();

    const withEps = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 0.1);

    for (const row of withEps) {
      for (const val of row) {
        expect(val).toBeGreaterThan(0);
      }
    }
  });

  it("higher epsilon increases entropy (more uniform)", () => {
    const responseEmbs = createTestEmbeddings(1);
    const likertEmbs = createLikertEmbeddings();

    const lowEps = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 0.001)[0];
    const highEps = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 0.1)[0];

    expect(entropy(highEps)).toBeGreaterThanOrEqual(entropy(lowEps));
  });

  it("handles empty response array gracefully", () => {
    const emptyResponses: number[][] = [];
    const likertEmbs = createLikertEmbeddings();

    const result = responseEmbeddingsToPmf(emptyResponses, likertEmbs);

    expect(result).toEqual([]);
  });

  it("PMF reflects embedding similarity ranking", () => {
    const likertEmbs = createLikertEmbeddings();

    // Create a response that IS the first Likert column (transposed)
    const col0: number[] = likertEmbs.map((row) => row[0]);
    const responseEmbs = [col0];

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 0.01);

    // First Likert point should have highest probability
    expect(argmax(result[0])).toBe(0);
  });
});

// ===================================================================
// Edge cases (port of TestEdgeCases)
// ===================================================================

describe("responseEmbeddingsToPmf edge cases", () => {
  it("works with a single response and realistic embeddings", () => {
    const responseEmbs = [[1.0, 0.5, -0.2]];
    const likertEmbs = [
      [1.0, 0.5, 0.0, -0.5, -1.0],
      [0.8, 0.2, 0.1, -0.3, -0.8],
      [0.6, 0.1, 0.0, -0.1, -0.6],
    ];

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    expect(result.length).toBe(1);
    expect(result[0].length).toBe(5);
    assertValidPmfMatrix(result, 1, 5);
  });

  it("handles small but non-zero embeddings", () => {
    const small = 1e-6;
    const responseEmbs = [
      Array(EMBEDDING_DIM).fill(small),
      Array(EMBEDDING_DIM).fill(small),
    ];
    const likertEmbs = createLikertEmbeddings();

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    expect(result.length).toBe(2);
    assertValidPmfMatrix(result);
    // Both responses are identical, so PMFs should match
    expect(allClose(result[0], result[1])).toBe(true);
  });

  it("handles extreme similarity values (scaled-up embeddings)", () => {
    const likertEmbs = createLikertEmbeddings();
    // Response is a scaled-up version of the 3rd column
    const col2: number[] = likertEmbs.map((row) => row[2] * 1000);
    const responseEmbs = [col2];

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    assertValidPmfMatrix(result, 1);
    // Should strongly prefer the 3rd Likert point
    expect(result[0][2]).toBeGreaterThan(0.8);
  });
});

// ===================================================================
// Numerical stability (port of TestNumericalStability)
// ===================================================================

describe("responseEmbeddingsToPmf numerical stability", () => {
  it("handles large embedding values without NaN or Infinity", () => {
    const responseEmbs = createTestEmbeddings(3).map((row) =>
      row.map((v) => v * 1000)
    );
    const likertEmbs = createLikertEmbeddings().map((row) =>
      row.map((v) => v * 1000)
    );

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    assertValidPmfMatrix(result);
    for (const row of result) {
      for (const val of row) {
        expect(Number.isFinite(val)).toBe(true);
      }
    }
  });

  it("handles very small epsilon values", () => {
    const responseEmbs = createTestEmbeddings(1);
    const likertEmbs = createLikertEmbeddings();

    const result = responseEmbeddingsToPmf(responseEmbs, likertEmbs, 1e-10);

    assertValidPmfMatrix(result);
    for (const val of result[0]) {
      expect(Number.isFinite(val)).toBe(true);
    }
  });

  it("handles degenerate input where all cosine similarities are equal", () => {
    // When a response has identical cosine similarity to all references
    // and epsilon=0, the denominator is 0. Our guard returns uniform PMF.
    // Construct: use a response that is equidistant from all Likert points.
    const likertEmbs = [
      [1.0, 0.0, -1.0],
      [0.0, 1.0,  0.0],
      [0.0, 0.0,  1.0],
    ];
    // A response along [1, 1, 1] has equal projection onto normalised columns
    // when the columns happen to be symmetric. Use a case where min == all.
    // Simplest: all Likert columns identical → all cosine sims equal.
    const identicalLikert = [
      [1.0, 1.0, 1.0],
      [0.5, 0.5, 0.5],
      [0.2, 0.2, 0.2],
    ];
    const responseEmbs = [[1.0, 0.5, 0.2]];

    const result = responseEmbeddingsToPmf(responseEmbs, identicalLikert, 0.0);

    expect(result.length).toBe(1);
    // Should be uniform since all similarities are equal
    for (const val of result[0]) {
      expect(Number.isFinite(val)).toBe(true);
      expect(val).toBeCloseTo(1 / 3, 5);
    }
  });
});

// ===================================================================
// Combined: scalePmf + responseEmbeddingsToPmf pipeline
// ===================================================================

describe("scalePmf + responseEmbeddingsToPmf pipeline", () => {
  it("applying temperature scaling after PMF conversion preserves validity", () => {
    const responseEmbs = createTestEmbeddings(3);
    const likertEmbs = createLikertEmbeddings();

    const pmfs = responseEmbeddingsToPmf(responseEmbs, likertEmbs);

    for (const t of [0.1, 0.5, 1.0, 2.0, 5.0]) {
      const scaled = pmfs.map((row) => scalePmf(row, t));
      assertValidPmfMatrix(scaled);
    }
  });
});
