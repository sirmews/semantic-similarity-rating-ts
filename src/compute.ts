/**
 * @module compute
 *
 * Core SSR (Semantic-Similarity Rating) algorithm functions.
 *
 * This module is a direct, line-by-line port of the Python `compute.py`.
 * Every function is **pure** — no side effects, no hidden state, no async.
 * Inputs and outputs are plain `number[]` / `number[][]` arrays, so every
 * function can be tested in complete isolation.
 *
 * ## Algorithm overview
 *
 * The SSR methodology converts LLM textual responses into probability
 * distributions over a Likert scale by:
 *
 * 1. Computing cosine similarities between response embeddings and
 *    reference-sentence embeddings.
 * 2. Subtracting the minimum similarity per response (so the least-similar
 *    reference gets probability ≈ 0).
 * 3. Adding optional epsilon regularisation to prevent exact zeros.
 * 4. Normalizing to a valid PMF (sums to 1).
 * 5. Optionally applying temperature scaling to sharpen or smooth the PMF.
 *
 * ## Correspondence to Python implementation
 *
 * | Python `compute.py`             | This module                           |
 * | ------------------------------- | ------------------------------------- |
 * | `scale_pmf(pmf, T, max_temp)`   | {@link scalePmf}                      |
 * | `response_embeddings_to_pmf(…)` | {@link responseEmbeddingsToPmf}       |
 */

import type { Vector, Matrix } from "./types";
import {
  argmax,
  argmin,
  columnNorms,
  divideColumns,
  divideRows,
  matMul,
  min,
  rowNorms,
  sum,
} from "./linalg";

// ---------------------------------------------------------------------------
// scalePmf
// ---------------------------------------------------------------------------

/**
 * Apply temperature scaling to a probability mass function.
 *
 * Temperature controls the "sharpness" of the distribution:
 * - **T → 0**: collapses to a one-hot vector at the mode (most confident).
 * - **T = 1**: returns the original PMF unchanged.
 * - **T > 1**: smooths toward uniform (less confident).
 *
 * The transformation is: `p_new[i] ∝ p_old[i]^(1/T)`, re-normalised to sum to 1.
 *
 * ### Correspondence to Python
 * Direct port of `compute.scale_pmf(pmf, temperature, max_temp)`.
 *
 * @param pmf         - Input probability mass function. All values must be ≥ 0
 *                      and sum to 1.
 * @param temperature - Scaling temperature. Must be ≥ 0.
 *                      - `0` → one-hot at argmax (or unchanged if uniform).
 *                      - `1` → identity (no change).
 *                      - `> 1` → smoother distribution.
 * @param maxTemp     - Upper bound on temperature. If `temperature > maxTemp`,
 *                      `maxTemp` is used instead. Defaults to `Infinity` (no cap).
 * @returns A new PMF array of the same length, scaled and re-normalised.
 *
 * @example
 * ```ts
 * // Identity: temperature=1 returns the same PMF
 * scalePmf([0.1, 0.2, 0.3, 0.4], 1.0);
 * // → [0.1, 0.2, 0.3, 0.4]
 *
 * // Sharpening: low temperature concentrates probability at the mode
 * scalePmf([0.1, 0.2, 0.3, 0.4], 0.01);
 * // → [~0, ~0, ~0, ~1]
 *
 * // Smoothing: high temperature moves toward uniform
 * scalePmf([0.1, 0.2, 0.3, 0.4], 10.0);
 * // → [~0.22, ~0.24, ~0.26, ~0.28]
 * ```
 *
 * @pure This function has no side effects.
 */
export function scalePmf(
  pmf: Vector,
  temperature: number,
  maxTemp: number = Infinity
): Vector {
  // --- T = 0: collapse to one-hot at the mode ---
  if (temperature === 0.0) {
    const allEqual = pmf.every((v) => v === pmf[0]);
    if (allEqual) {
      return [...pmf];
    }
    const result = new Array(pmf.length).fill(0);
    result[argmax(pmf)] = 1.0;
    return result;
  }

  // --- Apply temperature (capped at maxTemp) ---
  const effectiveTemp = temperature > maxTemp ? maxTemp : temperature;
  const exponent = 1 / effectiveTemp;
  const hist = pmf.map((p) => Math.pow(p, exponent));

  // --- Re-normalise ---
  const total = sum(hist);
  if (total === 0) {
    // All entries underflowed to 0 (extreme low temperature with small probs).
    // Fall back to one-hot at the original mode, matching T→0 behaviour.
    const result = new Array(pmf.length).fill(0);
    result[argmax(pmf)] = 1.0;
    return result;
  }
  return hist.map((h) => h / total);
}

// ---------------------------------------------------------------------------
// responseEmbeddingsToPmf
// ---------------------------------------------------------------------------

/**
 * Convert response embeddings and Likert-sentence embeddings into probability
 * mass functions using the SSR equation.
 *
 * This is the mathematical core of the SSR methodology. For each response
 * embedding it:
 *
 * 1. L2-normalises both the response and reference matrices.
 * 2. Computes scaled cosine similarities: `cos = (1 + dot(response, reference)) / 2`.
 * 3. Subtracts the per-row minimum similarity.
 * 4. Adds epsilon regularisation at the minimum-similarity position (Kronecker delta).
 * 5. Normalises each row to sum to 1, producing a valid PMF.
 *
 * ### Correspondence to Python
 * Direct port of `compute.response_embeddings_to_pmf(matrix_responses, matrix_likert_sentences, epsilon)`.
 *
 * ### Matrix layout
 *
 * ```
 * matrixResponses:       [numResponses × embeddingDim]    (row = one response)
 * matrixLikertSentences: [embeddingDim  × numLikertPoints] (col = one Likert point)
 * result:                [numResponses  × numLikertPoints] (row = PMF for one response)
 * ```
 *
 * @param matrixResponses       - Response embedding matrix.
 *   Shape: `[numResponses, embeddingDim]`. Each row is one LLM response embedding.
 * @param matrixLikertSentences - Reference-sentence embedding matrix in **column-major** layout.
 *   Shape: `[embeddingDim, numLikertPoints]`. Column `j` is the embedding for Likert point `j+1`.
 * @param epsilon               - Regularisation parameter (≥ 0). Adds a small constant to the
 *   minimum-similarity position in each row to prevent exact-zero probabilities.
 *   `0` means no regularisation. Default: `0`.
 * @returns A matrix of PMFs. Shape: `[numResponses, numLikertPoints]`.
 *   Each row sums to 1 and contains non-negative values.
 *   Returns an empty matrix `[]` if `matrixResponses` is empty.
 *
 * @example
 * ```ts
 * // 2 responses, 3-dimensional embeddings, 3 Likert points
 * const responses = [
 *   [1.0, 0.5, -0.2],
 *   [-0.3, 0.8, 0.1],
 * ];
 * const likert = [
 *   [1.0, 0.0, -1.0],   // dim 0
 *   [0.0, 1.0,  0.0],   // dim 1
 *   [0.0, 0.0,  1.0],   // dim 2
 * ];
 * const pmfs = responseEmbeddingsToPmf(responses, likert);
 * // pmfs[0] sums to 1, pmfs[1] sums to 1
 * ```
 *
 * @pure This function has no side effects.
 */
export function responseEmbeddingsToPmf(
  matrixResponses: Matrix,
  matrixLikertSentences: Matrix,
  epsilon: number = 0.0
): Matrix {
  const mLeft = matrixResponses;
  const mRight = matrixLikertSentences;

  // --- Handle empty input ---
  if (mLeft.length === 0) {
    return [];
  }

  // --- Step 1: L2-normalise the reference matrix (column-wise) ---
  // Python: norm_right = np.linalg.norm(M_right, axis=0)
  //         M_right = M_right / norm_right[None, :]
  const normRight = columnNorms(mRight);
  const mRightNormed = divideColumns(mRight, normRight);

  // --- Step 2: L2-normalise the response matrix (row-wise) ---
  // Python: norm_left = np.linalg.norm(M_left, axis=1)
  //         M_left = M_left / norm_left[:, None]
  const normLeft = rowNorms(mLeft);
  const mLeftNormed = divideRows(mLeft, normLeft);

  // --- Step 3: Compute scaled cosine similarities ---
  // Python: cos = (1 + M_left.dot(M_right)) / 2
  // cos[i][j] = similarity between response i and Likert point j
  const dotProduct = matMul(mLeftNormed, mRightNormed);
  const cos: Matrix = dotProduct.map((row) =>
    row.map((val) => (1 + val) / 2)
  );

  // --- Step 4: Subtract per-row minimum similarity ---
  // Python: cos_min = cos.min(axis=1)[:, None]
  //         numerator = cos - cos_min
  const cosMin = cos.map((row) => min(row));
  const numerator: Matrix = cos.map((row, i) =>
    row.map((val) => val - cosMin[i])
  );

  // --- Step 5: Apply epsilon regularisation (Kronecker delta) ---
  // Python: if epsilon > 0:
  //           min_indices = np.argmin(cos, axis=1)
  //           for i, min_idx in enumerate(min_indices):
  //               numerator[i, min_idx] += epsilon
  if (epsilon > 0) {
    for (let i = 0; i < cos.length; i++) {
      const minIdx = argmin(cos[i]);
      numerator[i][minIdx] += epsilon;
    }
  }

  // --- Step 6: Compute denominator and normalise ---
  // Python: denominator = cos.sum(axis=1)[:, None] - n_likert_points * cos_min + epsilon
  //         pmf = numerator / denominator
  //
  // Note on numerical safety: When all cosine similarities in a row are
  // identical AND epsilon=0, the denominator is 0. This mirrors NumPy's
  // behaviour (which returns NaN). In practice this only happens with
  // degenerate inputs (e.g., a zero-vector response). We return a uniform
  // distribution in that case, which is the most principled fallback.
  const nLikertPoints = cos[0].length;
  const pmf: Matrix = numerator.map((row, i) => {
    const denominator =
      sum(cos[i]) - nLikertPoints * cosMin[i] + epsilon;
    if (denominator === 0) {
      // Degenerate case: all similarities equal, return uniform PMF
      return row.map(() => 1 / nLikertPoints);
    }
    return row.map((val) => val / denominator);
  });

  return pmf;
}
