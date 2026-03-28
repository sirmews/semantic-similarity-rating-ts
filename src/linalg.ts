/**
 * @module linalg
 *
 * Pure linear-algebra helper functions operating on plain `number[]` arrays.
 *
 * ## Design rationale
 *
 * Every function in this module is **pure** (no side effects, no hidden state)
 * and operates on plain JavaScript arrays — no external numeric library required.
 * This makes each function independently testable with simple assertions and
 * removes any coupling to third-party matrix libraries.
 *
 * ## Naming conventions
 *
 * Functions are named after the NumPy operations they replace, so readers
 * familiar with the Python SSR implementation can map between the two:
 *
 * | NumPy                        | This module                  |
 * | ---------------------------- | ---------------------------- |
 * | `np.linalg.norm(v)`          | {@link vectorNorm}           |
 * | `v / np.linalg.norm(v)`      | {@link normalizeVector}      |
 * | `np.linalg.norm(M, axis=0)`  | {@link columnNorms}          |
 * | `np.linalg.norm(M, axis=1)`  | {@link rowNorms}             |
 * | `A.dot(B)`                   | {@link matMul}               |
 * | `np.argmax(v)`               | {@link argmax}               |
 * | `np.argmin(v)`               | {@link argmin}               |
 */

import type { Vector, Matrix } from "./types";

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

/**
 * Compute the L2 (Euclidean) norm of a vector.
 *
 * Equivalent to `np.linalg.norm(v)`.
 *
 * @param v - Input vector.
 * @returns The L2 norm: `√(v[0]² + v[1]² + … + v[n-1]²)`.
 *
 * @example
 * ```ts
 * vectorNorm([3, 4]); // → 5
 * vectorNorm([1, 0, 0]); // → 1
 * ```
 */
export function vectorNorm(v: Vector): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

/**
 * Return a unit-length copy of a vector (L2-normalized).
 *
 * Equivalent to `v / np.linalg.norm(v)`.
 *
 * @param v - Input vector (must be non-zero).
 * @returns A new vector with the same direction and unit length.
 *
 * @example
 * ```ts
 * normalizeVector([3, 4]); // → [0.6, 0.8]
 * ```
 */
export function normalizeVector(v: Vector): Vector {
  const norm = vectorNorm(v);
  return v.map((x) => x / norm);
}

/**
 * Compute the dot product of two vectors of equal length.
 *
 * Equivalent to `np.dot(a, b)` for 1-D arrays.
 *
 * @param a - First vector.
 * @param b - Second vector (must have same length as `a`).
 * @returns The scalar dot product: `a[0]*b[0] + a[1]*b[1] + …`.
 * @throws {Error} If vectors have different lengths.
 *
 * @example
 * ```ts
 * dot([1, 2, 3], [4, 5, 6]); // → 32
 * ```
 */
export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new Error(
      `dot: vector lengths must match, got ${a.length} and ${b.length}`
    );
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Return the index of the largest element in a vector.
 *
 * Equivalent to `np.argmax(v)`. Ties are broken by first occurrence.
 *
 * @param v - Input vector (must be non-empty).
 * @returns Index of the maximum value.
 *
 * @example
 * ```ts
 * argmax([0.1, 0.4, 0.3]); // → 1
 * ```
 */
export function argmax(v: Vector): number {
  let maxIdx = 0;
  let maxVal = v[0];
  for (let i = 1; i < v.length; i++) {
    if (v[i] > maxVal) {
      maxVal = v[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Return the index of the smallest element in a vector.
 *
 * Equivalent to `np.argmin(v)`. Ties are broken by first occurrence.
 *
 * @param v - Input vector (must be non-empty).
 * @returns Index of the minimum value.
 *
 * @example
 * ```ts
 * argmin([0.3, 0.1, 0.4]); // → 1
 * ```
 */
export function argmin(v: Vector): number {
  let minIdx = 0;
  let minVal = v[0];
  for (let i = 1; i < v.length; i++) {
    if (v[i] < minVal) {
      minVal = v[i];
      minIdx = i;
    }
  }
  return minIdx;
}

/**
 * Sum all elements of a vector.
 *
 * Equivalent to `v.sum()` or `np.sum(v)`.
 *
 * @param v - Input vector.
 * @returns The scalar sum of all elements.
 *
 * @example
 * ```ts
 * sum([0.2, 0.3, 0.5]); // → 1.0
 * ```
 */
export function sum(v: Vector): number {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += v[i];
  }
  return total;
}

/**
 * Return the minimum value in a vector.
 *
 * @param v - Input vector (must be non-empty).
 * @returns The smallest element.
 */
export function min(v: Vector): number {
  let minVal = v[0];
  for (let i = 1; i < v.length; i++) {
    if (v[i] < minVal) {
      minVal = v[i];
    }
  }
  return minVal;
}

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

/**
 * Multiply two matrices: `result = A × B`.
 *
 * Equivalent to `A.dot(B)` or `np.matmul(A, B)`.
 *
 * @param A - Left matrix of shape `[m, k]`.
 * @param B - Right matrix of shape `[k, n]`.
 * @returns Result matrix of shape `[m, n]`.
 * @throws {Error} If inner dimensions don't match.
 *
 * @example
 * ```ts
 * const A = [[1, 2], [3, 4]];
 * const B = [[5, 6], [7, 8]];
 * matMul(A, B); // → [[19, 22], [43, 50]]
 * ```
 */
export function matMul(A: Matrix, B: Matrix): Matrix {
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;

  if (B.length !== k) {
    throw new Error(
      `matMul: inner dimensions must match, got A=[${m},${k}] and B=[${B.length},${n}]`
    );
  }

  const result: Matrix = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let val = 0;
      for (let p = 0; p < k; p++) {
        val += A[i][p] * B[p][j];
      }
      result[i][j] = val;
    }
  }
  return result;
}

/**
 * Compute L2 norms of each column of a matrix.
 *
 * Equivalent to `np.linalg.norm(M, axis=0)`.
 *
 * @param M - Input matrix of shape `[rows, cols]`.
 * @returns A vector of length `cols` where each element is the L2 norm
 *          of the corresponding column.
 *
 * @example
 * ```ts
 * columnNorms([[3, 1], [4, 0]]); // → [5, 1]
 * ```
 */
export function columnNorms(M: Matrix): Vector {
  const rows = M.length;
  const cols = M[0].length;
  const norms = new Array(cols).fill(0);
  for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let i = 0; i < rows; i++) {
      s += M[i][j] * M[i][j];
    }
    norms[j] = Math.sqrt(s);
  }
  return norms;
}

/**
 * Compute L2 norms of each row of a matrix.
 *
 * Equivalent to `np.linalg.norm(M, axis=1)`.
 *
 * @param M - Input matrix of shape `[rows, cols]`.
 * @returns A vector of length `rows` where each element is the L2 norm
 *          of the corresponding row.
 *
 * @example
 * ```ts
 * rowNorms([[3, 4], [1, 0]]); // → [5, 1]
 * ```
 */
export function rowNorms(M: Matrix): Vector {
  return M.map((row) => vectorNorm(row));
}

/**
 * Divide each column of a matrix by a corresponding scalar.
 *
 * Equivalent to `M / norms[None, :]` (broadcasting a row vector over columns).
 *
 * @param M - Input matrix of shape `[rows, cols]`.
 * @param divisors - A vector of length `cols`.
 * @returns A new matrix where `result[i][j] = M[i][j] / divisors[j]`.
 *
 * @example
 * ```ts
 * divideColumns([[6, 9], [4, 3]], [2, 3]); // → [[3, 3], [2, 1]]
 * ```
 */
export function divideColumns(M: Matrix, divisors: Vector): Matrix {
  return M.map((row) => row.map((val, j) => val / divisors[j]));
}

/**
 * Divide each row of a matrix by a corresponding scalar.
 *
 * Equivalent to `M / norms[:, None]` (broadcasting a column vector over rows).
 *
 * @param M - Input matrix of shape `[rows, cols]`.
 * @param divisors - A vector of length `rows`.
 * @returns A new matrix where `result[i][j] = M[i][j] / divisors[i]`.
 *
 * @example
 * ```ts
 * divideRows([[6, 4], [9, 3]], [2, 3]); // → [[3, 2], [3, 1]]
 * ```
 */
export function divideRows(M: Matrix, divisors: Vector): Matrix {
  return M.map((row, i) => row.map((val) => val / divisors[i]));
}

/**
 * Transpose a matrix: swap rows and columns.
 *
 * Equivalent to `M.T` in NumPy.
 *
 * @param M - Input matrix of shape `[rows, cols]`.
 * @returns Transposed matrix of shape `[cols, rows]`.
 *
 * @example
 * ```ts
 * transpose([[1, 2, 3], [4, 5, 6]]);
 * // → [[1, 4], [2, 5], [3, 6]]
 * ```
 */
export function transpose(M: Matrix): Matrix {
  const rows = M.length;
  const cols = M[0].length;
  const result: Matrix = Array.from({ length: cols }, () =>
    new Array(rows).fill(0)
  );
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = M[i][j];
    }
  }
  return result;
}

/**
 * Get the shape `[rows, cols]` of a matrix.
 *
 * @param M - Input matrix.
 * @returns A tuple `[numRows, numCols]`.
 */
export function matrixShape(M: Matrix): [number, number] {
  return [M.length, M.length > 0 ? M[0].length : 0];
}
