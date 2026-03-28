/**
 * Tests for linalg module — pure linear algebra helpers.
 *
 * Every function is tested against hand-computable expected values
 * so correctness can be verified by reading the test alone.
 */

import {
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
} from "../linalg";
import { allClose } from "./helpers";

// ---------------------------------------------------------------------------
// vectorNorm
// ---------------------------------------------------------------------------

describe("vectorNorm", () => {
  it("computes the L2 norm of a 3-4-5 triangle", () => {
    expect(vectorNorm([3, 4])).toBeCloseTo(5, 10);
  });

  it("returns 0 for a zero vector", () => {
    expect(vectorNorm([0, 0, 0])).toBe(0);
  });

  it("returns the absolute value for a 1-D vector", () => {
    expect(vectorNorm([-7])).toBeCloseTo(7, 10);
  });

  it("handles unit vectors", () => {
    expect(vectorNorm([1, 0, 0])).toBeCloseTo(1, 10);
  });
});

// ---------------------------------------------------------------------------
// normalizeVector
// ---------------------------------------------------------------------------

describe("normalizeVector", () => {
  it("produces a unit-length vector", () => {
    const v = normalizeVector([3, 4]);
    expect(vectorNorm(v)).toBeCloseTo(1.0, 10);
    expect(v[0]).toBeCloseTo(0.6, 10);
    expect(v[1]).toBeCloseTo(0.8, 10);
  });
});

// ---------------------------------------------------------------------------
// dot
// ---------------------------------------------------------------------------

describe("dot", () => {
  it("computes the dot product of two vectors", () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it("returns 0 for orthogonal vectors", () => {
    expect(dot([1, 0], [0, 1])).toBe(0);
  });

  it("throws on mismatched lengths", () => {
    expect(() => dot([1, 2], [1, 2, 3])).toThrow("lengths must match");
  });
});

// ---------------------------------------------------------------------------
// argmax / argmin
// ---------------------------------------------------------------------------

describe("argmax", () => {
  it("returns the index of the maximum value", () => {
    expect(argmax([0.1, 0.4, 0.3])).toBe(1);
  });

  it("returns 0 for a single-element vector", () => {
    expect(argmax([42])).toBe(0);
  });

  it("breaks ties by first occurrence", () => {
    expect(argmax([5, 5, 5])).toBe(0);
  });
});

describe("argmin", () => {
  it("returns the index of the minimum value", () => {
    expect(argmin([0.3, 0.1, 0.4])).toBe(1);
  });

  it("breaks ties by first occurrence", () => {
    expect(argmin([1, 1, 1])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// sum / min
// ---------------------------------------------------------------------------

describe("sum", () => {
  it("sums all elements", () => {
    expect(sum([0.2, 0.3, 0.5])).toBeCloseTo(1.0, 10);
  });

  it("returns 0 for an empty vector", () => {
    expect(sum([])).toBe(0);
  });
});

describe("min", () => {
  it("returns the minimum element", () => {
    expect(min([3, 1, 4, 1, 5])).toBe(1);
  });

  it("handles negative values", () => {
    expect(min([-2, -5, -1])).toBe(-5);
  });
});

// ---------------------------------------------------------------------------
// matMul
// ---------------------------------------------------------------------------

describe("matMul", () => {
  it("multiplies 2x2 matrices correctly", () => {
    const A = [[1, 2], [3, 4]];
    const B = [[5, 6], [7, 8]];
    const result = matMul(A, B);
    expect(result).toEqual([[19, 22], [43, 50]]);
  });

  it("multiplies non-square matrices", () => {
    const A = [[1, 2, 3]]; // 1x3
    const B = [[4], [5], [6]]; // 3x1
    const result = matMul(A, B);
    expect(result).toEqual([[32]]); // 1x1
  });

  it("throws on dimension mismatch", () => {
    const A = [[1, 2]]; // 1x2
    const B = [[1, 2, 3]]; // 1x3 (inner dims don't match)
    expect(() => matMul(A, B)).toThrow("inner dimensions");
  });
});

// ---------------------------------------------------------------------------
// columnNorms / rowNorms
// ---------------------------------------------------------------------------

describe("columnNorms", () => {
  it("computes L2 norm of each column", () => {
    const M = [[3, 1], [4, 0]];
    const norms = columnNorms(M);
    expect(norms[0]).toBeCloseTo(5, 10);
    expect(norms[1]).toBeCloseTo(1, 10);
  });
});

describe("rowNorms", () => {
  it("computes L2 norm of each row", () => {
    const M = [[3, 4], [1, 0]];
    const norms = rowNorms(M);
    expect(norms[0]).toBeCloseTo(5, 10);
    expect(norms[1]).toBeCloseTo(1, 10);
  });
});

// ---------------------------------------------------------------------------
// divideColumns / divideRows
// ---------------------------------------------------------------------------

describe("divideColumns", () => {
  it("divides each column by the corresponding divisor", () => {
    const result = divideColumns([[6, 9], [4, 3]], [2, 3]);
    expect(result).toEqual([[3, 3], [2, 1]]);
  });
});

describe("divideRows", () => {
  it("divides each row by the corresponding divisor", () => {
    const result = divideRows([[6, 4], [9, 3]], [2, 3]);
    expect(result).toEqual([[3, 2], [3, 1]]);
  });
});

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

describe("transpose", () => {
  it("swaps rows and columns", () => {
    const M = [[1, 2, 3], [4, 5, 6]];
    expect(transpose(M)).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it("is its own inverse", () => {
    const M = [[1, 2], [3, 4]];
    expect(transpose(transpose(M))).toEqual(M);
  });
});

// ---------------------------------------------------------------------------
// matrixShape
// ---------------------------------------------------------------------------

describe("matrixShape", () => {
  it("returns [rows, cols]", () => {
    expect(matrixShape([[1, 2, 3], [4, 5, 6]])).toEqual([2, 3]);
  });

  it("handles empty matrix", () => {
    expect(matrixShape([])).toEqual([0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Integration: normalise → dot = cosine similarity
// ---------------------------------------------------------------------------

describe("cosine similarity via linalg primitives", () => {
  it("identical vectors have cosine similarity 1", () => {
    const v = normalizeVector([1, 2, 3]);
    expect(dot(v, v)).toBeCloseTo(1.0, 10);
  });

  it("orthogonal vectors have cosine similarity 0", () => {
    const a = normalizeVector([1, 0]);
    const b = normalizeVector([0, 1]);
    expect(dot(a, b)).toBeCloseTo(0, 10);
  });

  it("opposite vectors have cosine similarity -1", () => {
    const a = normalizeVector([1, 2]);
    const b = normalizeVector([-1, -2]);
    expect(dot(a, b)).toBeCloseTo(-1.0, 10);
  });
});
