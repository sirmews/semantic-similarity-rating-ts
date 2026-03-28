/**
 * @module embeddings
 *
 * Embedding provider implementations for computing sentence embeddings.
 *
 * This module provides:
 * - {@link TransformersEmbeddingProvider}: Production provider using `@xenova/transformers`.
 * - {@link EmbeddingProvider} interface (from `types.ts`): Contract for dependency injection.
 *
 * ## Testability
 *
 * The {@link EmbeddingProvider} interface allows tests to inject a deterministic
 * mock instead of loading a real model. See the `types.ts` module for an example
 * mock implementation.
 *
 * ## Correspondence to Python
 *
 * | Python                                         | This module                         |
 * | ---------------------------------------------- | ----------------------------------- |
 * | `SentenceTransformer(model_name)`              | {@link TransformersEmbeddingProvider} constructor |
 * | `model.encode(texts)`                          | {@link TransformersEmbeddingProvider.encode}      |
 *
 * ## Package note
 *
 * This module uses `@huggingface/transformers` (v3+), which supersedes the
 * legacy `@xenova/transformers` (v2) package. The v3 package adds WebGPU
 * support and is actively maintained by Hugging Face.
 */

import type { Matrix, EmbeddingProvider } from "./types";

/**
 * Embedding provider that uses `@xenova/transformers` for model inference.
 *
 * This loads the same `all-MiniLM-L6-v2` model used by the Python
 * `sentence-transformers` library, running locally via ONNX/WebAssembly.
 * Embeddings are numerically equivalent (within floating-point tolerance)
 * to the Python implementation.
 *
 * ### Usage
 *
 * ```ts
 * const provider = new TransformersEmbeddingProvider("Xenova/all-MiniLM-L6-v2");
 * await provider.load();
 * const embeddings = await provider.encode(["Hello world", "How are you?"]);
 * // embeddings is number[][] with shape [2, 384]
 * ```
 *
 * ### First-time model download
 *
 * The first call to {@link load} or {@link encode} downloads the ONNX model
 * (~50 MB) and caches it locally. Subsequent calls use the cache.
 */
export class TransformersEmbeddingProvider implements EmbeddingProvider {
  private pipeline: any = null;
  private readonly modelName: string;

  /**
   * @param modelName - HuggingFace model identifier.
   *   Defaults to `"Xenova/all-MiniLM-L6-v2"` (384-dimensional embeddings).
   */
  constructor(modelName: string = "Xenova/all-MiniLM-L6-v2") {
    this.modelName = modelName;
  }

  /**
   * Lazily load the model pipeline. Called automatically by {@link encode}
   * if not already loaded.
   *
   * Exposed publicly so callers can pre-warm the model before first use.
   */
  async load(): Promise<void> {
    if (this.pipeline) return;

    // Dynamic import to keep @huggingface/transformers as an optional peer dependency.
    // This means the library can be installed without it (for embedding-mode-only usage).
    const { pipeline } = await import("@huggingface/transformers");
    this.pipeline = await pipeline("feature-extraction", this.modelName, {
      quantized: false,
    });
  }

  /**
   * Compute embedding vectors for a batch of text strings.
   *
   * Each output row is a mean-pooled, L2-normalised embedding matching
   * the output of Python's `SentenceTransformer.encode()`.
   *
   * @param texts - Array of sentences to embed.
   * @returns Matrix of shape `[texts.length, embeddingDim]`.
   */
  async encode(texts: string[]): Promise<Matrix> {
    if (texts.length === 0) return [];

    await this.load();

    const output = await this.pipeline(texts, {
      pooling: "mean",
      normalize: true,
    });

    // Convert from the library's Tensor format to plain number[][]
    const numTexts = texts.length;
    const embeddingDim = output.dims[output.dims.length - 1];
    const result: Matrix = [];

    for (let i = 0; i < numTexts; i++) {
      const row: number[] = [];
      for (let j = 0; j < embeddingDim; j++) {
        row.push(output.data[i * embeddingDim + j]);
      }
      result.push(row);
    }

    return result;
  }
}
