/**
 * @module openai-embeddings
 *
 * Embedding provider that uses the OpenAI embeddings API via raw `fetch`.
 *
 * This provider requires no additional dependencies — it uses the native
 * `fetch` API available in Node 18+.
 *
 * ## Usage
 *
 * ```ts
 * const provider = new OpenAIEmbeddingProvider({
 *   apiKey: process.env.OPENAI_API_KEY!,
 * });
 * const rater = await ResponseRater.create(refs, { embeddingProvider: provider });
 * ```
 *
 * ## OpenAI-compatible APIs
 *
 * The `baseURL` option supports any OpenAI-compatible endpoint (Azure OpenAI,
 * local proxies, etc.):
 *
 * ```ts
 * const provider = new OpenAIEmbeddingProvider({
 *   apiKey: "my-key",
 *   baseURL: "https://my-proxy.example.com/v1",
 * });
 * ```
 */

import type { Matrix, EmbeddingProvider } from "./types";

/**
 * Configuration options for {@link OpenAIEmbeddingProvider}.
 */
export interface OpenAIEmbeddingProviderOptions {
  /**
   * OpenAI API key. Required.
   */
  apiKey: string;

  /**
   * Embedding model identifier.
   * @default "text-embedding-3-small"
   */
  model?: string;

  /**
   * Base URL for the API. Allows using OpenAI-compatible endpoints
   * (Azure OpenAI, local proxies, etc.).
   * @default "https://api.openai.com/v1"
   */
  baseURL?: string;
}

/**
 * Shape of a single embedding object in the OpenAI API response.
 */
interface OpenAIEmbeddingObject {
  index: number;
  embedding: number[];
}

/**
 * Shape of the OpenAI embeddings API response.
 */
interface OpenAIEmbeddingsResponse {
  data: OpenAIEmbeddingObject[];
}

/**
 * Shape of the OpenAI API error response.
 */
interface OpenAIErrorResponse {
  error?: {
    message?: string;
  };
}

/**
 * Embedding provider that calls the OpenAI embeddings API using native `fetch`.
 *
 * Zero additional dependencies — works with Node 18+ or any environment
 * that provides a global `fetch`.
 *
 * ### Supported models
 *
 * - `text-embedding-3-small` (1536 dimensions, default)
 * - `text-embedding-3-large` (3072 dimensions)
 * - `text-embedding-ada-002` (1536 dimensions, legacy)
 *
 * ### Usage
 *
 * ```ts
 * const provider = new OpenAIEmbeddingProvider({
 *   apiKey: process.env.OPENAI_API_KEY!,
 *   model: "text-embedding-3-small",
 * });
 * const embeddings = await provider.encode(["Hello world", "How are you?"]);
 * // embeddings is number[][] with shape [2, 1536]
 * ```
 */
export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseURL: string;

  constructor(options: OpenAIEmbeddingProviderOptions) {
    this.apiKey = options.apiKey;
    this.model = options.model ?? "text-embedding-3-small";
    this.baseURL = (options.baseURL ?? "https://api.openai.com/v1").replace(
      /\/$/,
      ""
    );
  }

  /**
   * Compute embedding vectors for a batch of text strings using the
   * OpenAI embeddings API.
   *
   * @param texts - Array of sentences to embed.
   * @returns Matrix of shape `[texts.length, embeddingDim]`.
   * @throws {Error} On API errors (authentication, rate limiting, etc.).
   */
  async encode(texts: string[]): Promise<Matrix> {
    if (texts.length === 0) return [];

    const response = await fetch(`${this.baseURL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: texts,
        model: this.model,
      }),
    });

    if (!response.ok) {
      let message = `OpenAI API error: ${response.status} ${response.statusText}`;
      try {
        const body = (await response.json()) as OpenAIErrorResponse;
        if (body.error?.message) {
          message = `OpenAI API error: ${body.error.message}`;
        }
      } catch {
        // Use the default status message
      }
      throw new Error(message);
    }

    const body = (await response.json()) as OpenAIEmbeddingsResponse;

    // Sort by index to guarantee row order matches input order
    const sorted = body.data.slice().sort((a, b) => a.index - b.index);

    return sorted.map((item) => item.embedding);
  }
}
