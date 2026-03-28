/**
 * Tests for OpenAIEmbeddingProvider.
 *
 * All tests mock `global.fetch` — no real API calls are made.
 */

import { OpenAIEmbeddingProvider } from "../openai-embeddings";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a fake OpenAI embeddings API response. */
function fakeOpenAIResponse(
  embeddings: number[][],
  options?: { shuffleIndices?: boolean }
) {
  const data = embeddings.map((embedding, i) => ({
    object: "embedding" as const,
    index: i,
    embedding,
  }));

  // Optionally shuffle to verify index-based reordering
  if (options?.shuffleIndices) {
    data.reverse();
  }

  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => ({ object: "list", data, model: "text-embedding-3-small" }),
  };
}

/** Build a fake error response. */
function fakeErrorResponse(
  status: number,
  statusText: string,
  errorMessage?: string
) {
  return {
    ok: false,
    status,
    statusText,
    json: async () =>
      errorMessage
        ? { error: { message: errorMessage } }
        : {},
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("OpenAIEmbeddingProvider", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("returns correct matrix for a batch of texts", async () => {
    const embeddings = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ];
    global.fetch = jest.fn().mockResolvedValue(fakeOpenAIResponse(embeddings));

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });
    const result = await provider.encode(["hello", "world"]);

    expect(result).toEqual(embeddings);
    expect(result.length).toBe(2);
    expect(result[0].length).toBe(3);
  });

  it("returns empty array for empty input", async () => {
    global.fetch = jest.fn();

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });
    const result = await provider.encode([]);

    expect(result).toEqual([]);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("sorts results by index to match input order", async () => {
    const embeddings = [
      [0.1, 0.2],
      [0.3, 0.4],
      [0.5, 0.6],
    ];
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse(embeddings, { shuffleIndices: true }));

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });
    const result = await provider.encode(["a", "b", "c"]);

    // Should be in original order despite shuffled response
    expect(result).toEqual(embeddings);
  });

  it("uses default model text-embedding-3-small", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse([[0.1]]));

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });
    await provider.encode(["test"]);

    const call = (global.fetch as jest.Mock).mock.calls[0];
    const body = JSON.parse(call[1].body);
    expect(body.model).toBe("text-embedding-3-small");
  });

  it("uses custom model when specified", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse([[0.1]]));

    const provider = new OpenAIEmbeddingProvider({
      apiKey: "test-key",
      model: "text-embedding-3-large",
    });
    await provider.encode(["test"]);

    const call = (global.fetch as jest.Mock).mock.calls[0];
    const body = JSON.parse(call[1].body);
    expect(body.model).toBe("text-embedding-3-large");
  });

  it("uses default base URL", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse([[0.1]]));

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });
    await provider.encode(["test"]);

    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[0]).toBe("https://api.openai.com/v1/embeddings");
  });

  it("uses custom base URL", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse([[0.1]]));

    const provider = new OpenAIEmbeddingProvider({
      apiKey: "test-key",
      baseURL: "https://my-proxy.example.com/v1",
    });
    await provider.encode(["test"]);

    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[0]).toBe("https://my-proxy.example.com/v1/embeddings");
  });

  it("strips trailing slash from base URL", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse([[0.1]]));

    const provider = new OpenAIEmbeddingProvider({
      apiKey: "test-key",
      baseURL: "https://my-proxy.example.com/v1/",
    });
    await provider.encode(["test"]);

    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[0]).toBe("https://my-proxy.example.com/v1/embeddings");
  });

  it("sends correct authorization header", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeOpenAIResponse([[0.1]]));

    const provider = new OpenAIEmbeddingProvider({ apiKey: "sk-my-secret" });
    await provider.encode(["test"]);

    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[1].headers["Authorization"]).toBe("Bearer sk-my-secret");
  });

  it("throws on 401 unauthorized with API error message", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(
        fakeErrorResponse(401, "Unauthorized", "Invalid API key provided")
      );

    const provider = new OpenAIEmbeddingProvider({ apiKey: "bad-key" });

    await expect(provider.encode(["test"])).rejects.toThrow(
      "OpenAI API error: Invalid API key provided"
    );
  });

  it("throws on 429 rate limit with status text fallback", async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(fakeErrorResponse(429, "Too Many Requests"));

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });

    await expect(provider.encode(["test"])).rejects.toThrow(
      "OpenAI API error: 429 Too Many Requests"
    );
  });

  it("throws on 500 with non-JSON error body", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not JSON");
      },
    });

    const provider = new OpenAIEmbeddingProvider({ apiKey: "test-key" });

    await expect(provider.encode(["test"])).rejects.toThrow(
      "OpenAI API error: 500 Internal Server Error"
    );
  });
});
