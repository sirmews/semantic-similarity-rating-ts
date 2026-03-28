/**
 * Type declaration for the optional `@huggingface/transformers` peer dependency.
 *
 * This shim allows TypeScript to compile without the package installed.
 * At runtime, the dynamic `import()` in `embeddings.ts` will resolve to
 * the real package (or throw if not installed).
 */
declare module "@huggingface/transformers" {
  export function pipeline(
    task: string,
    model: string,
    options?: Record<string, unknown>
  ): Promise<(
    input: string | string[],
    options?: Record<string, unknown>
  ) => Promise<any>>;
}
