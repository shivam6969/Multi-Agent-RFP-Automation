// Thin fetch wrapper: JSON in/out, session cookie included, readable errors.

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function errorMessage(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown })?.detail;
  if (typeof detail === "string") return detail;
  // FastAPI validation errors: [{loc, msg}, ...]
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) =>
        [d.loc?.filter((x) => x !== "body").join("."), d.msg].filter(Boolean).join(": "),
      )
      .join("; ");
  }
  return fallback;
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init;
  const response = await fetch(path, {
    credentials: "same-origin",
    ...rest,
    headers: json !== undefined ? { "Content-Type": "application/json", ...headers } : headers,
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, errorMessage(body, `${response.status} ${response.statusText}`));
  }
  return body as T;
}

export function uploadPdf<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  return api<T>(path, { method: "POST", body: form });
}
