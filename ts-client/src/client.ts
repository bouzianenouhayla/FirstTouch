const BASE_URL = process.env.API_URL ?? "http://localhost:8000";

export type Backend = "local" | "anthropic" | "agent" | "multi-agent";

export interface AskRequest {
  question: string;
  backend?: Backend;
  max_contexts?: number;
  session_id?: string;
}

export interface RetrievedContext {
  chunk_id: string;
  text: string;
  score: number;
}

export interface AskResponse {
  answer: string;
  contexts: RetrievedContext[];
  config_name: string;
  total_time_ms: number;
}

export interface UserProfile {
  session_id: string;
  position: "goalkeeper" | "defender" | "midfielder" | "forward";
  level: "just started" | "beginner" | "intermediate";
  sessions_per_week: number;
  playing_since:
    | "less than 3 months"
    | "3-6 months"
    | "6-12 months"
    | "1-2 years"
    | "2+ years";
  age: number;
  goal?: string;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export async function health(): Promise<{ status: string; backends: Backend[] }> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return res.json();
}

export async function ask(payload: AskRequest): Promise<AskResponse> {
  return post<AskResponse>("/ask", {
    session_id: crypto.randomUUID(),
    ...payload,
  });
}

export async function saveProfile(profile: UserProfile): Promise<{ status: string; facts_stored: number }> {
  return post("/profile", profile);
}
