// Same-origin API client (PRD §9.3). The device token goes only into the Authorization header of requests to this
// origin. Nothing here logs; error messages are fixed strings (no content, no URLs with ids).

const TOKEN_KEY = 'sair.token';
const WALKED_KEY = 'sair.walked';

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* storage unavailable: the app asks for setup again next time */
  }
}

/** First-walk flag (Door variant). Local to this device; true once a walk has been saved. */
export function hasWalked(): boolean {
  try {
    return localStorage.getItem(WALKED_KEY) === '1';
  } catch {
    return false;
  }
}

export function markWalked(): void {
  try {
    localStorage.setItem(WALKED_KEY, '1');
  } catch {
    /* ignore */
  }
}

export class HttpError extends Error {
  constructor(readonly status: number) {
    super('http_error');
  }
}

/** Network failure, 5xx, 408, 429 and auth problems (kept for retry) vs. a request the hub will never accept. */
export function isTransient(e: unknown): boolean {
  if (e instanceof HttpError) return e.status >= 500 || e.status === 408 || e.status === 429 || e.status === 401 || e.status === 403;
  return true;
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(path, { ...init, headers, cache: 'no-store', credentials: 'omit' });
  if (!res.ok) throw new HttpError(res.status);
  return res;
}

export type MissionToday = { mission_id: string; text: string; type: string; audio_path?: string | null };
export type FactChipDto = { subject: string; kind: string; certainty: 'sure' | 'unsure'; place?: string | null };
export type WalkStatusDto = {
  status: 'received' | 'done' | 'lost';
  date: string;
  entry?: string;
  facts?: FactChipDto[];
  mission_result?: 'done' | 'partly' | 'not_done' | 'not_mentioned';
  reason?: string;
};
export type DiaryEntryDto = {
  walk_id: string;
  date: string;
  group: string;
  mission_text?: string | null;
  entry: string;
  facts: (FactChipDto & { asked_again_on: string[] })[];
  mission_result: string;
};
export type DiaryDto = { summary: { walks_this_month: number; things_noticed: number }; entries: DiaryEntryDto[] };

export async function getMission(): Promise<MissionToday> {
  return (await request('/mission/today')).json();
}

/** Fetch the mission audio with the Authorization header and return a blob URL (the caller revokes it). */
export async function getAudioBlobUrl(path: string): Promise<string> {
  const res = await request(path);
  return URL.createObjectURL(await res.blob());
}

export async function postWalk(audio: Blob): Promise<{ walk_id: string }> {
  const body = new FormData();
  body.append('audio', audio, audio.type.includes('mp4') ? 'walk.m4a' : 'walk.webm');
  return (await request('/walk', { method: 'POST', body })).json();
}

export async function getWalk(id: string): Promise<WalkStatusDto> {
  return (await request(`/walk/${encodeURIComponent(id)}`)).json();
}

export async function getDiary(group?: string): Promise<DiaryDto> {
  return (await request(group ? `/diary?group=${encodeURIComponent(group)}` : '/diary')).json();
}
