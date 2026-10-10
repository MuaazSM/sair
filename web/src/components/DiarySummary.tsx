/** The one quiet summary line under the diary title (FR-D3: no streaks, no per-person counts). */
export function DiarySummary({ children }: { children: string }) {
  return <div style={{ fontSize: 20, lineHeight: '28px', color: '#5C6B60' }}>{children}</div>;
}
