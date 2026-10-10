import { Logo } from './Logo.tsx';

/** Logo + one muted line (a date, or "Today's walk"). Text comes from props. */
export function TopBar({ label }: { label: string }) {
  return (
    <div style={{ height: 72, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
      <Logo size={44} />
      <span style={{ fontSize: 22, color: '#5C6B60' }}>{label}</span>
    </div>
  );
}
