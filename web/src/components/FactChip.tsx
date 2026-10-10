import type { ReactNode } from 'react';
import { Icon } from './Icon.tsx';

type Props = { kind: 'sure' | 'maybe'; children: ReactNode };

/** A fact chip (design 2D/3A). Sure = solid with a check; maybe = dotted, prefixed by the italic word "maybe". */
export function FactChip({ kind, children }: Props) {
  const base = {
    minHeight: 44,
    boxSizing: 'border-box',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '0 16px',
    borderRadius: 999,
    fontSize: 20,
  } as const;
  if (kind === 'maybe') {
    return (
      <span style={{ ...base, background: 'transparent', border: '2px dotted #5C6B60', color: '#1D2A22' }}>
        <span style={{ fontStyle: 'italic', color: '#5C6B60' }}>maybe</span>
        {children}
      </span>
    );
  }
  return (
    <span style={{ ...base, background: '#EEF3EC', border: '1.5px solid #3F6B4A', color: '#2C4A34' }}>
      <Icon name="check" size={18} strokeWidth={2.25} />
      {children}
    </span>
  );
}
