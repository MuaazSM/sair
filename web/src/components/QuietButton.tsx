import type { ReactNode } from 'react';

type Props = { children: ReactNode; href?: string; onClick?: () => void };

/** The quiet text action under a primary button (e.g. "Back from your walk? Tell Sair"). */
export function QuietButton({ children, href = '#', onClick }: Props) {
  return (
    <a
      href={href}
      onClick={onClick ? (e) => { e.preventDefault(); onClick(); } : undefined}
      style={{
        marginTop: 12,
        minHeight: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        font: "700 21px 'Nunito'",
        color: '#2C4A34',
      }}
    >
      {children}
    </a>
  );
}
