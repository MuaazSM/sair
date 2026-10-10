import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon.tsx';

type Props = { icon?: IconName; children: ReactNode; onClick?: () => void };

/** The one primary action: full width, 80px min height, leaf green. */
export function PrimaryButton({ icon, children, onClick }: Props) {
  return (
    <button
      type="button"
      className="sair-primary-btn"
      onClick={onClick}
      style={{
        width: '100%',
        minHeight: 80,
        border: 0,
        borderRadius: 22,
        background: '#3F6B4A',
        color: '#FAFAF6',
        font: "700 24px 'Nunito'",
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 14,
        cursor: 'pointer',
        transition: 'background-color 300ms cubic-bezier(.22,.61,.36,1)',
      }}
    >
      {icon ? <Icon name={icon} size={30} /> : null}
      {children}
    </button>
  );
}
