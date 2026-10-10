import type { CSSProperties, ReactNode } from 'react';

/** Height of the band the design spends on its mock OS status bar. The app does not draw a status bar (the phone
 *  has its own); the band is kept so every screen lays out exactly as in the design. */
export const STATUS_BAR_BAND = 32;

type Props = { children: ReactNode; statusBarSpacer?: boolean; style?: CSSProperties };

/** Device-frame content box from the design: column flex, 24px side/bottom padding, paper background.
 *  The design draws a 1px border in the screen background colour; it is kept as 1px of extra padding. */
export function Screen({ children, statusBarSpacer = true, style }: Props) {
  return (
    <main
      style={{
        width: '100%',
        height: '100%',
        boxSizing: 'border-box',
        position: 'relative',
        overflow: 'hidden',
        background: '#FAFAF6',
        display: 'flex',
        flexDirection: 'column',
        padding: '1px 25px 25px',
        color: '#1D2A22',
        ...style,
      }}
    >
      {statusBarSpacer ? <div aria-hidden="true" style={{ height: STATUS_BAR_BAND, flex: 'none' }} /> : null}
      {children}
    </main>
  );
}
