import { Icon } from './Icon.tsx';

export type HoldToRecordMode = 'idle' | 'recording' | 'warning';

type Props = {
  mode: HoldToRecordMode;
  /** 0..1 share of the 60 s ring that is filled (recording mode only). */
  progress?: number;
  /** Breathing ring in warning mode (design keyframes sair-pulse, 2.4 s). Off for static frames. */
  animate?: boolean;
  label: string;
};

const CIRCUMFERENCE = 666.0;
const WARNING_OFFSET = 111.0;

const BUTTON_BASE = {
  position: 'absolute',
  border: 0,
  borderRadius: '50%',
  color: '#FAFAF6',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  touchAction: 'none',
  transition: 'inset 400ms cubic-bezier(.22,.61,.36,1),background-color 400ms cubic-bezier(.22,.61,.36,1)',
} as const;

/** One-line mission reminder with the sun icon (tell Sair, recording). */
export function MissionNote({ text }: { text: string }) {
  return (
    <div style={{ alignSelf: 'stretch', display: 'flex', alignItems: 'center', gap: 12, padding: '14px 18px', borderRadius: 18, background: '#EEF3EC', fontSize: 20, lineHeight: '28px', color: '#1D2A22' }}>
      <Icon name="sun" size={24} stroke="#A5641A" style={{ flex: 'none' }} />
      <span>{text}</span>
    </div>
  );
}

/** Hold-to-record ring + round button in its static design states: idle, recording, 10-second warning. */
export function HoldToRecord({ mode, progress = 0, animate = true, label }: Props) {
  const active = mode !== 'idle';
  return (
    <div style={{ position: 'relative', width: 232, height: 232, flex: 'none' }}>
      <svg width="232" height="232" viewBox="0 0 232 232" aria-hidden="true" style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}>
        <circle cx="116" cy="116" r="106" fill="none" stroke="#D9E0D6" strokeWidth="8" />
        {mode === 'recording' ? (
          <circle cx="116" cy="116" r="106" fill="none" stroke="#A5641A" strokeWidth="8" strokeLinecap="round" strokeDasharray={CIRCUMFERENCE} strokeDashoffset={(CIRCUMFERENCE * (1 - progress)).toFixed(1)} />
        ) : null}
        {mode === 'warning' ? (
          <circle cx="116" cy="116" r="106" fill="none" stroke="#A5641A" strokeWidth="8" strokeLinecap="round" strokeDasharray={CIRCUMFERENCE} strokeDashoffset={WARNING_OFFSET} style={animate ? { animation: 'sair-pulse 2.4s ease-in-out infinite' } : undefined} />
        ) : null}
      </svg>
      <button type="button" aria-label={label} style={{ ...BUTTON_BASE, inset: active ? 22 : 28, background: active ? '#2C4A34' : '#3F6B4A' }}>
        <Icon name="mic" size={64} />
      </button>
    </div>
  );
}
