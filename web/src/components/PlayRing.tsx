import { Icon } from './Icon.tsx';

const CIRCUMFERENCE = 490.09;

type Props = {
  /** 0 = nothing played, 1 = ring full. */
  progress: number;
  pauseLabel: string;
  onPause?: () => void;
};

/** Progress ring with the round pause button in the middle (mission playing). */
export function PlayRing({ progress, pauseLabel, onPause }: Props) {
  const offset = (CIRCUMFERENCE * (1 - progress)).toFixed(1);
  return (
    <div style={{ position: 'relative', width: 184, height: 184 }}>
      <svg width="184" height="184" viewBox="0 0 184 184" aria-hidden="true" style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}>
        <circle cx="92" cy="92" r="78" fill="none" stroke="#D9E0D6" strokeWidth="8" />
        <circle
          cx="92"
          cy="92"
          r="78"
          fill="none"
          stroke="#3F6B4A"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray="490.09"
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 150ms linear' }}
        />
      </svg>
      <button
        type="button"
        className="sair-round-btn"
        aria-label={pauseLabel}
        onClick={onPause}
        style={{
          position: 'absolute',
          inset: 28,
          border: 0,
          borderRadius: '50%',
          background: '#3F6B4A',
          color: '#FAFAF6',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
      >
        <Icon name="pause" size={44} />
      </button>
    </div>
  );
}
