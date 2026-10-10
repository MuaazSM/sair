import { FactChip, Icon, Logo, PrimaryButton, Screen } from '../../components/index.ts';
import type { SavedData } from '../../fixtures/saved.ts';

/** Saved (design 2D) and, with no facts and a `fromNote`, the short-note variant (design 2E). */
export function Saved({ data, onRead, onDone, showRead = true }: { data: SavedData; onRead?: () => void; onDone?: () => void; showRead?: boolean }) {
  const quiet = data.outcome?.kind === 'quiet';
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
      </div>
      <h1 style={{ margin: '4px 0 0', font: "800 40px/46px 'Nunito'" }}>{data.heading}</h1>
      <div
        style={{
          marginTop: 16,
          flex: 'none',
          background: '#FFFFFF',
          border: '1px solid #D9E0D6',
          borderRadius: 24,
          padding: 22,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          boxShadow: '0 1px 2px rgba(29,42,34,.06),0 6px 20px rgba(29,42,34,.06)',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ fontSize: 20, color: '#5C6B60' }}>{data.date}</div>
          {data.group ? <div style={{ font: "700 24px/30px 'Nunito'" }}>{data.group}</div> : null}
        </div>
        <div style={{ fontSize: 21, lineHeight: '31px', textWrap: 'pretty' }}>{data.entry}</div>
        {data.fromNote ? <div style={{ fontSize: 20, lineHeight: '28px', color: '#5C6B60' }}>{data.fromNote}</div> : null}
        {data.facts.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {data.facts.map((f, i) => (
              <FactChip key={i} kind={f.kind}>{f.text}</FactChip>
            ))}
          </div>
        ) : null}
      </div>
      {data.outcome ? (
      <div style={{ marginTop: 14, display: 'flex' }}>
        <span
          style={{
            alignSelf: 'flex-start',
            minHeight: 44,
            boxSizing: 'border-box',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 10,
            padding: '0 18px',
            borderRadius: 999,
            background: '#EEF3EC',
            color: '#2C4A34',
            font: "700 20px 'Nunito'",
          }}
        >
          <Icon name={quiet ? 'leaf' : 'check'} size={20} strokeWidth={quiet ? 1.75 : 2.25} />
          {data.outcome.label}
        </span>
      </div>
      ) : null}
      <div style={{ flex: 1 }} />
      {showRead ? (
      <>
      <button
        type="button"
        onClick={onRead}
        style={{
          width: '100%',
          minHeight: 64,
          flex: 'none',
          border: '2px solid #3F6B4A',
          borderRadius: 20,
          background: 'transparent',
          color: '#2C4A34',
          font: "700 22px 'Nunito'",
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
          cursor: 'pointer',
          transition: 'background-color 300ms cubic-bezier(.22,.61,.36,1)',
        }}
      >
        <Icon name="volume-2" size={26} />
        {data.readLabel}
      </button>
      <div style={{ height: 12, flex: 'none' }} />
      </>
      ) : null}
      <div style={{ flex: 'none' }}>
        <PrimaryButton onClick={onDone}>{data.doneLabel}</PrimaryButton>
      </div>
    </Screen>
  );
}
