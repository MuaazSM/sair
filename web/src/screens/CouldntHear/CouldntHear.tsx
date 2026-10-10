import { Icon, Logo, Screen } from '../../components/index.ts';
import type { CouldntHearData } from '../../fixtures/couldnt-hear.ts';

/** Couldn't hear (design 2F): blames the room, never the person. One button; skipping is quiet. */
export function CouldntHear({ data, onTry, onSkip }: { data: CouldntHearData; onTry?: () => void; onSkip?: () => void }) {
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 22, paddingBottom: 32, textAlign: 'center' }}>
        <div style={{ width: 112, height: 112, borderRadius: '50%', background: '#FBF3E6', color: '#8A5214', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Icon name="mic" size={52} />
        </div>
        <h1 style={{ margin: 0, font: "800 36px/42px 'Nunito'", textWrap: 'balance' }}>{data.heading}</h1>
        <p style={{ margin: 0, fontSize: 22, lineHeight: '32px', maxWidth: 320, textWrap: 'pretty' }}>{data.body}</p>
      </div>
      <button
        type="button"
        className="sair-primary-btn"
        onClick={onTry}
        style={{
          width: '100%',
          minHeight: 80,
          flex: 'none',
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
        <Icon name="mic" size={28} />
        {data.tryLabel}
      </button>
      <a
        href="#"
        onClick={(e) => { e.preventDefault(); onSkip?.(); }}
        style={{ marginTop: 8, minHeight: 56, flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', font: "700 21px 'Nunito'", color: '#2C4A34' }}
      >
        {data.skipLabel}
      </a>
    </Screen>
  );
}
