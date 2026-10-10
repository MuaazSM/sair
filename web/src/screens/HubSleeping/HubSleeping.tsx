import { Logo, PrimaryButton, Screen } from '../../components/index.ts';
import type { HubSleepingData } from '../../fixtures/hub-sleeping.ts';

/** Computer asleep (design 2G): the note is queued on the phone and sends itself later. Nothing for them to do. */
export function HubSleeping({ data, onOkay }: { data: HubSleepingData; onOkay?: () => void }) {
  const star = (s: React.CSSProperties) => <div style={{ position: 'absolute', borderRadius: '50%', background: '#8FBF9A', ...s }} />;
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
      </div>
      <div aria-hidden="true" style={{ height: 232, flex: 'none', borderRadius: 28, background: '#2C4A34', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', left: '50%', top: '50%', width: 112, height: 112, margin: '-56px 0 0 -56px', borderRadius: '50%', background: '#FBF3E6', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', width: 112, height: 112, borderRadius: '50%', background: '#2C4A34', left: 38, top: -22 }} />
        </div>
        {star({ left: 58, top: 52, width: 8, height: 8 })}
        {star({ left: 96, top: 160, width: 6, height: 6 })}
        {star({ right: 62, top: 70, width: 6, height: 6 })}
        {star({ right: 88, bottom: 44, width: 8, height: 8 })}
      </div>
      <h1 style={{ margin: '28px 0 0', font: "800 34px/40px 'Nunito'", textWrap: 'balance' }}>{data.heading}</h1>
      <p style={{ margin: '12px 0 0', fontSize: 22, lineHeight: '32px' }}>{data.body}</p>
      <div style={{ marginTop: 20, flex: 'none', display: 'flex', alignItems: 'center', gap: 14, padding: '16px 18px', borderRadius: 20, background: '#EEF3EC' }}>
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#2C4A34" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none' }} aria-hidden="true">
          <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" />
          <path d="m9 12 2 2 4-4" />
        </svg>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ font: "700 20px/26px 'Nunito'" }}>{data.noteTitle}</div>
          <div style={{ fontSize: 20, lineHeight: '28px', color: '#5C6B60' }}>{data.noteSub}</div>
        </div>
      </div>
      <div style={{ flex: 1 }} />
      <div style={{ flex: 'none' }}>
        <PrimaryButton onClick={onOkay}>{data.okayLabel}</PrimaryButton>
      </div>
    </Screen>
  );
}
