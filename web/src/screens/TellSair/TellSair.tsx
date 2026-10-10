import { HoldToRecord, Logo, MissionNote, Screen } from '../../components/index.ts';
import type { TellSairData } from '../../fixtures/tell-sair.ts';

/** Tell Sair (design 2A): hold-to-talk idle. The mission sits right above the button. */
export function TellSair({ data }: { data: TellSairData }) {
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
        <span style={{ fontSize: 22, color: '#5C6B60' }}>{data.date}</span>
      </div>
      <h1 style={{ margin: '16px 0 0', font: "800 42px/48px 'Nunito'", textWrap: 'balance' }}>{data.heading}</h1>
      <p style={{ margin: '12px 0 0', fontSize: 22, lineHeight: '32px', textWrap: 'pretty' }}>{data.body}</p>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', gap: 24, paddingBottom: 8 }}>
        <MissionNote text={data.mission} />
        <HoldToRecord mode="idle" label={data.holdLabel} />
        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ font: "800 28px/34px 'Nunito'" }}>{data.holdLabel}</div>
          <div style={{ fontSize: 20, color: '#5C6B60' }}>{data.hint}</div>
        </div>
      </div>
    </Screen>
  );
}
