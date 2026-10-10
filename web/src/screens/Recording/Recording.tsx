import { HoldToRecord, Logo, MissionNote, Screen } from '../../components/index.ts';
import type { RecordingData } from '../../fixtures/recording.ts';

type Props = { data: RecordingData; animate?: boolean };

/** Recording (design 2B) and its last-ten-seconds variant (2B', when data.countdown is set). Static states only. */
export function Recording({ data, animate = true }: Props) {
  const warning = data.countdown !== undefined;
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
        <span style={{ fontSize: 22, color: '#5C6B60' }}>{data.title}</span>
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 28, paddingBottom: 24 }}>
        <MissionNote text={data.mission} />
        <div style={{ font: "800 64px/64px 'Nunito'", fontVariantNumeric: 'tabular-nums' }}>{data.time}</div>
        <HoldToRecord mode={warning ? 'warning' : 'recording'} progress={data.progress} animate={animate} label={data.holdLabel} />
        {warning ? (
          <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ font: "800 26px/32px 'Nunito'", color: '#8A5214' }}>{data.countdown}</div>
            <div style={{ fontSize: 20, color: '#5C6B60' }}>{data.hint}</div>
          </div>
        ) : (
          <div style={{ font: "700 24px/30px 'Nunito'" }}>{data.hint}</div>
        )}
      </div>
    </Screen>
  );
}
