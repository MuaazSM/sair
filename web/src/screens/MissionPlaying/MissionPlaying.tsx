import { PlayRing, Screen, TopBar } from '../../components/index.ts';
import type { MissionPlayingData } from '../../fixtures/mission-playing.ts';

type Props = { data: MissionPlayingData; onPause?: () => void };

/** Mission playing (design 1B): words fade in as spoken, ring shows playback, one status line. */
export function MissionPlaying({ data, onPause }: Props) {
  return (
    <Screen>
      <TopBar label={data.heading} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 52, paddingBottom: 40 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', columnGap: '0.28em', rowGap: 4, font: "700 28px/40px 'Nunito'", textAlign: 'center', maxWidth: 330 }}>
          {data.words.map((item, i) => (
            <span key={i} style={{ opacity: item.visible ? 1 : 0, transition: 'opacity 450ms cubic-bezier(.22,.61,.36,1)' }}>{item.w}</span>
          ))}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22 }}>
          <PlayRing progress={data.progress} pauseLabel={data.pauseLabel} onPause={onPause} />
          <div style={{ font: "700 22px 'Nunito'", color: '#5C6B60', minHeight: 30 }}>{data.playLabel}</div>
        </div>
      </div>
    </Screen>
  );
}
