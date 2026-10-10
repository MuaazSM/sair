import type { ReactNode } from 'react';
import { MissionCard, PrimaryButton, QuietButton, Screen, TopBar } from '../../components/index.ts';
import type { DoorData } from '../../fixtures/door.ts';

type Props = { data: DoorData; extra?: ReactNode; onHear?: () => void; onTell?: () => void };

/** Door screen (design 1A). `extra` is rendered between the card and the primary button (first-walk note). */
export function Door({ data, extra, onHear, onTell }: Props) {
  return (
    <Screen>
      <TopBar label={data.date} />
      <h1 style={{ margin: '28px 0 0', font: "800 44px/52px 'Nunito'", textWrap: 'balance' }}>{data.heading}</h1>
      <MissionCard title={data.card.title} lineWidths={data.card.lineWidths} caption={data.card.caption} />
      {extra}
      <div style={{ flex: 1 }} />
      <PrimaryButton icon="volume-2" onClick={onHear}>{data.cta}</PrimaryButton>
      <QuietButton onClick={onTell}>{data.link}</QuietButton>
    </Screen>
  );
}
