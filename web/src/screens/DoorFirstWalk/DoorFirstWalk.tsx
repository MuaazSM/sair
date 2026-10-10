import { Icon } from '../../components/index.ts';
import type { DoorData } from '../../fixtures/door.ts';
import { Door } from '../Door/index.ts';

type Props = { data: DoorData & { note: string }; onHear?: () => void; onTell?: () => void };

/** First-walk variant of the door (design 1D): same markup plus the one-time leaf line. */
export function DoorFirstWalk({ data, onHear, onTell }: Props) {
  return (
    <Door
      data={data}
      onHear={onHear}
      onTell={onTell}
      extra={
        <div style={{ marginTop: 20, display: 'flex', gap: 12, alignItems: 'flex-start', padding: '0 4px', fontSize: 21, lineHeight: '30px', color: '#1D2A22' }}>
          <Icon name="leaf" size={26} stroke="#3F6B4A" style={{ flex: 'none', marginTop: 3 }} />
          <span>{data.note}</span>
        </div>
      }
    />
  );
}
