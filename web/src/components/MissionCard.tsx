import { Icon } from './Icon.tsx';

type Props = {
  title: string;
  /** Width (percent) of each soft line standing in for the hidden mission text. */
  lineWidths: number[];
  caption: string;
};

/** "Today's walk" card: the mission is shown as soft lines so it is heard before it is read. */
export function MissionCard({ title, lineWidths, caption }: Props) {
  return (
    <div style={{ marginTop: 32, background: '#EEF3EC', borderRadius: 24, padding: 28, display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, font: "700 22px 'Nunito'", color: '#2C4A34' }}>
        <Icon name="sun" size={28} stroke="#A5641A" />
        {title}
      </div>
      <div aria-hidden="true" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {lineWidths.map((w, i) => (
          <div key={i} style={{ height: 20, borderRadius: 10, background: '#D8E3D5', width: `${w}%` }} />
        ))}
      </div>
      <div style={{ fontSize: 20, lineHeight: '28px', color: '#5C6B60' }}>{caption}</div>
    </div>
  );
}
