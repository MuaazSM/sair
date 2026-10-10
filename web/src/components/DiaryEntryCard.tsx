import { FactChip } from './FactChip.tsx';
import { Icon } from './Icon.tsx';

export type DiaryEntry = {
  date: string;
  group: string;
  mission: string;
  /** null while the entry is still being written. */
  entry: string | null;
  facts: { kind: 'sure' | 'maybe'; text: string }[];
  askedAgain?: string;
  /** Shown instead of the entry while it is being written. NOT IN DESIGN (the design has no such card state). */
  writingLabel?: string;
};

/** One diary card (design 3A). `entry === null` is the being-written state (NOT IN DESIGN). */
export function DiaryEntryCard({ e }: { e: DiaryEntry }) {
  return (
    <article
      style={{
        background: '#FFFFFF',
        border: '1px solid #D9E0D6',
        borderRadius: 24,
        padding: 22,
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        boxShadow: '0 1px 2px rgba(29,42,34,.05),0 4px 14px rgba(29,42,34,.05)',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <h2 style={{ margin: 0, font: "800 26px/32px 'Nunito'" }}>{e.date}</h2>
        <div style={{ fontSize: 20, color: '#5C6B60' }}>{e.group}</div>
      </div>
      {e.mission ? (
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', fontSize: 20, lineHeight: '28px', fontStyle: 'italic', color: '#5C6B60' }}>
        <Icon name="sun" size={20} stroke="#A5641A" style={{ flex: 'none', marginTop: 4 }} />
        <span>{e.mission}</span>
      </div>
      ) : null}
      {e.entry === null ? (
        <div style={{ fontSize: 21, lineHeight: '31px', color: '#5C6B60' }}>{e.writingLabel}</div>
      ) : (
        <div style={{ fontSize: 21, lineHeight: '31px', textWrap: 'pretty' }}>{e.entry}</div>
      )}
      {e.facts.length > 0 ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {e.facts.map((f, i) => (
            <FactChip key={i} kind={f.kind}>{f.text}</FactChip>
          ))}
        </div>
      ) : null}
      {e.askedAgain ? (
        <a
          href="#"
          style={{ display: 'flex', alignItems: 'center', gap: 10, minHeight: 44, fontSize: 20, color: '#2C4A34', textDecoration: 'none', borderTop: '1px solid #D9E0D6', paddingTop: 14, marginTop: 2 }}
        >
          <Icon name="redo-2" size={22} stroke="#A5641A" style={{ flex: 'none' }} />
          <span style={{ textDecoration: 'underline', textUnderlineOffset: 4 }}>{e.askedAgain}</span>
        </a>
      ) : null}
    </article>
  );
}
