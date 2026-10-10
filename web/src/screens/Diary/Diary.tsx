import { DiaryEntryCard, DiaryFilterChips, DiarySummary, Screen } from '../../components/index.ts';
import type { DiaryData } from '../../fixtures/diary.ts';
import logoUrl from '../../assets/sair-icon.svg';

/** Diary list, mobile (design 3A): header scrolls away with the page; one quiet summary line; no streaks. */
export function Diary({ data, onFilter, empty }: { data: DiaryData; onFilter?: (index: number) => void; empty?: { heading: string; body: string } }) {
  return (
    <Screen style={{ padding: 1 }}>
      <div style={{ flex: 1, overflowY: 'auto', scrollbarWidth: 'none', padding: '4px 20px 40px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <header style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '8px 4px 8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <img src={logoUrl} alt="" style={{ width: 40, height: 40 }} />
            <h1 style={{ margin: 0, font: "800 36px/42px 'Nunito'" }}>{data.title}</h1>
          </div>
          {empty ? null : <DiarySummary>{data.summary}</DiarySummary>}
          {empty ? null : <DiaryFilterChips labels={data.filters} activeIndex={data.activeFilter} onSelect={onFilter} />}
        </header>
        {empty ? (
          // NOT IN DESIGN: the empty diary (design 3B) has a bare-path illustration; only its copy is used here.
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '24px 4px' }}>
            <h2 style={{ margin: 0, font: "800 30px/36px 'Nunito'" }}>{empty.heading}</h2>
            <p style={{ margin: 0, fontSize: 21, lineHeight: '30px', color: '#5C6B60' }}>{empty.body}</p>
          </div>
        ) : (
          <>
            {data.entries.map((e, i) => (
              <DiaryEntryCard key={`${e.date}-${i}`} e={e} />
            ))}
            <div style={{ textAlign: 'center', fontSize: 20, color: '#5C6B60', padding: '16px 0' }}>{data.footer}</div>
          </>
        )}
      </div>
    </Screen>
  );
}
