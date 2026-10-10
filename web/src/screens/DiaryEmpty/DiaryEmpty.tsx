import { Screen } from '../../components/index.ts';
import type { DiaryEmptyData } from '../../fixtures/diary-empty.ts';
import logoUrl from '../../assets/sair-icon.svg';

/** Diary, empty first use (design 3B): the bare logo path, nobody has walked it yet. */
export function DiaryEmpty({ data }: { data: DiaryEmptyData }) {
  return (
    <Screen style={{ padding: 1 }}>
      <div style={{ padding: '12px 24px 0', display: 'flex', alignItems: 'center', gap: 12 }}>
        <img src={logoUrl} alt="" style={{ width: 40, height: 40 }} />
        <h1 style={{ margin: 0, font: "800 36px/42px 'Nunito'" }}>{data.title}</h1>
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 28, padding: '0 32px 96px', textAlign: 'center' }}>
        <div style={{ width: 220, height: 220, borderRadius: '50%', background: '#EEF3EC', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <img src={logoUrl} alt="" style={{ width: 168, height: 168 }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
          <div style={{ font: "800 30px/36px 'Nunito'", textWrap: 'balance' }}>{data.heading}</div>
          <div style={{ fontSize: 21, lineHeight: '30px', color: '#5C6B60', maxWidth: 300, textWrap: 'pretty' }}>{data.body}</div>
        </div>
      </div>
    </Screen>
  );
}
