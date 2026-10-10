import { Logo, PrimaryButton, QuietButton, Screen } from '../../components/index.ts';
import type { LeavePhoneData } from '../../fixtures/leave-phone.ts';

type Props = { data: LeavePhoneData; dimmed?: boolean; onOkay?: () => void; onWake?: () => void; onHear?: () => void };

const abs = (s: React.CSSProperties): React.CSSProperties => ({ position: 'absolute', ...s });

/** Leave the phone (design 1C): flat phone-on-a-table scene built from shapes, then "Okay" dims the screen. */
export function LeavePhone({ data, dimmed = false, onOkay, onWake, onHear }: Props) {
  return (
    <Screen>
      <div style={{ height: 72, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
      </div>
      <div aria-hidden="true" style={{ height: 300, flex: 'none', borderRadius: 28, overflow: 'hidden', position: 'relative', background: '#FBF3E6' }}>
        <div style={abs({ left: 0, right: 0, bottom: 0, height: 64, background: '#F2E6D2' })} />
        <div style={abs({ top: 16, right: 26, width: 64, height: 64, borderRadius: '50%', background: '#F5E2C4', border: '6px solid #FAFAF6', overflow: 'hidden', boxSizing: 'content-box' })}>
          <div style={abs({ left: 15, bottom: -8, width: 34, height: 34, borderRadius: '50%', background: '#A5641A' })} />
        </div>
        <div style={abs({ left: 32, bottom: 64, width: 112, height: 200, borderRadius: '56px 56px 6px 6px', background: '#3F6B4A' })} />
        <div style={abs({ left: 120, bottom: 150, width: 12, height: 12, borderRadius: '50%', background: '#A5641A' })} />
        <div style={abs({ right: 26, bottom: 120, width: 140, height: 12, borderRadius: 6, background: '#2C4A34' })} />
        <div style={abs({ right: 42, bottom: 64, width: 8, height: 56, borderRadius: 4, background: '#2C4A34' })} />
        <div style={abs({ right: 142, bottom: 64, width: 8, height: 56, borderRadius: 4, background: '#2C4A34' })} />
        <div style={abs({ right: 70, bottom: 132, width: 44, height: 70, borderRadius: 10, background: '#1D2A22', padding: 4, boxSizing: 'border-box' })}>
          <div style={{ width: '100%', height: '100%', borderRadius: 7, background: '#EEF3EC', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Logo size={24} alt="" />
          </div>
        </div>
        <div style={abs({ right: 30, bottom: 24, width: 160, height: 18, borderRadius: '50%', background: '#FAFAF6' })} />
      </div>
      <h1 style={{ margin: '32px 0 0', font: "800 42px/48px 'Nunito'" }}>{data.heading}</h1>
      <p style={{ margin: '12px 0 0', fontSize: 22, lineHeight: '32px', color: '#1D2A22', textWrap: 'pretty' }}>{data.body}</p>
      <div style={{ flex: 1 }} />
      <PrimaryButton onClick={onOkay}>{data.cta}</PrimaryButton>
      <QuietButton href="#" onClick={onHear}>{data.link}</QuietButton>
      <div
        onClick={onWake}
        style={abs({ inset: 0, background: '#0E1311', opacity: dimmed ? 0.94 : 0, pointerEvents: dimmed ? 'auto' : 'none', transition: 'opacity 500ms cubic-bezier(.22,.61,.36,1)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 18, cursor: 'pointer' })}
      >
        <Logo size={56} alt="" style={{ opacity: 0.5 }} />
        <span style={{ font: "700 20px 'Nunito'", color: '#A9B5AC' }}>{data.wakeLabel}</span>
      </div>
    </Screen>
  );
}
