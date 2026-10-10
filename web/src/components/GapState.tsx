import { Logo } from './Logo.tsx';
import { PrimaryButton } from './PrimaryButton.tsx';
import { QuietButton } from './QuietButton.tsx';
import { Screen } from './Screen.tsx';

type Props = {
  heading: string;
  body: string;
  /** Two quiet lines in a soft card (hub sleeping: "Your note is safe on this phone" / "0:41 · it will send by itself"). */
  note?: [string, string];
  primary?: { label: string; onClick: () => void };
  quiet?: { label: string; onClick: () => void };
};

/** NOT IN DESIGN (PRD UI-7, REVIEW "Gap states"): one layout for the states the design does not draw as screens
 *  (couldn't hear, hub sleeping, unreachable, set-up needed, lost). Built only from existing tokens, Screen, Logo and buttons. */
export function GapState({ heading, body, note, primary, quiet }: Props) {
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
      </div>
      <h1 style={{ margin: '16px 0 0', font: "800 40px/46px 'Nunito'", textWrap: 'balance' }}>{heading}</h1>
      <p style={{ margin: '12px 0 0', fontSize: 22, lineHeight: '32px', textWrap: 'pretty' }}>{body}</p>
      {note ? (
        <div style={{ marginTop: 24, padding: '14px 18px', borderRadius: 18, background: '#EEF3EC', color: '#1D2A22', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{ font: "700 22px/30px 'Nunito'" }}>{note[0]}</div>
          <div style={{ fontSize: 20, lineHeight: '28px', color: '#5C6B60' }}>{note[1]}</div>
        </div>
      ) : null}
      <div style={{ flex: 1 }} />
      {primary ? <PrimaryButton onClick={primary.onClick}>{primary.label}</PrimaryButton> : null}
      {quiet ? <QuietButton onClick={quiet.onClick}>{quiet.label}</QuietButton> : null}
    </Screen>
  );
}
