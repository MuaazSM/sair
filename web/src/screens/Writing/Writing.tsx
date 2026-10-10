import { Logo, Screen } from '../../components/index.ts';
import type { WritingData } from '../../fixtures/writing.ts';
import sairIconSvg from '../../assets/sair-icon.svg?raw';

// Same data the design's trace() uses: centre line of the ribbon (mask) and the ribbon path read from sair-icon.svg.
const CENTER =
  'M152.0 88.0 L143.0 88.5 L134.0 89.3 L125.0 90.3 L116.0 91.7 L107.1 93.2 L98.3 95.2 L89.6 97.7 L81.1 100.7 L72.9 104.6 L65.5 109.7 L60.5 116.7 L59.6 124.6 L62.3 131.5 L67.1 137.4 L73.4 142.5 L80.9 146.6 L89.1 149.6 L97.8 151.9 L106.6 153.8 L115.4 155.4 L124.3 157.0 L133.3 158.5 L142.1 160.2 L151.0 162.1 L159.8 164.3 L168.3 167.1 L176.5 170.7 L183.9 175.4 L190.2 181.1 L194.5 188.3 L192.2 196.7 L187.2 203.9 L181.1 210.3 L174.3 215.9 L166.9 221.0 L159.2 225.6 L151.2 229.7 L143.1 233.5 L134.8 237.0 L126.4 240.3 L117.9 243.2 L109.4 246.0 L103.6 250.4 L100.5 256.4 L97.4 262.3 L94.3 268.1 L91.1 273.9';
const RIBBON = /<path d="([^"]+)"/.exec(sairIconSvg)?.[1] ?? '';

/** Logo trace in its frozen state: the design's sair-trace/sair-trace-fade animations paused at 3.5 s of 5 s
 *  (ribbon fully drawn: dashoffset 0, opacity 1). Animation itself is not wired yet. */
function Trace() {
  return (
    <svg viewBox="0 0 256 256" width={200} height={200} aria-hidden="true">
      <defs>
        <clipPath id="trlc"><circle cx={128} cy={146} r={100} /></clipPath>
        <mask id="trlm" maskUnits="userSpaceOnUse" x={0} y={0} width={256} height={256}>
          <path d={CENTER} fill="none" stroke="#fff" strokeWidth={58} strokeLinecap="round" strokeLinejoin="round" pathLength={1} strokeDasharray="1 1" strokeDashoffset={0} />
        </mask>
      </defs>
      <circle cx={128} cy={146} r={100} fill="#3F6B4A" />
      <g clipPath="url(#trlc)" style={{ opacity: 1 }}>
        <path d={RIBBON} fill="#FAFAF6" mask="url(#trlm)" />
      </g>
      <circle cx={158} cy={53} r={37} fill="#FAFAF6" />
      <circle cx={158} cy={53} r={31} fill="#A5641A" />
    </svg>
  );
}

/** Saving (design 2C): "Writing it down…" with the logo trace. No progress bar, no percentages. */
export function Writing({ data }: { data: WritingData }) {
  return (
    <Screen>
      <div style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo size={44} />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 36, paddingBottom: 64 }}>
        <div style={{ width: 200, height: 200 }}>
          <Trace />
        </div>
        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
          <div style={{ font: "800 34px/40px 'Nunito'" }}>{data.heading}</div>
          <div style={{ fontSize: 21, lineHeight: '30px', color: '#5C6B60', maxWidth: 300, textWrap: 'pretty' }}>{data.body}</div>
        </div>
      </div>
    </Screen>
  );
}
