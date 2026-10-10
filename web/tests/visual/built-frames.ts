// Frames that have a built screen. Each screen batch appends its ids here (and to src/dev/frames.tsx).
// textExclude: strings present in the design frame but deliberately not rendered by the build
//   (the mock OS status-bar clock, D-005). Excluded explicitly per frame, never silently.
// axeAllow: axe rule ids that are design-inherent for that frame (documented in REVIEW.md); empty by default.
export type BuiltFrame = { id: string; textExclude: string[]; axeAllow?: string[] };

export const BUILT_FRAMES: BuiltFrame[] = [
  { id: 'door', textExclude: ['6:42'] },
  { id: 'door-first-walk', textExclude: ['6:42'] },
  { id: 'mission-playing', textExclude: ['6:42'] },
  { id: 'leave-phone', textExclude: ['6:43'] },
  { id: 'tell-sair', textExclude: ['6:58'] },
  { id: 'recording', textExclude: ['6:58'] },
  { id: 'recording-warning', textExclude: ['6:59'] },
  { id: 'writing', textExclude: ['6:59'] },
  { id: 'saved', textExclude: ['7:00'] },
  { id: 'short-note', textExclude: ['7:00'] },
  { id: 'diary', textExclude: ['8:15'] },
  { id: 'couldnt-hear', textExclude: ['6:59'] },
  { id: 'hub-sleeping', textExclude: ['7:00'] },
  { id: 'diary-empty', textExclude: ['8:15'] },
];
