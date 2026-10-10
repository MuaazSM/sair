// Hand-written frame metadata. Strings and everything else are filled from the live design by build-inventory.ts.
export type Meta = {
  id: string; file: string; wrapper: string; tier: 'Must' | 'Should' | 'Could'; theme: 'light' | 'dark';
  component: string; route: string; state: string; selector?: string; viewport?: { width: number; height: number };
};
const B = 'Before the Walk.dc.html', A = 'After the Walk.dc.html', F = 'Family Diary.dc.html', I = 'Sair on iPhone.dc.html', H = 'Hub Status.dc.html';
const MOB = { width: 390, height: 844 };
type Base = Omit<Meta, 'theme' | 'tier'> & { tier: Meta['tier'] };
const light: Base[] = [
  { id: 'door', file: B, wrapper: '1a', tier: 'Must', component: 'Door', route: '/', state: 'Door screen, normal walk (mission hidden behind soft lines)' },
  { id: 'mission-playing', file: B, wrapper: '1b', tier: 'Must', component: 'MissionPlaying', route: '/mission', state: 'Mission audio playing; design default t=7.6 s (between the two plays, "Playing again in 2 seconds")' },
  { id: 'leave-phone', file: B, wrapper: '1c', tier: 'Must', component: 'LeavePhone', route: '/leave', state: 'Leave the phone; "Okay" dims the screen' },
  { id: 'door-first-walk', file: B, wrapper: '1d', tier: 'Must', component: 'Door', route: '/ (first walk variant)', state: 'First walk variant of the door screen' },
  { id: 'tell-sair', file: A, wrapper: '2a', tier: 'Must', component: 'TellSair', route: '/tell', state: 'Hold-to-talk idle' },
  { id: 'recording', file: A, wrapper: '2b', tier: 'Must', component: 'Recording', route: '/tell (recording)', state: 'Recording at 0:23' },
  { id: 'recording-warning', file: A, wrapper: '2b50', tier: 'Must', component: 'Recording', route: '/tell (recording, warning)', state: 'Recording at 0:50, "10 seconds left" countdown' },
  { id: 'writing', file: A, wrapper: '2c', tier: 'Must', component: 'Writing', route: '/writing', state: 'Saving / "Writing it down..." (logo trace animation)' },
  { id: 'saved', file: A, wrapper: '2d', tier: 'Must', component: 'Saved', route: '/saved', state: 'Entry ready with sure + maybe chips and "Mission done"' },
  { id: 'short-note', file: A, wrapper: '2e', tier: 'Must', component: 'Saved', route: '/saved (short note)', state: 'Short-note entry, "A quiet walk" chip' },
  { id: 'couldnt-hear', file: A, wrapper: '2f', tier: 'Should', component: 'CouldntHear', route: '/tell (couldn\'t hear)', state: 'Transcription failed, try again / skip' },
  { id: 'hub-sleeping', file: A, wrapper: '2g', tier: 'Should', component: 'HubSleeping', route: '/saved (hub asleep)', state: 'Computer asleep, note queued on phone' },
  { id: 'diary', file: F, wrapper: '3a', tier: 'Must', component: 'Diary', route: '/diary', state: 'Diary list, mobile, filter = all' },
  { id: 'diary-empty', file: F, wrapper: '3b', tier: 'Should', component: 'Diary', route: '/diary (empty)', state: 'Empty first use' },
  { id: 'diary-desktop', file: F, wrapper: '3c', tier: 'Should', component: 'Diary', route: '/diary (desktop)', state: 'Diary list, desktop 1280x800', viewport: { width: 1280, height: 800 } },
  { id: 'shortcut-siri', file: I, wrapper: '4a', tier: 'Should', component: '(Shortcuts mock, not routed)', route: 'n/a', state: 'Hey Siri, Sair', viewport: { width: 393, height: 852 } },
  { id: 'shortcut-bolo-talking', file: I, wrapper: '4b-1', tier: 'Should', component: '(Shortcuts mock, not routed)', route: 'n/a', state: 'Sair Bolo: talking', viewport: { width: 393, height: 852 } },
  { id: 'shortcut-bolo-saved', file: I, wrapper: '4b-2', tier: 'Should', component: '(Shortcuts mock, not routed)', route: 'n/a', state: 'Sair Bolo: saved', viewport: { width: 393, height: 852 } },
  { id: 'shortcut-notification', file: I, wrapper: '4c', tier: 'Should', component: '(Shortcuts mock, not routed)', route: 'n/a', state: 'ntfy-style notification on the phone', viewport: { width: 393, height: 852 } },
  { id: 'status-page', file: H, wrapper: '5a', tier: 'Should', component: 'Status', route: '/status', state: 'Owner status page, sample data', selector: '[id="5a"] + div', viewport: { width: 1280, height: 800 } },
];
const duskWrappers: Record<string, string> = { door: '1a-dusk', 'mission-playing': '1b-dusk', 'leave-phone': '1c-dusk', 'door-first-walk': '1d-dusk', 'tell-sair': '2a-dusk', recording: '2b-dusk', 'recording-warning': '2b50-dusk', writing: '2c-dusk', saved: '2d-dusk', 'short-note': '2e-dusk', 'couldnt-hear': '2f-dusk', 'hub-sleeping': '2g-dusk', diary: '3a-dusk', 'diary-empty': '3b-dusk', 'diary-desktop': '3c-dusk' };
export const META: Meta[] = [];
for (const m of light) META.push({ ...m, theme: 'light', viewport: m.viewport ?? MOB });
for (const m of light) if (duskWrappers[m.id]) META.push({ ...m, id: m.id + '-dusk', wrapper: duskWrappers[m.id], theme: 'dark', tier: 'Could', state: m.state + ' (dusk)', viewport: m.viewport ?? MOB });
