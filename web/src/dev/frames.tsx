// Registry of fixture frames for the visual harness: frame id (inventory.json) -> rendered screen with its fixture.
// To add a frame: add its fixture in src/fixtures, add a line here, then add it to tests/visual/built-frames.ts.
import type { ReactNode } from 'react';
import { door, doorFirstWalk, missionPlaying, leavePhone, tellSair, recording, recordingWarning } from '../fixtures/index.ts';
import { Door } from '../screens/Door/index.ts';
import { DoorFirstWalk } from '../screens/DoorFirstWalk/index.ts';
import { MissionPlaying } from '../screens/MissionPlaying/index.ts';
import { LeavePhone } from '../screens/LeavePhone/index.ts';
import { TellSair } from '../screens/TellSair/index.ts';
import { Recording } from '../screens/Recording/index.ts';
import { writing, saved, shortNote, diary } from '../fixtures/index.ts';
import { Writing } from '../screens/Writing/index.ts';
import { Saved } from '../screens/Saved/index.ts';
import { ShortNote } from '../screens/ShortNote/index.ts';
import { Diary } from '../screens/Diary/index.ts';
import { couldntHear, hubSleeping, diaryEmpty } from '../fixtures/index.ts';
import { CouldntHear } from '../screens/CouldntHear/index.ts';
import { HubSleeping } from '../screens/HubSleeping/index.ts';
import { DiaryEmpty } from '../screens/DiaryEmpty/index.ts';

export const FRAMES: Record<string, () => ReactNode> = {
  door: () => <Door data={door} />,
  'door-first-walk': () => <DoorFirstWalk data={{ ...doorFirstWalk, note: doorFirstWalk.note! }} />,
  'mission-playing': () => <MissionPlaying data={missionPlaying} />,
  'leave-phone': () => <LeavePhone data={leavePhone} />,
  'tell-sair': () => <TellSair data={tellSair} />,
  recording: () => <Recording data={recording} />,
  'recording-warning': () => <Recording data={recordingWarning} animate={false} />,
  writing: () => <Writing data={writing} />,
  saved: () => <Saved data={saved} />,
  'short-note': () => <ShortNote data={shortNote} />,
  diary: () => <Diary data={diary} />,
  'couldnt-hear': () => <CouldntHear data={couldntHear} />,
  'hub-sleeping': () => <HubSleeping data={hubSleeping} />,
  'diary-empty': () => <DiaryEmpty data={diaryEmpty} />,
};
