// Fixture data from "Before the Walk.dc.html" frame 1B at its initial state (t = 7.6 s: all words shown,
// ring full, "Playing again in 2 seconds").
export type MissionPlayingData = {
  heading: string;
  words: { w: string; visible: boolean }[];
  progress: number;
  playLabel: string;
  pauseLabel: string;
};

const MISSION = 'Three walks ago, the gulmohar on your lane had no flowers. Look at it again today.';

export const missionPlaying: MissionPlayingData = {
  heading: "Today's walk",
  words: MISSION.split(' ').map((w) => ({ w, visible: true })),
  progress: 1,
  playLabel: 'Playing again in 2 seconds',
  pauseLabel: 'Pause',
};
