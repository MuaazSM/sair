// Fixture data copied from "After the Walk.dc.html" frame 2B (recording at 0:23).
export type RecordingData = {
  title: string;
  mission: string;
  time: string;
  /** 0..1 of the 60 s ring. */
  progress: number;
  holdLabel: string;
  hint: string;
  /** Present from 0:50 only. */
  countdown?: string;
};

export const recording: RecordingData = {
  title: 'Listening',
  mission: 'Today: the gulmohar on your lane',
  time: '0:23',
  progress: 1 - 410.7 / 666,
  holdLabel: 'Hold to talk',
  hint: "Let go when you're done.",
};
