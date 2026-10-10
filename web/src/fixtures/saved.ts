// Fixture data from "After the Walk.dc.html" frame 2D (Saved, entry ready).
export type SavedData = {
  heading: string;
  date: string;
  group: string;
  entry: string;
  /** Only on short notes: the walker's own words, quoted. */
  fromNote?: string;
  facts: { kind: 'sure' | 'maybe'; text: string }[];
  outcome: { kind: 'done' | 'quiet'; label: string };
  readLabel: string;
  doneLabel: string;
};

export const saved: SavedData = {
  heading: 'Got it. Saved.',
  date: 'Saturday, 10 October',
  group: 'Mom and Dad',
  entry: 'Mom and Dad walked down the lane past the temple. The gulmohar had a few red flowers again. Dad thinks he heard a koel, maybe near the school.',
  facts: [
    { kind: 'sure', text: 'gulmohar · red flowers' },
    { kind: 'maybe', text: 'koel' },
  ],
  outcome: { kind: 'done', label: 'Mission done' },
  readLabel: 'Read it to me',
  doneLabel: 'Done',
};
