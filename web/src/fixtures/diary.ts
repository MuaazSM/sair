// Fixture data from "Family Diary.dc.html" frame 3A (mobile, filter = all). Group names are fixture data (C10).
import type { DiaryEntry } from '../components/DiaryEntryCard.tsx';

export type DiaryData = {
  title: string;
  summary: string;
  filters: string[];
  activeFilter: number;
  entries: DiaryEntry[];
  footer: string;
};

export const diary: DiaryData = {
  title: 'Our walks',
  summary: '7 walks this month · 12 things noticed',
  filters: ['All', 'Mom', 'Dad'],
  activeFilter: 0,
  entries: [
    {
      date: 'Sat 10 Oct',
      group: 'Mom and Dad',
      mission: 'Three walks ago, the gulmohar on your lane had no flowers. Look at it again today.',
      entry: 'Mom and Dad walked down the lane past the temple. The gulmohar had a few red flowers again. Dad thinks he heard a koel, maybe near the school.',
      facts: [
        { kind: 'sure', text: 'gulmohar · red flowers' },
        { kind: 'maybe', text: 'koel near the school' },
      ],
    },
    {
      date: 'Fri 9 Oct',
      group: 'Dad',
      mission: 'Is the ginger cat still on the blue scooter? Look outside building 4.',
      entry: 'Dad walked alone; Mom had her sister over. No cat on the scooter tonight. The chai stall has moved to the other side of the temple gate. Same chaiwala, same price.',
      facts: [
        { kind: 'sure', text: 'chai stall moved' },
        { kind: 'sure', text: 'no cat tonight' },
      ],
    },
    {
      date: 'Thu 8 Oct',
      group: 'Mom and Dad',
      mission: 'Look up at the sky just before you turn back.',
      entry: 'The sky was pink and clear, the first evening without monsoon clouds. A breeze came in from the sea at the end of the lane. Mom said it smelled of salt and someone frying bhajiyas.',
      facts: [
        { kind: 'sure', text: 'clear pink sky' },
        { kind: 'sure', text: 'sea breeze' },
      ],
    },
    {
      date: 'Wed 7 Oct',
      group: 'Mom',
      mission: 'Notice one sound you usually don’t hear.',
      entry: 'Mom went for a short walk.',
      facts: [],
    },
    {
      date: 'Tue 6 Oct',
      group: 'Mom and Dad',
      mission: 'Count the cats you see. Only the ones that look back.',
      entry: 'The ginger cat was asleep on the blue scooter again, outside building 4. The crows were loud around the big peepal tree by the temple.',
      facts: [
        { kind: 'sure', text: 'ginger cat on scooter' },
        { kind: 'sure', text: 'crows at the peepal' },
      ],
      askedAgain: 'Asked about again on Fri 9 Oct',
    },
    {
      date: 'Sun 4 Oct',
      group: 'Mom and Dad',
      mission: 'Look at the gulmohar on your lane. What colour is it right now?',
      entry: 'The gulmohar near the temple is all green now, no flowers left. Dad says it will flower again before Diwali.',
      facts: [{ kind: 'sure', text: 'gulmohar · no flowers' }],
      askedAgain: 'Asked about again on Sat 10 Oct',
    },
  ],
  footer: "That's everything from October so far.",
};
