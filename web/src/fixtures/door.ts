// Fixture data copied from "Before the Walk.dc.html" frame 1A (strings identical to the design).
export type DoorData = {
  date: string;
  heading: string;
  card: { title: string; lineWidths: number[]; caption: string };
  /** Reassurance line with the leaf icon; first walk only. */
  note?: string;
  cta: string;
  link: string;
};

export const door: DoorData = {
  date: 'Saturday, 10 October',
  heading: 'Ready for your sair?',
  card: { title: "Today's walk", lineWidths: [100, 86, 58], caption: 'Listen first. The words come after.' },
  cta: "Hear today's walk",
  link: 'Back from your walk? Tell Sair',
};
