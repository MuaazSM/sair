import { Saved } from '../Saved/index.ts';
import type { SavedData } from '../../fixtures/saved.ts';

/** Short note (design 2E): the Saved layout with no chips and "A quiet walk" instead of a mission result. */
export function ShortNote({ data }: { data: SavedData }) {
  return <Saved data={data} />;
}
