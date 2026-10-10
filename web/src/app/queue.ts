// FR-J9: offline queue for recordings the hub could not take. IndexedDB only, no library. An entry exists only
// between a failed upload and the hub's 202; it is deleted the moment the hub acks. Audio never goes to localStorage.

const DB = 'sair-queue';
const STORE = 'pending';

export type Pending = { id: number; blob: Blob; seconds: number; createdAt: number };

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(new Error('idb_open'));
  });
}

async function run<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = () => reject(new Error('idb_tx'));
      tx.onabort = () => reject(new Error('idb_tx'));
    });
  } finally {
    db.close();
  }
}

export async function addPending(blob: Blob, seconds: number): Promise<void> {
  await run('readwrite', (s) => s.add({ blob, seconds, createdAt: Date.now() }));
}

export async function listPending(): Promise<Pending[]> {
  return run('readonly', (s) => s.getAll() as IDBRequest<Pending[]>);
}

export async function deletePending(id: number): Promise<void> {
  await run('readwrite', (s) => s.delete(id));
}
