import type { MergedPayload } from "@/lib/types";

export function createExplorer(
  root: HTMLElement,
  data: MergedPayload,
  hooks?: { onSelect?: (names: string[]) => void },
): {
  toggleName: (name: string, withSubtree: boolean) => boolean;
  destroy: () => void;
};

export function contract(
  N: { i: number[]; e: string | null }[],
  ids: number[],
  key: (id: number) => number,
  excluded: (id: number) => boolean,
  grouping: boolean,
  expanded: Set<number>,
): {
  G: { ids: number[]; inp: Record<number, number[]>; kids: Record<number, number[]>; nOps: number };
  groupOf: Record<number, { top: number; members: number[]; vid: number; key: number }>;
  groupByVid: Record<number, { top: number; members: number[]; vid: number; key: number }>;
};
