import type { MergedPayload } from "@/lib/types";

export function createExplorer(
  root: HTMLElement,
  data: MergedPayload,
  hooks?: { onSelect?: (names: string[]) => void },
): {
  toggleName: (name: string, withSubtree: boolean) => boolean;
  destroy: () => void;
};
