"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { createExplorer } from "./engine.js";
import type { MergedPayload } from "@/lib/types";
import "./explorer.css";

export type ExplorerHandle = { toggleName: (name: string, withSubtree: boolean) => boolean };

type Engine = ExplorerHandle & { destroy: () => void };

/** Markup the engine binds to (data-pa hooks); all behaviour lives in engine.js. */
export const Explorer = forwardRef<ExplorerHandle, {
  data: MergedPayload;
  onSelect?: (names: string[]) => void;
}>(function Explorer({ data, onSelect }, ref) {
  const root = useRef<HTMLDivElement>(null);
  const engine = useRef<Engine | null>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  useEffect(() => {
    if (!root.current) return;
    engine.current = createExplorer(root.current, data, {
      onSelect: (names: string[]) => onSelectRef.current?.(names),
    }) as Engine;
    return () => {
      engine.current?.destroy();
      engine.current = null;
    };
  }, [data]);

  useImperativeHandle(ref, () => ({
    toggleName: (name, withSubtree) => engine.current?.toggleName(name, withSubtree) ?? false,
  }), []);

  return (
    <div className="pa-explorer" ref={root}>
      <div className="pa-side">
        <h3>Pipelines <span data-pa="count" /></h3>
        <div className="pa-acts">
          <button data-pa="all">all</button>
          <button data-pa="none">none</button>
          <button data-pa="invert">invert</button>
          <button data-pa="best" title="root → best-scoring pipeline">best path</button>
          <button data-pa="roots">roots</button>
        </div>
        <input data-pa="filter" placeholder="filter by name…" />
        <div data-pa="list" />
      </div>
      <div className="pa-main">
        <div className="pa-tools">
          <label>colour: <select data-pa="mode" defaultValue="share">
            <option value="share">how widely shared</option>
            <option value="pipe">by pipeline</option>
            <option value="diff">diff vs parent</option>
          </select></label>
          <button data-pa="zin" title="zoom in">+</button>
          <button data-pa="zout" title="zoom out">−</button>
          <button data-pa="fit">reset view</button>
          <button data-pa="full" title="give the graph the whole window (Esc to leave)">full screen</button>
          <span className="muted">scroll to zoom, drag to pan, click an operation for its pipelines</span>
          <span data-pa="modehint" />
        </div>
        <div className="legend" data-pa="legend" />
        <div className="pa-canvas"><svg data-pa="svg" /></div>
        <p data-pa="stats" />
        <div data-pa="inspect" />
      </div>
    </div>
  );
});
