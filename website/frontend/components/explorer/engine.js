/* Client-side explorer for the merged operator DAG.
 *
 * Ported from tools/pipeline_analyzer/explorer.js (the static report's copy):
 * bound to one root element instead of page-global ids, the payload passed in
 * instead of read from an inline <script>, the search tree driven through
 * ``hooks.onSelect`` and the returned api, and every document-level listener
 * removed again by ``destroy()`` so React can mount and unmount it.
 *
 * The payload (pipeline_analyzer/merged.py) is the union of
 * every pipeline's operator DAG, nodes referenced by index, each carrying the
 * list of pipelines that contain it. Everything below is driven by which
 * pipelines are ticked -- the visible sub-DAG is re-laid-out and re-drawn on
 * every change, which is why the layout is done here rather than by graphviz at
 * generation time: a fixed graphviz layout of the full union would leave a
 * selection of three pipelines scattered across a page-sized canvas.
 *
 * Grouping: before layout, every connected region of operations carried by the
 * same ticked pipelines collapses into one box (see ``buildView``), which is
 * what keeps a union DAG of hundreds of operations legible. A box expands on
 * click; the inspector collapses it again.
 *
 * Layout is a small layered (Sugiyama) pipeline: longest-path layering, tightened
 * so every node sits just above its earliest child; dummy nodes for edges that
 * span more than one layer; barycenter ordering sweeps keeping the crossing-
 * minimal pass; then order-preserving coordinate relaxation.
 */
export function createExplorer(root, D, hooks) {
  "use strict";
  hooks = hooks || {};
  function $(k) { return root.querySelector('[data-pa="' + k + '"]'); }
  // Every listener goes through on(), so destroy() removes them all. The DOM
  // outlives the engine (React re-runs the effect on the same nodes -- twice in
  // dev's StrictMode), and a listener left behind doubles each click: two
  // full-screen handlers left the page stuck in the overlay.
  var listeners = new AbortController();
  function on(el, type, fn, opts) {
    el.addEventListener(type, fn, Object.assign({ signal: listeners.signal }, opts));
  }
  var P = D.pipelines, N = D.nodes;
  var NP = P.length;

  // ---- indexes -------------------------------------------------------------
  var nodesOf = P.map(function () { return []; });      // pipeline -> node ids
  N.forEach(function (n, i) {
    n.id = i;
    for (var k = 0; k < n.m.length; k++) nodesOf[n.m[k]].push(i);
  });
  var memberSet = N.map(function (n) {
    var s = new Set();
    n.m.forEach(function (p) { s.add(p); });
    return s;
  });
  var kidsOf = N.map(function () { return []; });        // node -> node ids
  N.forEach(function (n) {
    n.i.forEach(function (u) { kidsOf[u].push(n.id); });
  });

  var okIdx = [];
  P.forEach(function (p, i) { if (p.ok) okIdx.push(i); });

  // ---- state ---------------------------------------------------------------
  var sel = new Set();
  var mode = "share";
  var hover = -1;          // pipeline being hovered in the list
  var picked = -1;         // node id shown in the inspector
  var inspectOpen = null;  // remembered <details> state of the inspector
  var view = { k: 1, x: 0, y: 0 };
  var grouping = true;         // collapse same-membership regions
  var collapseEst = false;     // let estimators collapse into regions too
  var bottomUp = true;         // sources at the bottom, as query plans are drawn
  var expanded = new Set();    // top node id of every group the reader expanded
  var groupOf = {};            // visible node id -> its group (size >= 2), per draw
  var groupByVid = {};         // virtual item id -> group, per draw

  // ---- palette -------------------------------------------------------------
  var GRAY = ["#eceff3", "#9aa4b2"];
  var EST_BORDER = "#2563eb";
  var DIFF = {
    frontier: ["#b7f0c6", "#15a34a"],
    bubbled: ["#fde6b0", "#d09a1e"],
    shared: GRAY,
    removed: ["#f7c9c9", "#dc2626"]
  };
  // Identity colours handed out in selection order for "by pipeline" colouring.
  var IDENT = ["#2563eb", "#e8590c", "#0d9488", "#7c3aed", "#b45309",
               "#be185d", "#4d7c0f", "#0369a1", "#a16207", "#475569"];
  var identOf = {};      // pipeline -> colour, rebuilt on every selection change

  function mix(a, b, t) {
    function h(c) {
      return [parseInt(c.substr(1, 2), 16), parseInt(c.substr(3, 2), 16),
              parseInt(c.substr(5, 2), 16)];
    }
    var x = h(a), y = h(b), o = "#";
    for (var i = 0; i < 3; i++) {
      var v = Math.round(x[i] + (y[i] - x[i]) * t);
      o += (v < 16 ? "0" : "") + v.toString(16);
    }
    return o;
  }

  /* Share ramp: shared by every selected pipeline is the quiet default; the
   * fewer pipelines carry an operation, the hotter it reads, so what the run
   * actually varied is what lights up. */
  function shareColor(k, n) {
    if (n <= 1 || k >= n) return GRAY;
    // Gamma-corrected so "shared by most" stays quiet and only the operations a
    // handful of pipelines carry read as hot; a linear ramp turned a run whose
    // pipelines share little (the usual case) into a wall of orange.
    var t = Math.pow((n - k) / (n - 1), 1.6);
    return [mix("#f4f6f9", "#ff9f43", t), mix("#b9c2cd", "#c2410c", t)];
  }

  function identColors() {
    identOf = {};
    var i = 0;
    sel.forEach(function (pi) { identOf[pi] = IDENT[i++ % IDENT.length]; });
  }

  // ---- geometry ------------------------------------------------------------
  var CH = 6.35, LH = 13, PADX = 9, PADY = 6, WRAP = 26;
  var GAP = 20, DGAP = 9, RANKSEP = 46;

  function wrap(text) {
    var words = String(text).split(" "), lines = [], cur = "";
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      if (cur && cur.length + w.length + 1 > WRAP) { lines.push(cur); cur = w; }
      else cur = cur ? cur + " " + w : w;
    }
    if (cur) lines.push(cur);
    // A single unbroken token (a long column list) still has to be cut.
    var out = [];
    lines.forEach(function (l) {
      while (l.length > WRAP + 8) { out.push(l.slice(0, WRAP + 6)); l = l.slice(WRAP + 6); }
      out.push(l);
    });
    return out.length ? out : [""];
  }

  function measure(lines) {
    var mx = 0;
    lines.forEach(function (l) { mx = Math.max(mx, l.length); });
    return { w: Math.max(52, Math.round(mx * CH) + 2 * PADX),
             h: lines.length * LH + 2 * PADY };
  }

  /* Layered layout of the view graph ``G`` (``buildView``): ``G.ids`` in
   * topological order, ``G.inp`` / ``G.kids`` its edges, ``G.lines`` labels. */
  function layout(G) {
    var ids = G.ids, i, l;

    // 1. layering: longest path from a source, then tightened downward so each
    //    node sits directly above its earliest child (fewer, shorter edges).
    var layer = {};
    ids.forEach(function (id) {
      var mx = -1;
      G.inp[id].forEach(function (u) { mx = Math.max(mx, layer[u]); });
      layer[id] = mx + 1;
    });
    for (i = ids.length - 1; i >= 0; i--) {
      var id = ids[i], mn = Infinity;
      G.kids[id].forEach(function (v) { mn = Math.min(mn, layer[v]); });
      if (mn !== Infinity) layer[id] = mn - 1;
    }

    var nL = 0;
    ids.forEach(function (id) { nL = Math.max(nL, layer[id] + 1); });
    var L = [];
    for (l = 0; l < nL; l++) L.push([]);

    // 2. items: one per visible node, plus a dummy per layer an edge crosses.
    var items = [], itemOf = {};
    ids.forEach(function (id) {
      var lines = G.lines[id], m = measure(lines);
      var it = { idx: items.length, node: id, l: layer[id], w: m.w, h: m.h,
                 lines: lines, up: [], down: [] };
      items.push(it); itemOf[id] = it; L[it.l].push(it);
    });
    var edges = [], segs = [];
    ids.forEach(function (v) {
      G.inp[v].forEach(function (u) {
        var prev = itemOf[u], chain = [];
        for (var lv = layer[u] + 1; lv < layer[v]; lv++) {
          var d = { idx: items.length, node: -1, l: lv, w: 1, h: 1, up: [], down: [] };
          items.push(d); L[lv].push(d); chain.push(d);
          segs.push([prev, d]); prev = d;
        }
        segs.push([prev, itemOf[v]]);
        edges.push({ u: u, v: v, chain: chain });
      });
    });
    segs.forEach(function (s) { s[0].down.push(s[1]); s[1].up.push(s[0]); });

    // 3. ordering: barycenter sweeps, keeping the crossing-minimal pass.
    function reindex() {
      L.forEach(function (row) { row.forEach(function (it, k) { it.pos = k; }); });
    }
    function crossings() {
      var total = 0;
      for (var lv = 0; lv + 1 < nL; lv++) {
        var pairs = [];
        L[lv].forEach(function (a) {
          a.down.forEach(function (b) { pairs.push([a.pos, b.pos]); });
        });
        for (var x = 0; x < pairs.length; x++)
          for (var y = x + 1; y < pairs.length; y++)
            if ((pairs[x][0] - pairs[y][0]) * (pairs[x][1] - pairs[y][1]) < 0) total++;
      }
      return total;
    }
    function bary(it, dir) {
      var ns = dir > 0 ? it.up : it.down;
      if (!ns.length) return it.pos;
      var s = 0;
      ns.forEach(function (n) { s += n.pos; });
      return s / ns.length;
    }
    reindex();
    var best = L.map(function (row) { return row.slice(); }), bestX = crossings();
    for (var pass = 0; pass < 10; pass++) {
      var dir = pass % 2 === 0 ? 1 : -1;
      var order = [];
      for (l = 0; l < nL; l++) order.push(dir > 0 ? l : nL - 1 - l);
      order.forEach(function (lv) {
        var row = L[lv];
        var keys = row.map(function (it) { return [bary(it, dir), it.pos, it]; });
        keys.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
        L[lv] = keys.map(function (k) { return k[2]; });
        L[lv].forEach(function (it, k) { it.pos = k; });
      });
      var c = crossings();
      if (c < bestX) { bestX = c; best = L.map(function (row) { return row.slice(); }); }
    }
    L = best; reindex();

    // 4. x coordinates: pack in order, then relax toward neighbour medians while
    //    keeping the order and a minimum gap.
    function sep(a, b) {
      return (a.w + b.w) / 2 + (a.node < 0 || b.node < 0 ? DGAP : GAP);
    }
    L.forEach(function (row) {
      var x = 0;
      row.forEach(function (it, k) {
        if (k) x += sep(row[k - 1], it);
        it.x = x;
      });
    });
    function pack(row, want) {
      var n = row.length, a = new Array(n), b = new Array(n), k;
      for (k = 0; k < n; k++)
        a[k] = k ? Math.max(want[k], a[k - 1] + sep(row[k - 1], row[k])) : want[k];
      for (k = n - 1; k >= 0; k--)
        b[k] = k < n - 1 ? Math.min(want[k], b[k + 1] - sep(row[k], row[k + 1])) : want[k];
      for (k = 0; k < n; k++) row[k].x = (a[k] + b[k]) / 2;
      for (k = 1; k < n; k++)                       // restore the invariant
        row[k].x = Math.max(row[k].x, row[k - 1].x + sep(row[k - 1], row[k]));
    }
    for (pass = 0; pass < 16; pass++) {
      var dn = pass % 2 === 0;
      for (var q = 0; q < nL; q++) {
        l = dn ? q : nL - 1 - q;
        var row = L[l];
        var want = row.map(function (it) {
          var ns = dn ? it.up : it.down;
          if (!ns.length) return it.x;
          var xs = ns.map(function (n) { return n.x; }).sort(function (a, b) { return a - b; });
          var m = xs.length >> 1;
          return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
        });
        pack(row, want);
      }
    }

    // 5. y coordinates and bounding box.
    var y = 0;
    L.forEach(function (row) {
      var h = 0;
      row.forEach(function (it) { h = Math.max(h, it.h); });
      row.forEach(function (it) { it.y = y + h / 2; });
      y += h + RANKSEP;
    });
    var minX = Infinity, maxX = -Infinity;
    items.forEach(function (it) {
      minX = Math.min(minX, it.x - it.w / 2);
      maxX = Math.max(maxX, it.x + it.w / 2);
    });
    if (!items.length) { minX = 0; maxX = 0; }
    var pad = 16;
    items.forEach(function (it) { it.x += pad - minX; it.y += pad; });

    return { items: items, itemOf: itemOf, edges: edges, layers: L,
             w: maxX - minX + 2 * pad, h: Math.max(y - RANKSEP, 0) + 2 * pad,
             crossings: bestX };
  }

  // ---- what is visible, and how it is coloured -----------------------------
  function diffPartner() {
    /* Diff colouring needs exactly one selected pipeline with an extracted
     * parent; otherwise the mode falls back to sharing. */
    if (sel.size !== 1) return null;
    var c = sel.values().next().value;
    var p = P[c].p;
    if (p === null || p === undefined || !P[p].ok) return null;
    return { child: c, parent: p };
  }

  function visibleIds() {
    var pair = mode === "diff" ? diffPartner() : null;
    var want = new Set();
    if (pair) {
      nodesOf[pair.child].forEach(function (id) { want.add(id); });
      nodesOf[pair.parent].forEach(function (id) { want.add(id); });
    } else {
      sel.forEach(function (pi) {
        nodesOf[pi].forEach(function (id) { want.add(id); });
      });
    }
    var ids = [];
    for (var i = 0; i < N.length; i++) if (want.has(i)) ids.push(i);   // topological
    return { ids: ids, pair: pair };
  }

  /* Pipelines carrying an operation, reduced to a key comparable along edges.
   * In the merged DAG an operation's inputs belong to every pipeline the
   * operation belongs to (they are part of its signature), so along an edge the
   * set can only shrink: equal counts across an edge mean equal sets. */
  function memberKey(id, pair) {
    if (pair) {
      return (memberSet[id].has(pair.child) ? 1 : 0) + (memberSet[id].has(pair.parent) ? 2 : 0);
    }
    var k = 0;
    sel.forEach(function (pi) { if (memberSet[id].has(pi)) k++; });
    return k;
  }

  /* The graph layout() draws: ``contract`` (below) plus labels. */
  function buildView(ids, pair) {
    var r = contract(N, ids, function (id) { return memberKey(id, pair); },
                     function (id) { return !collapseEst && !!N[id].e; },
                     grouping, expanded);
    groupOf = r.groupOf; groupByVid = r.groupByVid;
    var G = r.G;
    G.ids.forEach(function (it) {
      G.lines[it] = it < N.length ? wrap(N[it].l) : groupLines(groupByVid[it]);
    });
    return G;
  }

  function groupLines(g) {
    var types = {};
    g.members.forEach(function (id) { types[N[id].t] = (types[N[id].t] || 0) + 1; });
    var top = Object.keys(types).sort(function (a, b) { return types[b] - types[a]; })
      .slice(0, 2).map(function (t) { return t.replace(/Op$/, "") + " ×" + types[t]; });
    return ["⊞ " + g.members.length + " operations"].concat(top);
  }

  /* The node a view item stands for: itself, or its group's top node. */
  function repOf(item) { return item < N.length ? item : groupByVid[item].top; }

  function colorForItem(item, pair) {
    if (item < N.length) return colorFor(item, pair);
    var g = groupByVid[item];
    // child-only region in diff mode: "new" if any member is a genuinely new op
    if (pair && g.key === 1) {
      var fresh = g.members.some(function (id) {
        return N[id].i.every(function (u) { return memberSet[u].has(pair.parent); });
      });
      return fresh ? DIFF.frontier : DIFF.bubbled;
    }
    return colorFor(g.top, pair);
  }

  function colorFor(id, pair) {
    var n = N[id];
    if (pair) {
      var inC = memberSet[id].has(pair.child), inP = memberSet[id].has(pair.parent);
      if (inC && inP) return DIFF.shared;
      if (!inC) return DIFF.removed;
      var frontier = n.i.every(function (u) { return memberSet[u].has(pair.parent); });
      return frontier ? DIFF.frontier : DIFF.bubbled;
    }
    var k = 0, owner = -1;
    sel.forEach(function (pi) {
      if (memberSet[id].has(pi)) { k++; owner = pi; }
    });
    if (mode === "pipe") {
      if (k !== 1) return GRAY;
      var c = identOf[owner] || "#64748b";
      return [mix("#ffffff", c, 0.2), c];
    }
    return shareColor(k, sel.size);
  }

  // ---- rendering -----------------------------------------------------------
  var svg = $("svg");
  var vp = null;

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function edgePath(pts) {
    var d = "M" + pts[0].x.toFixed(1) + "," + pts[0].y.toFixed(1);
    for (var i = 1; i < pts.length; i++) {
      var a = pts[i - 1], b = pts[i], my = (a.y + b.y) / 2;
      d += "C" + a.x.toFixed(1) + "," + my.toFixed(1) + " " +
           b.x.toFixed(1) + "," + my.toFixed(1) + " " +
           b.x.toFixed(1) + "," + b.y.toFixed(1);
    }
    return d;
  }

  function draw() {
    identColors();
    var v = visibleIds(), ids = v.ids, pair = v.pair;
    var t0 = performance.now();
    var G = buildView(ids, pair);
    var laid = layout(G);
    // layout() puts sources on top; mirror it for the query-plan orientation
    if (bottomUp) laid.items.forEach(function (it) { it.y = laid.h - it.y; });
    var ms = performance.now() - t0;

    var parts = [];
    if (!ids.length) {
      svg.innerHTML = '<text x="50%" y="52%" text-anchor="middle" ' +
        'fill="#5b6472" font-size="13">tick a pipeline to draw its operations' +
        "</text>";
      svg.setAttribute("viewBox", "0 0 400 120");
      svg.parentNode.style.height = fsActive() ? "" : "160px";
      vp = null;
      $("stats").textContent = "nothing selected";
      $("legend").innerHTML = "";
      $("modehint").textContent = "";
      return;
    }
    parts.push('<defs><marker id="pa-arrow" viewBox="0 0 8 8" refX="7" refY="4" ' +
      'markerWidth="6" markerHeight="6" orient="auto-start-reverse">' +
      '<path d="M0,1 L7,4 L0,7 z" fill="#98a2b3"/></marker></defs>');
    parts.push('<g class="pa-vp">');

    laid.edges.forEach(function (e) {
      var a = laid.itemOf[e.u], b = laid.itemOf[e.v];
      // edges leave the side facing the consumer: the top when drawn bottom-up
      var s = bottomUp ? -1 : 1;
      var pts = [{ x: a.x, y: a.y + s * a.h / 2 }];
      e.chain.forEach(function (d) { pts.push({ x: d.x, y: d.y }); });
      pts.push({ x: b.x, y: b.y - s * b.h / 2 });
      var cls = "pa-e";
      if (pair && !memberSet[repOf(e.v)].has(pair.child)) cls += " gone";
      parts.push('<path class="' + cls + '" d="' + edgePath(pts) +
                 '" marker-end="url(#pa-arrow)"/>');
    });

    laid.items.forEach(function (it) {
      if (it.node < 0) return;
      if (it.node >= N.length) { parts.push(groupSvg(it, pair)); return; }
      var n = N[it.node], c = colorFor(it.node, pair);
      var stroke = n.e ? EST_BORDER : c[1];
      var x = it.x - it.w / 2, y = it.y - it.h / 2;
      var tx = it.x, ty = y + PADY + 10;
      var tspans = it.lines.map(function (l, k) {
        return '<tspan x="' + tx.toFixed(1) + '" dy="' + (k ? LH : 0) + '">' +
               esc(l) + "</tspan>";
      }).join("");
      var k = 0;
      sel.forEach(function (pi) { if (memberSet[it.node].has(pi)) k++; });
      var tip = n.l + "\n" + n.t + " · in " + k + " of " + sel.size +
                " selected (" + n.m.length + " of " + NP + " total)";
      parts.push('<g class="pa-n" data-id="' + it.node + '"><title>' + esc(tip) +
        '</title><rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' +
        it.w + '" height="' + it.h + '" rx="5" fill="' + c[0] + '" stroke="' +
        stroke + '" stroke-width="' + (n.e ? 1.8 : 1.3) + '"/>' +
        '<text x="' + tx.toFixed(1) + '" y="' + ty.toFixed(1) + '">' + tspans +
        "</text></g>");
    });
    parts.push("</g>");

    svg.innerHTML = parts.join("");
    svg.setAttribute("viewBox", "0 0 " + Math.ceil(laid.w) + " " + Math.ceil(laid.h));
    // A union DAG is often far wider than it is tall; letting the canvas keep a
    // fixed height would fit the graph into a thin band with dead space above
    // and below it.
    var box = svg.parentNode, avail = box.clientWidth || 800;
    box.style.height = fsActive() ? "" : Math.round(Math.min(760, Math.max(380,
      laid.h * avail / Math.max(laid.w, 1) + 24))) + "px";
    vp = svg.querySelector(".pa-vp");
    applyView();
    applyHover();
    applyPicked();

    var shared = 0, unique = 0;
    ids.forEach(function (id) {
      var k = 0;
      sel.forEach(function (pi) { if (memberSet[id].has(pi)) k++; });
      if (k === sel.size && sel.size > 1) shared++;
      if (k === 1 && sel.size > 1) unique++;
    });
    var nGroups = G.ids.filter(function (it) { return it >= N.length; }).length;
    var bits = [ids.length + " operations" + (nGroups ?
      " in " + G.ids.length + " boxes (" + nGroups + " collapsed group" +
      (nGroups > 1 ? "s" : "") + ")" : "")];
    if (sel.size > 1) {
      bits.push(shared + " in all " + sel.size);
      bits.push(unique + " in only one");
    }
    bits.push(laid.crossings + " edge crossings · laid out in " + ms.toFixed(0) + " ms");
    $("stats").textContent = bits.join(" · ");
    $("legend").innerHTML = legendHtml(pair);
    var hint = "";
    if (mode === "diff" && !pair)
      hint = "diff needs exactly one pipeline ticked, with an analyzed parent — showing sharing instead";
    else if (mode === "pipe" && sel.size > IDENT.length)
      hint = sel.size + " ticked, so identity colours repeat every " + IDENT.length;
    $("modehint").textContent = hint;
  }

  function groupSvg(it, pair) {
    var g = groupByVid[it.node], c = colorForItem(it.node, pair);
    var x = it.x - it.w / 2, y = it.y - it.h / 2;
    var tx = it.x, ty = y + PADY + 10;
    var tspans = it.lines.map(function (l, k) {
      return '<tspan x="' + tx.toFixed(1) + '" dy="' + (k ? LH : 0) + '"' +
             (k ? "" : ' font-weight="600"') + ">" + esc(l) + "</tspan>";
    }).join("");
    var k = pair ? null : memberKey(g.top, null);
    var tip = g.members.length + " operations, collapsed" +
      (k === null ? "" : " · in " + k + " of " + sel.size + " selected") +
      "\nclick to expand";
    function rect(dx, dy) {
      return '<rect x="' + (x + dx).toFixed(1) + '" y="' + (y + dy).toFixed(1) +
        '" width="' + it.w + '" height="' + it.h + '" rx="6" fill="' + c[0] +
        '" stroke="' + c[1] + '" stroke-width="1.4"/>';
    }
    // a second, offset rect reads as a stack: "there is more in here"
    return '<g class="pa-g" data-group="' + g.top + '"><title>' + esc(tip) + "</title>" +
      rect(3, 3) + rect(0, 0) +
      '<text x="' + tx.toFixed(1) + '" y="' + ty.toFixed(1) + '">' + tspans + "</text></g>";
  }

  function legendHtml(pair) {
    function sw(c, label) {
      return '<span><span class="sw" style="background:' + c[0] + ";border-color:" +
        c[1] + '"></span>' + esc(label) + "</span>";
    }
    var out = [];
    if (pair) {
      out.push(sw(DIFF.frontier, "new operation"));
      out.push(sw(DIFF.bubbled, "ancestor shifted by a change below it"));
      out.push(sw(DIFF.shared, "unchanged"));
      out.push(sw(DIFF.removed, "only in the parent (removed)"));
    } else if (mode === "pipe" && sel.size) {
      out.push(sw(GRAY, "in 2 or more of the ticked pipelines"));
      if (sel.size <= IDENT.length) {
        sel.forEach(function (pi) {
          var c = identOf[pi];
          out.push(sw([mix("#ffffff", c, 0.2), c], "only in " + P[pi].n));
        });
      } else {
        out.push('<span class="muted">one colour per pipeline &#8212; see the ' +
                 "dots in the list (they repeat past " + IDENT.length + ")</span>");
      }
    } else if (sel.size > 1) {
      var n = sel.size;
      out.push(sw(GRAY, "in all " + n + " selected"));
      if (n > 2) out.push(sw(shareColor(Math.max(2, Math.round(n / 2)), n), "in about half"));
      out.push(sw(shareColor(1, n), "in only one"));
    } else {
      out.push(sw(GRAY, "operation"));
    }
    out.push('<span><span class="sw" style="background:#fff;border-color:' +
             EST_BORDER + '"></span>estimator</span>');
    if (grouping) {
      out.push('<span><span class="sw sw-stack"></span>collapsed group (click to expand)</span>');
    }
    return out.join("");
  }

  // ---- pan / zoom ----------------------------------------------------------
  function applyView() {
    if (vp) vp.setAttribute("transform",
      "translate(" + view.x + "," + view.y + ") scale(" + view.k + ")");
  }
  function fit() { view = { k: 1, x: 0, y: 0 }; applyView(); }

  function zoomBy(f) {
    var vb = svg.viewBox.baseVal;
    var cx = vb.width / 2, cy = vb.height / 2;
    var k = Math.min(6, Math.max(0.15, view.k * f));
    f = k / view.k;
    view.x = cx - f * (cx - view.x);
    view.y = cy - f * (cy - view.y);
    view.k = k;
    applyView();
  }

  // ---- full screen ---------------------------------------------------------
  var explorer = root;
  var fsBtn = $("full");

  function fsActive() {
    return document.fullscreenElement === explorer ||
           explorer.classList.contains("pa-fs");
  }

  function fsSync() {
    fsBtn.textContent = fsActive() ? "exit full screen" : "full screen";
    draw();     // the canvas sizes differently in and out of full screen
  }

  function fsOverlay(on) {               // fallback when the API is unavailable
    explorer.classList.toggle("pa-fs", on);
    // The overlay is only fixed-position, so the page behind it would still
    // scroll under the wheel; native full screen needs no such help.
    document.body.style.overflow = on ? "hidden" : "";
    fsSync();
  }

  on(fsBtn, "click", function () {
    if (fsActive()) {
      if (document.fullscreenElement === explorer) document.exitFullscreen();
      else fsOverlay(false);
      return;
    }
    var req;
    try {
      req = explorer.requestFullscreen && explorer.requestFullscreen();
    } catch (err) {
      req = null;
    }
    if (req && req.then) req.then(function () {}, function () { fsOverlay(true); });
    else if (!req) fsOverlay(true);
  });

  function onKey(e) {
    if (e.key === "Escape" && explorer.classList.contains("pa-fs")) fsOverlay(false);
  }
  on(document, "fullscreenchange", fsSync);
  on(document, "keydown", onKey);

  on(svg, "wheel", function (e) {
    e.preventDefault();
    var r = svg.getBoundingClientRect();
    var vb = svg.viewBox.baseVal;
    var s = vb.width / r.width || 1;
    var mx = (e.clientX - r.left) * s, my = (e.clientY - r.top) * s;
    var f = Math.exp(-e.deltaY * 0.0015);
    var k = Math.min(6, Math.max(0.15, view.k * f));
    f = k / view.k;
    view.x = mx - f * (mx - view.x);
    view.y = my - f * (my - view.y);
    view.k = k;
    applyView();
  }, { passive: false });

  var drag = null;
  on(svg, "pointerdown", function (e) {
    // The element under the pointer has to be remembered here: capturing the
    // pointer on the svg retargets every later event (pointerup included) to
    // the svg itself, so ``e.target`` there is never the node that was hit.
    drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false,
             hit: e.target.closest ? e.target.closest(".pa-n, .pa-g") : null };
    svg.setPointerCapture(e.pointerId);
  });
  on(svg, "pointermove", function (e) {
    if (!drag) return;
    var r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    var s = vb.width / r.width || 1;
    var dx = (e.clientX - drag.x) * s, dy = (e.clientY - drag.y) * s;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    view.x = drag.vx + dx; view.y = drag.vy + dy;
    applyView();
  });
  on(svg, "pointerup", function () {
    var d = drag;
    drag = null;
    if (!d || d.moved) return;
    if (d.hit && d.hit.hasAttribute("data-group")) {
      expanded.add(+d.hit.getAttribute("data-group"));
      draw();
      return;
    }
    pick(d.hit ? +d.hit.getAttribute("data-id") : -1);
  });

  // ---- inspector -----------------------------------------------------------
  function pick(id) {
    picked = id;
    var box = $("inspect");
    if (id < 0) {
      inspectOpen = null;
      box.innerHTML = '<p class="muted">Click an operation in the graph to see ' +
        "which pipelines share it.</p>";
      applyPicked();
      return;
    }
    var n = N[id];
    var k = 0;
    sel.forEach(function (pi) { if (memberSet[id].has(pi)) k++; });
    var rows = n.m.map(function (pi) {
      var p = P[pi];
      return '<label class="pa-chip' + (sel.has(pi) ? " on" : "") + '">' +
        '<input type="checkbox" data-pipe="' + pi + '"' + (sel.has(pi) ? " checked" : "") +
        '><span class="dot" style="background:' +
        ((mode === "pipe" && identOf[pi]) || p.c) + '"></span>' +
        '<a href="#step-' + esc(p.n) + '">' + esc(p.n) + "</a></label>";
    }).join("");
    // Long member lists start collapsed -- with 68 pipelines the open list is
    // most of the panel -- but a state the reader chose survives the re-render
    // that ticking a chip in it triggers.
    var want = inspectOpen === null ? n.m.length <= 8 : inspectOpen;
    var open = want ? " open" : "";
    box.innerHTML =
      '<div class="pa-inspect-head"><code>' + esc(n.l) + "</code>" +
      '<span class="pill">' + esc(n.t) + "</span>" +
      (n.e ? '<span class="pill">' + esc(n.e) + "</span>" : "") + "</div>" +
      '<p class="muted">in <b>' + k + "</b> of " + sel.size + " ticked · <b>" +
      n.m.length + "</b> of " + NP + " pipelines overall</p>" +
      "<details" + open + '><summary>pipelines containing this operation (' +
      n.m.length + ")</summary><div class=\"pa-chips\">" + rows + "</div></details>";
    var g = groupOf[id];
    if (g && expanded.has(g.top)) {
      box.innerHTML += '<p><button data-collapse="' + g.top + '">collapse its group (' +
        g.members.length + " operations)</button></p>";
      on(box.querySelector("button[data-collapse]"), "click", function () {
        expanded.delete(g.top);
        draw();
        pick(id);
      });
    }
    box.querySelectorAll("input[data-pipe]").forEach(function (cb) {
      on(cb, "change", function () {
        toggle(+cb.getAttribute("data-pipe"), cb.checked);
      });
    });
    var det = box.querySelector("details");
    if (det) on(det, "toggle", function () { inspectOpen = det.open; });
    // A chip's link jumps to that pipeline's section; without this the click
    // would bubble to the surrounding label and tick the pipeline as well.
    box.querySelectorAll(".pa-chip a").forEach(function (a) {
      on(a, "click", function (e) { e.stopPropagation(); });
    });
    applyPicked();
  }

  function applyPicked() {
    svg.querySelectorAll(".pa-n.picked").forEach(function (g) {
      g.classList.remove("picked");
    });
    svg.querySelectorAll(".pa-g.picked").forEach(function (g) {
      g.classList.remove("picked");
    });
    if (picked < 0) return;
    var g = elementFor(picked);
    if (g) g.classList.add("picked");
  }

  /* The drawn element standing for node ``id``: its own box, or its
   * collapsed group's. */
  function elementFor(id) {
    var g = groupOf[id];
    if (g && !expanded.has(g.top)) return svg.querySelector('.pa-g[data-group="' + g.top + '"]');
    return svg.querySelector('.pa-n[data-id="' + id + '"]');
  }

  function applyHover() {
    svg.querySelectorAll(".pa-n.hl, .pa-g.hl").forEach(function (g) { g.classList.remove("hl"); });
    if (hover < 0) return;
    nodesOf[hover].forEach(function (id) {
      var g = elementFor(id);
      if (g) g.classList.add("hl");
    });
  }

  // ---- pipeline list -------------------------------------------------------
  var list = $("list");

  function rowHtml(pi) {
    var p = P[pi];
    var score = p.s === null || p.s === undefined ? "—" : (+p.s).toFixed(5);
    var up = p.up;
    var cls = up === null || up === undefined ? "flat" : up > 0.0005 ? "up"
            : up < -0.0005 ? "down" : "flat";
    var d = p.d === null || p.d === undefined ? "" :
      (p.d >= 0 ? "+" : "") + (+p.d).toFixed(4);
    return '<label class="pa-row" data-pipe="' + pi + '" title="' +
      esc(p.desc || p.n) + '">' +
      '<input type="checkbox" data-pipe="' + pi + '"' + (sel.has(pi) ? " checked" : "") +
      (p.ok ? "" : " disabled") + ">" +
      '<span class="dot" style="background:' + p.c + '"></span>' +
      '<span class="nm">' + esc(p.n) + (p.ok ? "" : ' <span class="err">✗</span>') +
      "</span>" +
      '<span class="sc">' + score + "</span>" +
      '<span class="dl ' + cls + '">' + d + "</span>" +
      '<span class="ops">' + p.ops + "</span>" +
      '<a class="jump" href="#step-' + esc(p.n) + '" title="jump to details">↗</a>' +
      "</label>";
  }

  function buildList() {
    var groups = [], byPhase = {};
    P.forEach(function (p, i) {
      var key = p.ph || "";
      if (!(key in byPhase)) { byPhase[key] = []; groups.push(key); }
      byPhase[key].push(i);
    });
    var html = "";
    groups.forEach(function (g) {
      if (groups.length > 1 || g) {
        html += '<div class="pa-group"><span class="dot" style="background:' +
          (P[byPhase[g][0]].c) + '"></span>' + esc(g || "pipelines") +
          ' <span class="muted">(' + byPhase[g].length + ")</span>" +
          '<button class="mini" data-phase="' + esc(g) + '" data-on="1">all</button>' +
          '<button class="mini" data-phase="' + esc(g) + '" data-on="0">none</button>' +
          "</div>";
      }
      html += byPhase[g].map(rowHtml).join("");
    });
    list.innerHTML = html;
    list.querySelectorAll("input[data-pipe]").forEach(function (cb) {
      on(cb, "change", function () {
        toggle(+cb.getAttribute("data-pipe"), cb.checked);
      });
    });
    list.querySelectorAll(".pa-row").forEach(function (row) {
      var pi = +row.getAttribute("data-pipe");
      on(row, "mouseenter", function () { hover = pi; applyHover(); });
      on(row, "mouseleave", function () { hover = -1; applyHover(); });
    });
    list.querySelectorAll(".pa-row .jump").forEach(function (a) {
      on(a, "click", function (e) { e.stopPropagation(); });
    });
    list.querySelectorAll("button[data-phase]").forEach(function (b) {
      on(b, "click", function (e) {
        e.preventDefault();
        var g = b.getAttribute("data-phase"), on = b.getAttribute("data-on") === "1";
        P.forEach(function (p, i) {
          if ((p.ph || "") === g && p.ok) { if (on) sel.add(i); else sel.delete(i); }
        });
        refresh();
      });
    });
  }

  function syncList() {
    list.querySelectorAll("input[data-pipe]").forEach(function (cb) {
      cb.checked = sel.has(+cb.getAttribute("data-pipe"));
    });
    list.querySelectorAll(".pa-row").forEach(function (row) {
      var pi = +row.getAttribute("data-pipe");
      var dot = row.querySelector(".dot");
      // In identity mode the dot is the graph's key, so it has to show the
      // colour that pipeline's own operations are drawn in.
      dot.style.background = (mode === "pipe" && identOf[pi]) || P[pi].c;
    });
    $("count").textContent =
      sel.size + " of " + NP + " pipelines";
  }

  function syncTree() {
    if (hooks.onSelect) {
      var names = [];
      sel.forEach(function (pi) { names.push(P[pi].n); });
      hooks.onSelect(names);
    }
  }

  function toggle(pi, on) {
    if (on === undefined) on = !sel.has(pi);
    if (!P[pi].ok) return;
    if (on) sel.add(pi); else sel.delete(pi);
    refresh();
  }

  function setSel(ids) {
    sel = new Set(ids.filter(function (i) { return P[i].ok; }));
    refresh();
  }

  function pathToRoot(pi) {
    var out = [], seen = new Set();
    while (pi !== null && pi !== undefined && !seen.has(pi)) {
      seen.add(pi); out.push(pi); pi = P[pi].p;
    }
    return out;
  }

  function bestPipeline() {
    var best = -1;
    okIdx.forEach(function (i) {
      if (P[i].s === null || P[i].s === undefined) return;
      if (best < 0) { best = i; return; }
      // Orientation is baked into ``up`` per node, so rank by walking the run:
      // the reported best is simply the extreme score in the winning direction.
      var better = D.lowerIsBetter ? P[i].s < P[best].s : P[i].s > P[best].s;
      if (better) best = i;
    });
    return best;
  }

  function refresh() {
    syncList(); syncTree(); draw();
    if (picked >= 0) pick(picked);
  }

  // ---- wiring --------------------------------------------------------------
  buildList();

  on($("all"), "click", function () {
    setSel(okIdx.slice());
  });
  on($("none"), "click", function () {
    setSel([]);
  });
  on($("invert"), "click", function () {
    setSel(okIdx.filter(function (i) { return !sel.has(i); }));
  });
  on($("best"), "click", function () {
    var b = bestPipeline();
    setSel(b < 0 ? okIdx.slice(0, 1) : pathToRoot(b));
  });
  on($("roots"), "click", function () {
    setSel(okIdx.filter(function (i) {
      return P[i].p === null || P[i].p === undefined;
    }));
  });
  on($("fit"), "click", fit);
  on($("zin"), "click", function () { zoomBy(1.4); });
  on($("zout"), "click", function () { zoomBy(1 / 1.4); });
  on($("mode"), "change", function (e) {
    mode = e.target.value;
    identColors();
    syncList();
    draw();
  });
  on($("expand"), "click", function () {
    Object.keys(groupByVid).forEach(function (v) { expanded.add(groupByVid[v].top); });
    draw();
  });
  on($("collapse"), "click", function () {
    expanded.clear();
    draw();
  });
  on($("grouping"), "change", function (e) {
    grouping = e.target.checked;
    $("collapse-est").disabled = !grouping;
    draw();
  });
  on($("sources-top"), "change", function (e) {
    bottomUp = !e.target.checked;
    draw();
  });
  on($("collapse-est"), "change", function (e) {
    collapseEst = e.target.checked;
    draw();
  });
  on($("filter"), "input", function (e) {
    var q = e.target.value.toLowerCase();
    list.querySelectorAll(".pa-row").forEach(function (row) {
      var pi = +row.getAttribute("data-pipe");
      row.style.display = !q || P[pi].n.toLowerCase().indexOf(q) >= 0 ? "" : "none";
    });
  });

  // Clicking a node of the lineage tree ticks that pipeline; shift-click takes
  // its whole subtree, which is how you ask "what did this branch explore?".
  var kidsOfPipe = P.map(function () { return []; });
  P.forEach(function (p, i) {
    if (p.p !== null && p.p !== undefined) kidsOfPipe[p.p].push(i);
  });
  function subtree(pi) {
    var out = [], stack = [pi];
    while (stack.length) {
      var x = stack.pop(); out.push(x);
      kidsOfPipe[x].forEach(function (c) { stack.push(c); });
    }
    return out;
  }
  // Default view: the winning path from root to the best-scoring pipeline --
  // small enough to read, and the one thread through the run that mattered.
  var b = bestPipeline();
  setSel(b < 0 ? okIdx.slice(0, Math.min(3, okIdx.length)) : pathToRoot(b));
  pick(-1);

  var indexOf = {};
  P.forEach(function (p, i) { indexOf[p.n] = i; });
  return {
    /** Tick/untick one pipeline by name; with ``withSubtree`` its whole subtree
     *  follows the pipeline's new state. Returns false for an unknown name. */
    toggleName: function (name, withSubtree) {
      var pi = indexOf[name];
      if (pi === undefined) return false;
      if (!withSubtree) { toggle(pi); return true; }
      var on = !sel.has(pi);
      subtree(pi).forEach(function (i) {
        if (!P[i].ok) return;
        if (on) sel.add(i); else sel.delete(i);
      });
      refresh();
      return true;
    },
    destroy: function () {
      listeners.abort();
      if (fsActive()) {
        if (document.fullscreenElement === explorer) document.exitFullscreen();
        else fsOverlay(false);
      }
    }
  };
}

/* Collapse every connected region of same-membership operations into one
 * item, and return the contracted graph -- pure, so it can be tested on its own.
 *
 *   N         merged-DAG nodes (``i``: input ids)
 *   ids       visible node ids, in topological order
 *   key(id)   membership key, comparable along edges
 *   excluded  nodes that never join a region (estimators kept visible)
 *   grouping  false: nothing collapses
 *   expanded  Set of group top ids drawn expanded
 *
 * Regions are the components of the edges whose two ends carry the same
 * pipelines. Contracting them keeps the graph acyclic: a path leaving a region
 * and coming back passes only through operations with that same membership
 * (sets only shrink along a path), which are then in the region. Excluded nodes
 * break that argument -- a region could flow into an estimator and back out --
 * so nodes are additionally keyed by the set of same-membership excluded nodes
 * upstream of them ("tag"), which cuts a region at each estimator and restores
 * the guarantee. One pass over the edges, plus the tag sets.
 *
 * Returns { G: {ids, inp, kids, lines, nOps}, groupOf, groupByVid }; a
 * collapsed group is the item ``vid`` (>= N.length) and ``G.ids`` is a
 * topological order of the contracted graph. */
export function contract(N, ids, key, excluded, grouping, expanded) {
  var vis = new Set(ids), k = {}, tagSet = {}, tag = {}, uf = {};
  var groupOf = {}, groupByVid = {};
  ids.forEach(function (id) {
    k[id] = key(id);
    var t = new Set();
    N[id].i.forEach(function (u) {
      if (!vis.has(u) || k[u] !== k[id]) return;
      tagSet[u].forEach(function (x) { t.add(x); });
      if (excluded(u)) t.add(u);
    });
    tagSet[id] = t;
    tag[id] = Array.from(t).sort(function (a, b) { return a - b; }).join(",");
    uf[id] = id;
  });
  function find(x) {
    while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; }
    return x;
  }
  if (grouping) {
    ids.forEach(function (v) {
      if (excluded(v)) return;
      N[v].i.forEach(function (u) {
        if (!vis.has(u) || excluded(u) || k[u] !== k[v] || tag[u] !== tag[v]) return;
        var a = find(u), b = find(v);
        if (a !== b) { if (a < b) uf[b] = a; else uf[a] = b; }   // root = smallest id
      });
    });
  }
  var byRoot = {};
  ids.forEach(function (id) { var r = find(id); (byRoot[r] = byRoot[r] || []).push(id); });

  var itemOf = {}, vid = N.length;
  Object.keys(byRoot).forEach(function (r) {
    var members = byRoot[r];
    if (members.length < 2) return;
    var g = { top: +r, members: members, vid: vid++, key: k[r] };
    members.forEach(function (id) { groupOf[id] = g; });
    groupByVid[g.vid] = g;
  });
  ids.forEach(function (id) {
    var g = groupOf[id];
    itemOf[id] = g && !expanded.has(g.top) ? g.vid : id;
  });

  var G = { ids: [], inp: {}, kids: {}, lines: {}, nOps: ids.length };
  var items = [], seenItem = new Set(), seenEdge = new Set();
  ids.forEach(function (id) {
    var it = itemOf[id];
    if (seenItem.has(it)) return;
    seenItem.add(it); items.push(it); G.inp[it] = []; G.kids[it] = [];
  });
  ids.forEach(function (v) {
    N[v].i.forEach(function (u) {
      if (!vis.has(u)) return;
      var a = itemOf[u], b = itemOf[v], e = a + ">" + b;
      if (a === b || seenEdge.has(e)) return;
      seenEdge.add(e); G.inp[b].push(a); G.kids[a].push(b);
    });
  });
  // topological order of the contracted graph (Kahn); roots by first member
  function first(it) { return it < N.length ? it : groupByVid[it].top; }
  var indeg = {}, order = [], queue = [];
  items.forEach(function (it) { indeg[it] = G.inp[it].length; if (!indeg[it]) queue.push(it); });
  queue.sort(function (a, b) { return first(a) - first(b); });
  while (queue.length) {
    var it = queue.shift();
    order.push(it);
    G.kids[it].forEach(function (c) { if (--indeg[c] === 0) queue.push(c); });
  }
  if (order.length !== items.length) {
    throw new Error("explorer: grouping produced a cycle (" + order.length + " of " +
                    items.length + " items ordered)");
  }
  G.ids = order;
  return { G: G, groupOf: groupOf, groupByVid: groupByVid };
}
