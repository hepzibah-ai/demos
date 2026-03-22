---
name: drawio
description: |
  Create and edit draw.io diagram files (.drawio). Use when user wants a block diagram,
  architecture diagram, data flow, FSM, or any visual diagram. Generates XML files
  saved directly to the repository for version control and human editing.
---

# draw.io Diagram Skill

Generate `.drawio` XML files saved to the repository. Files are version-controlled,
diffable, and editable in the draw.io desktop app, VS Code extension, or web editor.

## Invocation

`/drawio <description of diagram>`

Examples:
- `/drawio tile architecture block diagram`
- `/drawio NoC packet flow from manager to PE array`
- `/drawio controller FSM states`

## File Placement

Save diagrams in the figures directory:

| Context | Location |
|---------|----------|
| Default | `figures/<name>.drawio` |
| User specifies path | Use that path |

Create the `figures/` directory if it doesn't exist.

**Naming**: `snake_case.drawio` — descriptive, matching the doc it illustrates.

## Format Selection

draw.io files are always XML. But the *input notation* can vary:

| Approach | When to use |
|----------|-------------|
| **Direct XML** | Most diagrams. Full control over layout, positioning, styling. |
| **Mermaid embedded** | Simple flowcharts/sequences where auto-layout is fine. |

Prefer direct XML — it gives precise control and is what the file stores natively.

## XML Template

Every `.drawio` file must be well-formed XML. Use this skeleton:

```xml
<mxfile host="Claude" modified="YYYY-MM-DDTHH:MM:SS" type="device">
  <diagram id="ID" name="Page-1">
    <mxGraphModel dx="1024" dy="768" grid="1" gridSize="10" guides="1"
                  tooltips="1" connect="1" arrows="1" fold="1" page="1"
                  pageScale="1" pageWidth="1100" pageHeight="850">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <!-- Diagram content here -->
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

- `id="0"` is the root layer (required, never omit)
- `id="1"` is the default layer (required, parent for all content cells)
- Content cells start at `id="2"` and increment

## Cell ID Convention

Use sequential integer IDs starting at 2. For edges, continue the sequence after nodes.

## Common Shape Styles

### Blocks / Boxes
```
rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;
```

### Rounded hardware blocks
```
rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#d5e8d4;strokeColor=#82b366;
```

### Group / Container
```
rounded=1;whiteSpace=wrap;html=1;container=1;collapsible=0;fillColor=#f5f5f5;strokeColor=#666666;strokeWidth=2;dashed=1;
```

### Arrow / Edge (with arrowhead)
```
edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;
```

### Plain line (no arrowhead)
```
endArrow=none;html=1;strokeColor=#666666;strokeWidth=1.5;
```
Use for axis lines, outlines, and decorative lines that shouldn't have arrows.

### Text label (no border)
```
text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;
```

### Ellipse / Dot
```
ellipse;fillColor=#FF0000;strokeColor=none;aspect=fixed;
```
Use small ellipses (6-10px) as data-point markers on charts.

## Shape Library

Draw.io includes a large stencil library beyond basic rectangles. Use the `shape=` style key.

### Built-in geometric shapes

| Shape | Style | Notes |
|-------|-------|-------|
| Ellipse | `ellipse;` | Circles when `aspect=fixed` |
| Triangle | `triangle;` | Isoceles, point-right by default |
| Diamond | `rhombus;` | Decision nodes in flowcharts |
| Cylinder | `shape=cylinder3;` | Databases, storage |
| Hexagon | `shape=hexagon;` | |
| Parallelogram | `shape=parallelogram;` | I/O in flowcharts |
| Star | `shape=mxgraph.basic.star;` | |
| Acute triangle | `shape=mxgraph.basic.acute_triangle;dx=0.5;` | `dx` controls apex position (0-1) |

### Stencil libraries (`mxgraph.*`)

Draw.io organizes extended shapes into libraries. Common ones:

| Library | Prefix | Examples |
|---------|--------|----------|
| Basic | `mxgraph.basic.*` | `polygon`, `star`, `acute_triangle`, `cone` |
| Flowchart | `mxgraph.flowchart.*` | `process`, `decision`, `document` |
| Electrical | `mxgraph.electrical.*` | Logic gates, components |
| Network | `mxgraph.network.*` | Servers, routers, clouds |

### Custom polygons

For arbitrary shapes (triangles, pentagons, data-driven regions), use a cell with
explicit geometry points. Draw each side as a separate edge with `endArrow=none`:

```xml
<!-- Triangle outline: three edges forming a closed path -->
<mxCell id="10" value="" style="endArrow=none;html=1;strokeColor=#FF0000;strokeWidth=2;"
        edge="1" parent="1">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="400" y="200" as="sourcePoint"/>
    <mxPoint x="600" y="400" as="targetPoint"/>
  </mxGeometry>
</mxCell>
<mxCell id="11" value="" style="endArrow=none;html=1;strokeColor=#FF0000;strokeWidth=2;"
        edge="1" parent="1">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="600" y="400" as="sourcePoint"/>
    <mxPoint x="200" y="400" as="targetPoint"/>
  </mxGeometry>
</mxCell>
<mxCell id="12" value="" style="endArrow=none;html=1;strokeColor=#FF0000;strokeWidth=2;"
        edge="1" parent="1">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="200" y="400" as="sourcePoint"/>
    <mxPoint x="400" y="200" as="targetPoint"/>
  </mxGeometry>
</mxCell>
```

This approach gives no fill. For filled custom polygons, use `acute_triangle` with
`rotation` and `dx` parameters, but note these are hard to position precisely from
code (designed for GUI interaction).

## Color Palette (H0 conventions)

Use consistent colors across H0 diagrams:

| Element | Fill | Stroke | Style constant |
|---------|------|--------|----------------|
| PE / Compute | `#dae8fc` | `#6c8ebf` | Blue |
| Memory / CRAM | `#d5e8d4` | `#82b366` | Green |
| NoC / Interconnect | `#fff2cc` | `#d6b656` | Yellow |
| Control / Manager | `#f8cecc` | `#b85450` | Red |
| External / DRAM | `#e1d5e7` | `#9673a6` | Purple |
| Grouping / Tile | `#f5f5f5` | `#666666` | Grey |

## XML Well-Formedness Rules

**CRITICAL**: The XML must parse without errors.

1. **No double hyphens in comments**: `--` is illegal inside `<!-- -->`. Use single hyphens or rephrase.
2. **Escape special characters** in `value` attributes: `&amp;` `&lt;` `&gt;` `&quot;`
3. **Quote all attributes**: `style="..."` not `style=...`
4. **Self-close empty elements**: `<mxGeometry ... />` not `<mxGeometry ...></mxGeometry>`
5. **Encode line breaks** in labels: Use `&#xa;` for newlines inside `value` attributes, or use `<br>` with `html=1` in the style.

## Layout Guidelines

- **Grid-align** positions to multiples of 10 (matches gridSize=10)
- **Standard block size**: 120x60 for simple boxes, 160x80 for labeled blocks
- **Spacing**: 40-60px between blocks, 80-100px between groups
- **Flow direction**: Left-to-right or top-to-bottom (state preference in the diagram)
- **Keep it readable**: If a diagram has >20 nodes, consider splitting into multiple pages (multiple `<diagram>` elements in the `<mxfile>`)

## Multi-Page Diagrams

For complex systems, use multiple pages:

```xml
<mxfile host="Claude">
  <diagram id="overview" name="Overview">
    <mxGraphModel>...</mxGraphModel>
  </diagram>
  <diagram id="detail" name="Detail View">
    <mxGraphModel>...</mxGraphModel>
  </diagram>
</mxfile>
```

## Edge Definitions

Edges connect cells by ID using `source` and `target`:

```xml
<mxCell id="10" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;"
        edge="1" parent="1" source="2" target="3">
  <mxGeometry relative="1" as="geometry"/>
</mxCell>
```

For labeled edges, set the `value` attribute.

## When to Use drawio (and When Not To)

**Good fit** (use this skill):
- Architecture block diagrams (tiles, chiplets, arrays)
- Data flow diagrams (weight streaming, KV cache paths)
- FSM state diagrams
- Network topology diagrams
- Anything with labeled boxes, arrows, and grouping

**Poor fit** (use matplotlib/marimo instead):
- Data-driven charts (scatter, bar, line, radar/spider)
- Plots with computed coordinates from analysis data
- Anything requiring axes with tick marks and scales
- Figures that change when the underlying data changes

**Borderline** (drawio works but is painful):
- Schematic radar/triangle plots for marketing (data-driven geometry)
- Timing diagrams (better in wavedrom)
- Tables with many rows (better as markdown)

The key question: **are the positions computed from data or hand-placed?**
If computed, use a plotting tool. If hand-placed for visual communication, use drawio.

## Data-Driven Diagrams in drawio

If you must create a data-driven diagram in drawio (e.g. a radar plot):

1. **Compute coordinates externally** — use Python/marimo to calculate vertex
   positions from data values, then template them into XML
2. **Document the mapping** — include the formula in XML comments so future
   editors know how positions relate to data (e.g. `f = log10(value) / 3`)
3. **Use edge-based outlines** — draw polygons as N edges with `endArrow=none`
   (see Custom Polygons above); this is the most reliable approach
4. **Add small ellipse dots** at data vertices for visual emphasis
5. **Accept limitations** — no fill on edge-based polygons; axis ticks are
   manually positioned text labels; coordinate errors are hard to spot

## Process

1. **Understand what's being diagrammed** — ask if unclear
2. **Check fit** — is drawio the right tool? (see above)
3. **Choose layout** — flow direction, grouping, number of pages
4. **Generate XML** — well-formed, grid-aligned, using H0 color palette
5. **Write the file** — to the appropriate `docs/figures/` location
6. **Validate** — run `xmllint --noout <file>` to catch XML errors
7. **Report** — tell the user the file path and how to open it

## Opening the Diagram

After creating the file, tell the user:

> Diagram saved to `<path>`. Open with:
> - **VS Code**: Install the "Draw.io Integration" extension, then click the file
> - **Desktop**: Open with [draw.io desktop app](https://github.com/jgraph/drawio-desktop/releases)
> - **Web**: Go to app.diagrams.net and use File > Open to load the file

## Editing Existing Diagrams

When asked to modify an existing `.drawio` file:

1. **Read the file first** — understand existing structure and IDs
2. **Preserve existing IDs** — don't renumber cells that haven't changed
3. **Increment from max ID** — new cells get IDs above the current maximum
4. **Use Edit tool** — for targeted changes to specific cells
5. **Use Write tool** — only for complete rewrites

## Edges Without Cells

Edges don't need `source`/`target` cell IDs. You can use explicit source/target points
for free-floating lines (axis lines, outlines, annotations):

```xml
<mxCell id="10" value="" style="endArrow=none;html=1;strokeColor=#666666;strokeWidth=1.5;"
        edge="1" parent="1">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="100" y="400" as="sourcePoint"/>
    <mxPoint x="500" y="400" as="targetPoint"/>
  </mxGeometry>
</mxCell>
```

Use `endArrow=classic` for a one-way arrow, or `endArrow=classic;startArrow=classic` for
bidirectional. Use `endArrow=none` for plain lines.

## Anti-Patterns

- **Don't generate SVG or PNG** — generate the editable `.drawio` source
- **Don't embed base64 images** — use draw.io's built-in shapes
- **Don't use absolute coordinates larger than page** — stay within pageWidth/pageHeight
- **Don't create orphan edges** — edges with `source`/`target` attributes must reference valid cell IDs (edges with explicit sourcePoint/targetPoint geometry are fine without cell references)
- **Don't use drawio for data plots** — use matplotlib/marimo for charts with computed data; drawio is for hand-designed diagrams
- **Don't compute trig in your head** — if a diagram needs coordinate math, compute it in Python first and template the results into XML
