# scripts/make_bom.py
"""One-shot generator for the Phase 2 Bill of Materials spreadsheet.

Consolidates the BOM that was previously scattered across the spec (§5 deltas)
and the Phase 2 checklist into one buyable docs/phase2/roybot-BOM.xlsx.

Run once:  .venv/Scripts/python scripts/make_bom.py
After that the .xlsx is hand-maintained (prices/links/status) in Excel — this
script is the starting point, not a live source of truth. Prices are rough
ballparks for budgeting, NOT quotes.
"""
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule

OUT = Path("docs/phase2/roybot-BOM.xlsx")

# Status vocabulary (used for the dropdown + color rules)
STATUSES = ["Have", "Bought", "Included in kit", "To buy", "Later", "Don't order"]

# (Category, Component, Qty, Spec / purpose, Status, Suggested source, Est. unit USD)
ROWS = [
    # ---- Drivetrain / structure ----
    ("Drivetrain", "Adafruit 3216 Mini Round Robot Chassis Kit", 1,
     "The robot body (matches the sim twin): round aluminum frame + 2 DC motors + 2 wheels + caster ball",
     "Bought", "Adafruit / reseller", 20),
    ("Drivetrain", "  - 2x DC drive motor (3-6 V)", 2,
     "In the 3216 kit. Open-loop (no encoder)", "Included in kit", "(3216 kit)", 0),
    ("Drivetrain", "  - 2x wheel + tire", 2, "In the 3216 kit", "Included in kit", "(3216 kit)", 0),
    ("Drivetrain", "  - caster ball", 1, "In the 3216 kit; free-sliding 3rd contact",
     "Included in kit", "(3216 kit)", 0),
    ("Drivetrain", "TB6612FNG dual H-bridge breakout", 1,
     "Motor driver for the two DC motors. Chosen over DRV8833/L298N (efficiency)",
     "To buy", "Adafruit / Pololu / Amazon", 5),
    ("Drivetrain", "N20 gearmotor w/ magnetic encoder (UPGRADE)", 2,
     "LATER: closed-loop speed + odometry for self-docking (#6). Needs N20 brackets",
     "Later", "Pololu", 8),

    # ---- Compute ----
    ("Compute", "Raspberry Pi Zero 2 W", 1, "Onboard brain; runs policy MLP @50 Hz",
     "Have", "(GrowBot carry-over)", 15),
    ("Compute", "microSD card 32 GB", 1, "Raspberry Pi OS + roybot code", "To buy", "Amazon", 8),
    ("Compute", "40-pin GPIO header (if Pi unheadered)", 1, "Solder header for motor/sensor wiring",
     "To buy", "Adafruit", 2),

    # ---- Sensors ----
    ("Sensors", "MPU-6050 IMU", 1, "Roll/pitch + tip/stall reflexes; proprioception for the policy",
     "Have", "(GrowBot carry-over)", 4),
    ("Sensors", "OV5647 camera (CSI)", 1, "Cat tracking (perception, sub-project #2)",
     "Have", "(GrowBot carry-over)", 10),
    ("Sensors", "Pi Zero CSI ribbon (mini -> standard)", 1, "Pi Zero needs the narrow CSI cable",
     "To buy", "Adafruit", 3),

    # ---- Audio / expression ----
    ("Audio/Expression", "MAX98357A I2S amplifier", 1, "Speaker driver (voice reports, #5)",
     "Have", "(GrowBot carry-over)", 6),
    ("Audio/Expression", "Speaker 4-8 ohm (small)", 1, "Audio out", "Have", "(GrowBot carry-over)", 2),
    ("Audio/Expression", "INMP441 I2S microphone", 1, "Wake-word / voice in (#5)",
     "Have", "(GrowBot carry-over)", 5),
    ("Audio/Expression", "WS2812 RGB LED ring", 1, "Expressive 'personality' output",
     "Have", "(GrowBot carry-over)", 8),

    # ---- Power ----
    ("Power", "LiPo battery (~6 V: 2S 7.4 V small, or 6 V pack)", 1,
     "Main power. Motors are 3-6 V DC -> ~6 V target. Pins the cell-count decision",
     "To buy", "HobbyKing / Amazon", 10),
    ("Power", "5 V buck regulator", 1, "Dedicated Pi rail; isolates Pi from motor current surges",
     "To buy", "Pololu / Amazon", 4),
    ("Power", "Bulk capacitor 470-1000 uF", 2, "Brownout protection on the Pi + motor rails",
     "To buy", "Amazon", 1),
    ("Power", "Charge management (1S TP4056 OR 2S BMS)", 1,
     "Per cell choice; couples to the dock charge path (#6)", "To buy", "Amazon", 2),
    ("Power", "Inline fuse + holder", 1, "Battery short protection", "To buy", "Amazon", 2),
    ("Power", "Power switch (SPST)", 1, "Main cutoff", "To buy", "Amazon", 2),

    # ---- Cat-safety (non-negotiable) ----
    ("Cat-safety", "Captive lure (feather/pom on short rigid/braided arm)", 1,
     "#1 hazard control: NO free end, over-molded. No loose string ever", "To buy", "craft + over-mold", 5),
    ("Cat-safety", "PETG filament (known-safe), spool", 1,
     "Cover / wheel shroud / lure-boom prints", "To buy", "Amazon", 25),
    ("Cat-safety", "Heat-set inserts + screws (captive fasteners)", 1,
     "Tool-locked battery hatch; no swallowable parts", "To buy", "Amazon", 8),

    # ---- Wiring / consumables ----
    ("Wiring", "Silicone hookup wire (assorted AWG)", 1, "Motor + logic wiring", "To buy", "Amazon", 8),
    ("Wiring", "JST / Dupont connectors", 1, "Removable connections", "To buy", "Amazon", 6),
    ("Wiring", "Heat-shrink assortment", 1, "Insulation", "Have", "Amazon", 5),
    ("Wiring", "Standoffs / M2.5 hardware", 1, "Mount Pi + boards to the chassis", "To buy", "Amazon", 6),

    # ---- Removed vs GrowBot (do NOT order) ----
    ("Removed", "2x SCS0009 serial servo", 0, "Replaced by DC motors + H-bridge", "Don't order", "-", 0),
    ("Removed", "1 kohm resistor (servo bus)", 0, "Servo-bus only", "Don't order", "-", 0),
    ("Removed", "MT3608 boost converter", 0, "Not needed with the new power plan", "Don't order", "-", 0),
]

TOOLS = [
    ("Soldering iron + solder", "Wiring the H-bridge, headers, power"),
    ("Multimeter", "Continuity, voltage, current checks"),
    ("Digital calipers", "The 4 sim-refit measurements (wheel OD, track, caster offset)"),
    ("Kitchen scale", "Assembled weight for the sim refit"),
    ("3D printer", "Cover / wheel shroud / lure boom (PETG)"),
    ("Wire strippers / flush cutters", "Wiring"),
    ("Small screwdriver + hex set", "Assembly"),
]

# ---- styling helpers ----
HEAD_FILL = PatternFill("solid", fgColor="1F3A8A")
HEAD_FONT = Font(bold=True, color="FFFFFF", size=11)
TITLE_FONT = Font(bold=True, size=15, color="1F3A8A")
CAT_FILL = PatternFill("solid", fgColor="E8ECF7")
THIN = Side(style="thin", color="D0D4DC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(vertical="top", wrap_text=True)
TOP = Alignment(vertical="top")
MONEY = "$#,##0.00"

GREEN = PatternFill("solid", fgColor="E3F2E5")
RED = PatternFill("solid", fgColor="FBE3E0")
YELLOW = PatternFill("solid", fgColor="FBF3DA")
GREY = PatternFill("solid", fgColor="ECECEC")


def build():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BOM"

    headers = ["Category", "Component", "Qty", "Key spec / purpose", "Status",
               "Suggested source", "Est. unit (USD)", "Est. line (USD)", "Link / Part # (you fill)"]

    # Title + note
    ws.merge_cells("A1:I1")
    ws["A1"] = "Roybot - Phase 2 Bill of Materials"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A2:I2")
    ws["A2"] = ("Prices are rough ballparks for budgeting, not quotes. Status drives the colors and the "
                "'still to buy' total below. Edit freely - this sheet is yours to maintain.")
    ws["A2"].font = Font(italic=True, size=10, color="666666")
    ws["A2"].alignment = WRAP
    ws.row_dimensions[2].height = 28

    head_row = 4
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=head_row, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BORDER

    first_data = head_row + 1
    r = first_data
    for (cat, comp, qty, spec, status, src, unit) in ROWS:
        ws.cell(row=r, column=1, value=cat).alignment = TOP
        ws.cell(row=r, column=2, value=comp).alignment = TOP
        ws.cell(row=r, column=3, value=qty).alignment = TOP
        ws.cell(row=r, column=4, value=spec).alignment = WRAP
        ws.cell(row=r, column=5, value=status).alignment = TOP
        ws.cell(row=r, column=6, value=src).alignment = TOP
        u = ws.cell(row=r, column=7, value=unit)
        u.number_format = MONEY
        u.alignment = TOP
        line = ws.cell(row=r, column=8, value=f"=C{r}*G{r}")
        line.number_format = MONEY
        line.alignment = TOP
        ws.cell(row=r, column=9, value="").alignment = TOP
        for c in range(1, 10):
            ws.cell(row=r, column=c).border = BORDER
        r += 1
    last_data = r - 1

    # Totals
    r += 1
    ws.cell(row=r, column=7, value="Estimated still to buy:").font = Font(bold=True)
    t1 = ws.cell(row=r, column=8,
                 value=f'=SUMIF(E{first_data}:E{last_data},"To buy",H{first_data}:H{last_data})')
    t1.number_format = MONEY
    t1.font = Font(bold=True, color="B3261E")
    r += 1
    ws.cell(row=r, column=7, value="Estimated total (all rows):").font = Font(bold=True)
    t2 = ws.cell(row=r, column=8, value=f"=SUM(H{first_data}:H{last_data})")
    t2.number_format = MONEY
    t2.font = Font(bold=True)

    # Status dropdown
    dv = DataValidation(type="list", formula1='"%s"' % ",".join(STATUSES), allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E{first_data}:E{last_data}")

    # Conditional colors on Status column
    rng = f"E{first_data}:E{last_data}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"To buy"'], fill=RED))
    for val in ('"Have"', '"Bought"', '"Included in kit"'):
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[val], fill=GREEN))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"Later"'], fill=YELLOW))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"Don\'t order"'], fill=GREY))

    # Widths + freeze
    widths = [16, 42, 5, 46, 16, 24, 14, 14, 26]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws.freeze_panes = f"A{first_data}"

    # ---- Tools sheet ----
    ts = wb.create_sheet("Tools")
    ts.merge_cells("A1:B1")
    ts["A1"] = "Tools (not consumed - reference)"
    ts["A1"].font = TITLE_FONT
    for c, h in enumerate(["Tool", "Used for"], start=1):
        cell = ts.cell(row=3, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.border = BORDER
    for i, (tool, use) in enumerate(TOOLS, start=4):
        ts.cell(row=i, column=1, value=tool).border = BORDER
        ts.cell(row=i, column=2, value=use).border = BORDER
    ts.column_dimensions["A"].width = 30
    ts.column_dimensions["B"].width = 52
    ts.freeze_panes = "A4"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}  ({last_data - first_data + 1} line items)")


if __name__ == "__main__":
    build()
