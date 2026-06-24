# scripts/make_bom.py
"""One-shot generator for the Phase 2 Bill of Materials spreadsheet.

Consolidates the BOM (previously scattered across the spec §5 and the checklist)
into one buyable docs/phase2/roybot-BOM.xlsx for the FINAL product. Split into
Purchased vs To-buy with live subtotals; Status column drives the color + totals.

Run once:  .venv/Scripts/python scripts/make_bom.py
After that the .xlsx is hand-maintained (flip Status as you order, edit prices/
links). This script is the starting point, not a live source of truth. Prices
are rough ballparks for budgeting, NOT quotes.
"""
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule

OUT = Path("docs/phase2/roybot-BOM.xlsx")
STATUSES = ["Purchased", "To buy"]

# (Status, Subsystem, Component, Qty, Spec / purpose, Suggested source, Est. unit USD)
ROWS = [
    # ===== PURCHASED / ON HAND =====
    ("Purchased", "Bot", "Adafruit 3216 Mini Round Robot Chassis Kit", 1,
     "The bot body: round aluminum frame + caster ball. (Its 2 DC motors + 2 wheels are now SPARE - superseded by the N20 upgrade.)",
     "Adafruit / reseller", 20),
    ("Purchased", "Compute", "Raspberry Pi Zero 2 W - full setup kit", 1,
     "The brain (runs the policy @50 Hz). Setup kit covers the sub-items below",
     "(owned)", 35),
    ("Purchased", "Compute", "  - microSD card (preloaded)", 1, "In the Pi setup kit", "(Pi kit)", 0),
    ("Purchased", "Compute", "  - USB power supply", 1, "In the Pi setup kit", "(Pi kit)", 0),
    ("Purchased", "Compute", "  - GPIO headers + adapters + case", 1, "In the Pi setup kit", "(Pi kit)", 0),

    # ===== TO BUY - Drivetrain (N20 upgrade, now core) =====
    ("To buy", "Drivetrain", "N20 gearmotor w/ magnetic encoder, 6 V", 2,
     "Final drivetrain: closed-loop speed + odometry (needed for docking). Pick gear ratio for speed/torque",
     "Pololu", 9),
    ("To buy", "Drivetrain", "N20 motor bracket / mount", 2,
     "Mount N20s to the 3216 frame - likely needs a small printed adapter to fit the frame's hole pattern",
     "Pololu / Amazon", 3),
    ("To buy", "Drivetrain", "Wheel for N20 (3 mm D-shaft), ~60 mm", 2,
     "The 3216 wheels don't fit the N20 shaft", "Pololu / Amazon", 4),
    ("To buy", "Drivetrain", "TB6612FNG dual H-bridge breakout", 1,
     "Motor driver (efficient; chosen over DRV8833/L298N)", "Adafruit / Pololu", 5),

    # ===== TO BUY - Sensors =====
    ("To buy", "Sensors", "MPU-6050 IMU", 1, "Roll/pitch + tip/stall reflexes; proprioception for the policy",
     "Amazon", 4),
    ("To buy", "Sensors", "OV5647 camera (Pi-compatible)", 1, "Cat tracking (perception, #2)", "Amazon", 10),
    ("To buy", "Sensors", "Pi Zero CSI ribbon (mini -> standard)", 1, "Pi Zero needs the narrow CSI cable",
     "Adafruit", 3),
    ("To buy", "Sensors", "IR reflectance cliff/edge sensor", 2,
     "Edge detect -> back off (safety floor). Stops falls off tables/stairs", "Amazon", 2),
    ("To buy", "Sensors", "Bump microswitch (lever)", 2, "Front bump detect -> back off", "Amazon", 1),

    # ===== TO BUY - Audio / expression (companion layer #5) =====
    ("To buy", "Audio", "INMP441 I2S microphone", 1, "Wake-word / voice in (#5)", "Amazon", 5),
    ("To buy", "Audio", "MAX98357A I2S amplifier", 1, "Speaker driver (voice reports #5)", "Adafruit", 6),
    ("To buy", "Audio", "Speaker 4-8 ohm (small)", 1, "Audio out", "Amazon", 2),
    ("To buy", "Expression", "WS2812 RGB LED ring", 1, "Expressive 'personality' output", "Amazon", 8),

    # ===== TO BUY - Power =====
    ("To buy", "Power", "LiPo battery 2S 7.4 V (small)", 1,
     "Main power for the 6 V N20s (PWM-limited). Couples to the charger/BMS choice", "HobbyKing / Amazon", 12),
    ("To buy", "Power", "5 V buck regulator (3 A)", 1, "Dedicated Pi rail; isolates Pi from motor surges",
     "Pololu / Amazon", 5),
    ("To buy", "Power", "Bulk capacitor 470-1000 uF", 2, "Brownout protection (Pi + motor rails)", "Amazon", 1),
    ("To buy", "Power", "2S LiPo BMS / protection + charge board", 1,
     "Safe charge/discharge; this is the on-bot side of docking", "Amazon", 4),
    ("To buy", "Power", "XT30 / JST battery connector set", 1, "Battery + charge connectors", "Amazon", 3),
    ("To buy", "Power", "Inline fuse + holder", 1, "Battery short protection", "Amazon", 2),
    ("To buy", "Power", "Power switch (SPST)", 1, "Main cutoff", "Amazon", 2),

    # ===== TO BUY - Self-charging dock (#6) =====
    ("To buy", "Dock", "Pogo-pin charging contacts (bot + dock pair)", 1,
     "Spring-pin charge contacts for self-docking", "Amazon", 6),
    ("To buy", "Dock", "TSOP38238 IR receiver", 1, "On the bot: IR homing to the dock", "Amazon", 2),
    ("To buy", "Dock", "IR LED beacon + driver", 1, "On the dock: homing beacon", "Amazon", 3),
    ("To buy", "Dock", "Dock PSU (5 V / 2 A) + barrel jack", 1, "Powers the charging dock", "Amazon", 8),
    ("To buy", "Dock", "Dock base + alignment funnel (PETG print)", 1, "Passive funnel; printed", "(PETG)", 0),

    # ===== TO BUY - Structure / cat-safety =====
    ("To buy", "Cat-safety", "Captive lure (feather/pom on short rigid/braided arm)", 1,
     "#1 hazard control: NO free end, over-molded. No loose string ever", "craft + over-mold", 5),
    ("To buy", "Structure", "PETG filament (known-safe) spool", 1,
     "Cover / wheel shroud / lure boom / dock funnel / N20 adapter prints", "Amazon", 25),
    ("To buy", "Structure", "Heat-set inserts + screws", 1, "Tool-locked battery hatch; captive fasteners",
     "Amazon", 8),

    # ===== TO BUY - Wiring / consumables =====
    ("To buy", "Wiring", "Silicone hookup wire (assorted AWG)", 1, "Motor + logic wiring", "Amazon", 8),
    ("To buy", "Wiring", "JST / Dupont connector kit", 1, "Removable connections", "Amazon", 6),
    ("To buy", "Wiring", "Standoffs / M2.5 hardware kit", 1, "Mount Pi + boards to the chassis", "Amazon", 6),
    ("To buy", "Wiring", "Heat-shrink assortment", 1, "Insulation", "Amazon", 5),
]

TOOLS = [
    ("Soldering iron + solder", "Wiring the H-bridge, headers, power, encoders"),
    ("Multimeter", "Continuity, voltage, current checks"),
    ("Digital calipers", "The 4 sim-refit measurements (wheel OD, track, caster offset)"),
    ("Kitchen scale", "Assembled weight for the sim refit"),
    ("3D printer", "Cover / shroud / lure boom / dock funnel / N20 adapter (PETG)"),
    ("Wire strippers / flush cutters", "Wiring"),
    ("Small screwdriver + hex set", "Assembly"),
]

# ---- styling ----
HEAD_FILL = PatternFill("solid", fgColor="1F3A8A")
HEAD_FONT = Font(bold=True, color="FFFFFF", size=11)
TITLE_FONT = Font(bold=True, size=15, color="1F3A8A")
THIN = Side(style="thin", color="D0D4DC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(vertical="top", wrap_text=True)
TOP = Alignment(vertical="top")
MONEY = "$#,##0.00"
GREEN = PatternFill("solid", fgColor="E3F2E5")
RED = PatternFill("solid", fgColor="FBE3E0")


def build():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BOM"

    ws.merge_cells("A1:I1")
    ws["A1"] = "Roybot - Final-Product Bill of Materials"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A2:I2")
    ws["A2"] = ("Everything for the complete bot + self-charging dock. Flip Status (Purchased / To buy) as you "
                "order - colors and the totals below follow. Prices are rough ballparks, not quotes. "
                "Note: N20 mounting to the 3216 frame may need a small printed adapter.")
    ws["A2"].font = Font(italic=True, size=10, color="666666")
    ws["A2"].alignment = WRAP
    ws.row_dimensions[2].height = 42

    # Summary block (forward-references the data range below)
    headers = ["Status", "Subsystem", "Component", "Qty", "Key spec / purpose",
               "Suggested source", "Est. unit (USD)", "Est. line (USD)", "Link / Part # (you fill)"]
    head_row = 8
    first_data = head_row + 1
    last_data = first_data + len(ROWS) - 1
    status_rng = f"A{first_data}:A{last_data}"
    line_rng = f"H{first_data}:H{last_data}"

    def summary(row, label, formula, color=None, bold=True):
        c = ws.cell(row=row, column=6, value=label)
        c.font = Font(bold=bold)
        c.alignment = Alignment(horizontal="right")
        v = ws.cell(row=row, column=8, value=formula)
        v.number_format = MONEY
        v.font = Font(bold=bold, color=color or "000000")

    summary(4, "Already spent (purchased):", f'=SUMIF({status_rng},"Purchased",{line_rng})', "2E7D32")
    summary(5, "Still to buy:", f'=SUMIF({status_rng},"To buy",{line_rng})', "B3261E")
    summary(6, "FINAL-PRODUCT TOTAL:", f"=SUM({line_rng})")

    # Header
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=head_row, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BORDER

    # Data
    for i, (status, sub, comp, qty, spec, src, unit) in enumerate(ROWS):
        r = first_data + i
        vals = [status, sub, comp, qty, spec, src, unit, f"=D{r}*G{r}", ""]
        for c, val in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.border = BORDER
            cell.alignment = WRAP if c in (3, 5) else TOP
            if c in (7, 8):
                cell.number_format = MONEY

    # Dropdown + color by status
    dv = DataValidation(type="list", formula1='"%s"' % ",".join(STATUSES), allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(status_rng)
    ws.conditional_formatting.add(status_rng, CellIsRule(operator="equal", formula=['"Purchased"'], fill=GREEN))
    ws.conditional_formatting.add(status_rng, CellIsRule(operator="equal", formula=['"To buy"'], fill=RED))

    # Filter + widths + freeze
    ws.auto_filter.ref = f"A{head_row}:I{last_data}"
    widths = [11, 14, 40, 5, 44, 22, 13, 13, 24]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws.freeze_panes = f"A{first_data}"

    # Tools sheet
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
    ts.column_dimensions["B"].width = 56
    ts.freeze_panes = "A4"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}  ({len(ROWS)} line items)")


if __name__ == "__main__":
    build()
