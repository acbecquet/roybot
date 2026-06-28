# scripts/make_bom.py
"""One-shot generator for the Phase 2 first-prototype Bill of Materials.

Scoped to a GrowBot-style first prototype (drive + chase brain + GrowBot guts +
portable power); the self-charging dock and cliff/bump sensors are deferred to
iteration 2. Reconciles what becqu already has against GrowBot's BOM
(https://github.com/britcruise9/GrowBot/blob/main/BOM.md) and the locked Roybot
N20 drivetrain spec. Power follows GrowBot's 1S->TP4056->MT3608 topology, dual-
rail (5.1V Pi / 6.0V motors) so the motors stay at the sim-tuned 6V.

Run once:  .venv/Scripts/python scripts/make_bom.py
After that the .xlsx is hand-maintained. Prices are rough ballparks, not quotes.
"""
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule

OUT = Path("docs/phase2/roybot-prototype-BOM.xlsx")
STATUSES = ["Buy", "Print", "Have", "Spare"]

# (Status, Group, Component, Qty, Spec / note, Suggested source, Est. unit USD)
ROWS = [
    # ===== BUY - Drivetrain (Roybot N20 spec; replaces GrowBot's servos) =====
    ("Buy", "Drivetrain", "Pololu HPCB 6V 100:1 micro metal gearmotor, EXTENDED back-shaft", 2,
     "Verified pick (~0.75 m/s, strong torque). EXTENDED shaft is REQUIRED so the encoder mounts. Replaces the 3216's spare DC motors", "Pololu", 18),
    ("Buy", "Drivetrain", "Pololu magnetic quadrature encoder pair (micro metal gearmotor)", 1,
     "1200 CPR/wheel-rev at 100:1 (already x4-decoded; don't x4 again). VCC=3.3V. Closed-loop speed + odometry", "Pololu", 9),
    ("Buy", "Drivetrain", "60 mm wheel for N20 (3 mm D-shaft), rubber tire", 2,
     "3216 wheels don't fit the N20 shaft. Buy rubber-tired (don't print - need grip)", "Pololu / Amazon", 4),
    ("Buy", "Drivetrain", "TB6612FNG dual H-bridge breakout", 1,
     "Motor driver. VM = 6V motor rail, VCC = 3.3V, PWM >=20 kHz, encoder-static stall cutoff", "Adafruit / Pololu", 5),

    # ===== BUY - Compute & sensing =====
    ("Buy", "Compute", "microSD card, 32 GB Class 10 / A1", 1,
     "NOT in your Pi kit! Raspberry Pi OS Lite 64-bit (GrowBot #2)", "Amazon", 8),
    ("Buy", "Sensing", "OV5647 camera - Pi ZERO version (narrow 22->15 CSI ribbon)", 1,
     "MUST be the Pi Zero narrow-ribbon version, NOT the standard camera ribbon (GrowBot #7)", "Amazon", 12),
    ("Buy", "Sensing", "MPU-6050 IMU (GY-521)", 1,
     "Tip/stall reflexes + proprioception. I2C 0x68, 3.3V (GrowBot #6)", "Amazon", 4),

    # ===== BUY - Voice & lights (GrowBot guts) =====
    ("Buy", "Voice/Lights", "INMP441 I2S microphone", 1, "Wake-word / voice in. 3.3V (GrowBot #8)", "Amazon", 4),
    ("Buy", "Voice/Lights", "MAX98357A I2S amplifier", 1, "Speaker driver, 3.2W class-D (GrowBot #9)", "Amazon", 5),
    ("Buy", "Voice/Lights", "Speaker 8 ohm, 0.5-3 W (small)", 1, "Audio out (GrowBot #10)", "Amazon", 2),
    ("Buy", "Voice/Lights", "WS2812B LED ring, 7 px, 5 V", 1, "Expressive 'personality' output (GrowBot #11)", "Amazon", 3),

    # ===== BUY - Power (GrowBot topology, dual-rail for the 6V N20) =====
    ("Buy", "Power", "18650 Li-ion cell, 2500-3500 mAh (1S)", 1,
     "Main battery. Bigger than GrowBot's 1S for N20 headroom + runtime (GrowBot #12)", "Amazon", 6),
    ("Buy", "Power", "18650 battery holder", 1, "Cell holder + contacts", "Amazon", 2),
    ("Buy", "Power", "TP4056 USB-C charge + protect module", 1,
     "In-place recharging; get the PROTECTED version with OUT+/OUT- (GrowBot #13)", "Amazon", 2),
    ("Buy", "Power", "MT3608 boost module (x2)", 2,
     "Dual-rail: set one to ~5.1V (Pi+logic+amp+LED), one to ~6.0V (motors). Isolates Pi from motor surge; GrowBot note #5 endorses a 2nd MT3608 for motors (GrowBot #14)", "Amazon", 2),
    ("Buy", "Power", "Electrolytic cap 470-1000 uF, 10 V+", 2, "One across each rail to cover surge dips (GrowBot #15)", "Amazon", 1),
    ("Buy", "Power", "SPST power switch", 1, "Main cutoff (GrowBot #16)", "Amazon", 2),

    # ===== BUY - Build / safety =====
    ("Buy", "Build", "Hookup wire + Dupont jumpers", 1, "Wire per the GrowBot diagram + the TB6612", "Amazon", 8),
    ("Buy", "Build", "3M double-sided mounting squares", 1, "GrowBot's no-screw board mounting (or print mounts)", "Amazon", 5),
    ("Buy", "Build", "PETG filament (known-safe), 1 spool", 1, "For the printed parts below - skip if your printer came with filament", "Amazon", 25),
    ("Buy", "Safety", "Captive lure (feather/pom on short rigid/braided arm)", 1,
     "#1 cat-safety item: NO free end, over-molded. Attended play only", "craft / Amazon", 5),

    # ===== PRINT (you have a printer) =====
    ("Print", "Drivetrain", "N20 -> 3216 motor-mount adapter", 2,
     "The 3216's mounts fit its DC motors, not N20s - print an adapter bracket", "3D print", 0),
    ("Print", "Structure", "Top cover + wheel shroud", 1,
     "Cat-safety: shroud wheel gaps (<4-6 mm or >25 mm), cover the electronics", "3D print", 0),
    ("Print", "Structure", "Captive-lure boom arm", 1, "Rigid arm holding the lure with no free end", "3D print", 0),
    ("Print", "Structure", "Battery + board mounts (optional)", 1, "Or just use the 3M squares above", "3D print", 0),

    # ===== HAVE (don't buy) =====
    ("Have", "Compute", "Raspberry Pi Zero 2 W", 1, "In your Pi kit", "(have)", 0),
    ("Have", "Compute", "2x20 GPIO header", 1, "In your Pi kit - needs SOLDERING onto the Pi", "(have)", 0),
    ("Have", "Compute", "Cables incl. USB power cable", 1, "In your Pi kit (bench power/flashing; on-robot the Pi runs off the 5V rail)", "(have)", 0),
    ("Have", "Chassis", "Adafruit 3216 chassis - frame + caster ball + hardware", 1, "Your base kit = the robot body", "(have)", 0),

    # ===== SPARE (have, unused in this build) =====
    ("Spare", "Chassis", "3216 bundled DC motors + 2 wheels", 1, "Superseded by the N20 drivetrain - keep as spares", "(have)", 0),
]

TOOLS = [
    ("Soldering iron + solder", "REQUIRED - solder the 40-pin header to the Pi + wire boards/motors (GrowBot: 'you will need to solder a 40-pin header')"),
    ("Multimeter", "REQUIRED - set each MT3608's output voltage BEFORE connecting the Pi (GrowBot setup note #1; they ship turned up high)"),
    ("3D printer", "You have it - for the Print rows"),
    ("Digital calipers", "Measure real wheel OD + track for the sim refit"),
    ("Kitchen scale", "Assembled weight for the sim refit"),
    ("Small screwdriver / hex set", "Assembly"),
    ("Wire strippers / flush cutters", "Wiring"),
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
BUY = PatternFill("solid", fgColor="FBE3E0")     # red  = buy
HAVE = PatternFill("solid", fgColor="E3F2E5")    # green = have
PRINT = PatternFill("solid", fgColor="E1ECF7")   # blue = print
SPARE = PatternFill("solid", fgColor="ECECEC")   # grey = spare


def build():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BOM"

    ws.merge_cells("A1:I1")
    ws["A1"] = "Roybot - First-Prototype Bill of Materials"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A2:I2")
    ws["A2"] = ("GrowBot-style first prototype (drive + chase brain + guts + portable power). Dock and "
                "cliff/bump sensors deferred to iteration 2. RED = buy (your shopping list), BLUE = 3D-print, "
                "GREEN = already have, GREY = spare. Power = 1S->TP4056->dual MT3608 (5.1V Pi / 6.0V motors). "
                "Prices are rough ballparks, not quotes.")
    ws["A2"].font = Font(italic=True, size=10, color="666666")
    ws["A2"].alignment = WRAP
    ws.row_dimensions[2].height = 54

    headers = ["Status", "Group", "Component", "Qty", "Spec / note",
               "Suggested source", "Est. unit (USD)", "Est. line (USD)", "Link / Part # (you fill)"]
    head_row = 8
    first_data = head_row + 1
    last_data = first_data + len(ROWS) - 1
    status_rng = f"A{first_data}:A{last_data}"
    line_rng = f"H{first_data}:H{last_data}"

    def summary(row, label, formula, color):
        c = ws.cell(row=row, column=6, value=label)
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="right")
        v = ws.cell(row=row, column=8, value=formula)
        v.number_format = MONEY
        v.font = Font(bold=True, color=color)

    summary(4, "TO BUY (your shopping list):", f'=SUMIF({status_rng},"Buy",{line_rng})', "B3261E")
    summary(5, "Already have / spare:", f'=SUMIF({status_rng},"Have",{line_rng})+SUMIF({status_rng},"Spare",{line_rng})', "2E7D32")
    ws.cell(row=6, column=6, value="Print count:").font = Font(bold=True)
    ws.cell(row=6, column=6).alignment = Alignment(horizontal="right")
    ws.cell(row=6, column=8, value=f'=COUNTIF({status_rng},"Print")').font = Font(bold=True, color="1F3A8A")

    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=head_row, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BORDER

    for i, (status, grp, comp, qty, spec, src, unit) in enumerate(ROWS):
        r = first_data + i
        vals = [status, grp, comp, qty, spec, src, unit, f"=D{r}*G{r}", ""]
        for c, val in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.border = BORDER
            cell.alignment = WRAP if c in (3, 5) else TOP
            if c in (7, 8):
                cell.number_format = MONEY

    dv = DataValidation(type="list", formula1='"%s"' % ",".join(STATUSES), allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(status_rng)
    ws.conditional_formatting.add(status_rng, CellIsRule(operator="equal", formula=['"Buy"'], fill=BUY))
    ws.conditional_formatting.add(status_rng, CellIsRule(operator="equal", formula=['"Have"'], fill=HAVE))
    ws.conditional_formatting.add(status_rng, CellIsRule(operator="equal", formula=['"Print"'], fill=PRINT))
    ws.conditional_formatting.add(status_rng, CellIsRule(operator="equal", formula=['"Spare"'], fill=SPARE))

    ws.auto_filter.ref = f"A{head_row}:I{last_data}"
    widths = [9, 13, 42, 5, 50, 22, 13, 13, 22]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws.freeze_panes = f"A{first_data}"

    # Tools sheet
    ts = wb.create_sheet("Tools")
    ts.merge_cells("A1:B1")
    ts["A1"] = "Tools (not consumed - check you have these)"
    ts["A1"].font = TITLE_FONT
    for c, h in enumerate(["Tool", "Needed for"], start=1):
        cell = ts.cell(row=3, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.border = BORDER
    for i, (tool, use) in enumerate(TOOLS, start=4):
        ts.cell(row=i, column=1, value=tool).border = BORDER
        a = ts.cell(row=i, column=2, value=use)
        a.border = BORDER
        a.alignment = WRAP
    ts.column_dimensions["A"].width = 28
    ts.column_dimensions["B"].width = 70
    ts.freeze_panes = "A4"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    n_buy = sum(1 for r in ROWS if r[0] == "Buy")
    print(f"wrote {OUT}  ({len(ROWS)} rows, {n_buy} to buy)")


if __name__ == "__main__":
    build()
