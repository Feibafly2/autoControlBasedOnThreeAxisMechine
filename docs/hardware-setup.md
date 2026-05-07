# Hardware Setup

![Hardware prototype](../assets/hardware-prototype.jpg)

This document records the current hardware plan for the dual-Y three-axis slider prototype.

## Mechanical Platform

- Type: dual-Y XYZ PLC lead-screw positioning slider
- Frame: 2040 aluminum extrusion
- X axis: reinforced X axis with two connecting rods
- Guide rods: 8 mm chrome-plated rods with linear bearings
- Lead screws: 8 mm T-type lead screws
- Lead: 8 mm per revolution
- Travel:
  - X: 300 mm
  - Y: 300 mm
  - Z: 130 mm

With 200-step/rev motors and TB6600 drivers set to 8 microsteps:

```text
steps_per_mm = 200 * 8 / 8 = 200 steps/mm
```

## Electronics

- TB6600 stepper drivers x3
- Arduino Uno R3
- ESP8266 network bridge
- LRS-150-24 power supply
- NPN normally-open proximity switches x6
- Dupont wires
- 4P female motor cables
- 18AWG power wire
- 2-in 8-out terminal blocks x5
- Touch stylus
- Phone clamp
- Three-core power cable with three-pin plug

## Current Driver Wiring

The TB6600 enable signal is active-high in this build.

| Axis | Motors | Driver Current | Microstep | PUL- | DIR- | ENA- |
|---|---:|---:|---:|---:|---:|---:|
| X | 1 | 1.5 A | 8 | D2 | D3 | D4 |
| Y | 2 in parallel | 3.5 A | 8 | D5 | D6 | D7 |
| Z | 1 | 1.5 A | 8 | D8 | D9 | D10 |

Recommended proximity switch mapping:

| Switch | Arduino Pin |
|---|---:|
| X min | A0 |
| X max | A1 |
| Y min | A2 |
| Y max | A3 |
| Z min | A4 |
| Z max | A5 |

## Safety Notes

- Verify TB6600 current settings before powering motors.
- Test one axis at a time with low speed and short movement distance.
- Keep emergency power cutoff physically accessible.
- Confirm motor direction before homing.
- Do not home at full speed until every limit switch is verified.
- Parallel Y motors can rack the gantry if one motor stalls or is wired with opposite phase.

## Protocol

The Python controller currently talks to the motion controller through a line-based TCP command protocol. Each command ends with `\n`; each final response starts with `OK` or `ERROR`.

Supported command examples:

```text
STATUS
HOME
O
X 10
Y -5
Z 2
G X10 Y20 Z5 M1
```

`G X10 Y20 Z5 M1` means move to the absolute target position and perform a click/touch action.

