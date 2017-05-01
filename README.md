# CG-BTTC-Arena
Referee program to play games between Back to the Code AIs (Codingame.com). **Linux only**. **1v1 only**. **Ignores BACK**.

## Usage:
* Compile the Arena program
* Have two of your AIs' executable binaries/scripts in the same folder
* Run the Arena program with the names of the AI executables as command line parameters. e.g: Arena V13 V12

## Optional:
* Specify the number of threads as a command line parameter. e.g: Arena V13 V12 2
* Set timeout behavior on or off via the "constexpr bool Timeout" variable. This can be useful as I've noticed timeouts if the computer is being used for something else.

## Notes:
* The error bars on the win rate are approximate. The approximation is good around 50% win rate.
