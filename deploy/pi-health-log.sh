#!/usr/bin/env bash
# Sample the Pi's power and thermal state next to the recording, once every 5 s.
#
# WHY THIS EXISTS: 15_pi_live.py reads vcgencmd get_throttled ONCE, at startup,
# and only warns. Nothing observes the rail during the run. The throttle bits
# are sticky within a boot but clear on reboot, so an in-flight brownout on the
# drone's UBEC leaves no evidence at all once the Pi is power-cycled -- which is
# exactly what happens when the battery comes out after landing.
#
# Bit meanings from get_throttled (hex):
#   0x1     under-voltage NOW          0x10000  under-voltage HAS OCCURRED
#   0x2     ARM frequency capped NOW   0x20000  frequency capping HAS OCCURRED
#   0x4     currently throttled        0x40000  throttling HAS OCCURRED
#   0x8     soft temperature limit     0x80000  soft limit HAS OCCURRED
# 0x0 is the only clean reading. The 0x1xxxx family is what matters after a
# flight: it says it happened even if the rail has since recovered.
#
# Timestamps are ISO-8601 from the system clock. With no RTC that clock is only
# as good as the last `date -s` or NTP sync, so set the clock BEFORE the flight
# and these lines will align with the detection log's epochs.
set -uo pipefail

OUT="${1:-/home/pi/sar_recordings/health.log}"
INTERVAL="${2:-5}"

mkdir -p "$(dirname "$OUT")"

while true; do
    printf '%s %s %s %s\n' \
        "$(date -Is)" \
        "$(vcgencmd measure_temp 2>/dev/null || echo 'temp=?')" \
        "$(vcgencmd get_throttled 2>/dev/null || echo 'throttled=?')" \
        "$(vcgencmd measure_volts 2>/dev/null || echo 'volt=?')" \
        >> "$OUT"
    sleep "$INTERVAL"
done
