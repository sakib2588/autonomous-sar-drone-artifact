#!/usr/bin/env bash
# Pre-flight check for the Pi4 field deployment. Run from the laptop before
# leaving for the field, with the Pi powered on.
#
# Every check here exists because the corresponding failure has actually
# happened, or was found by inspection on 2026-08-28:
#   - the Pi's address has drifted by SUBNET three times, and a phone once took
#     over its old address and answered ping while refusing port 22
#   - wifi power save made the Pi vanish while it kept recording perfectly
#   - a TFLite tensor with quantization scale 0.0 makes a naive dequantiser
#     multiply every output by zero: no detections, no error, normal-looking video
#   - the Pi had exactly one saved wifi network, so it would have joined nothing
#     at the field site, and you cannot ssh in to add a network without a network
set -uo pipefail

PI_HOST="pi.local"                       # avahi/mDNS: survives any address change
PI_USER="pi"
PI_OUI_REGEX='dc:a6:32|b8:27:eb|e4:5f:01|2c:cf:67|d8:3a:dd'
SERVICE="sar-pi-live.service"
HEALTH_SERVICE="sar-pi-health.service"
EXPECT_MODEL="tiled416_float32.tflite"
EXPECT_CONF="0.10"
HOME_SSID="Sakib"                        # the network that is NOT usable in the field

pass() { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }
warn() { echo "  [WARN] $1"; }
FAILED=0

echo "== 1. Locating Pi =="
PI_IP=""
# mDNS first. It is immune to the subnet drift that has burned this project
# three times, and it works identically on a field phone hotspot.
if PI_IP=$(getent hosts "$PI_HOST" 2>/dev/null | awk '{print $1}' | head -n1) && [ -n "$PI_IP" ]; then
    pass "resolved $PI_HOST -> $PI_IP via mDNS"
else
    warn "mDNS did not resolve $PI_HOST, falling back to a subnet sweep"
    SUBNET=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}' | cut -d. -f1-3)
    [ -z "$SUBNET" ] && { fail "could not determine local subnet"; exit 1; }
    # Sweep more than once on purpose: the Pi takes about 40 s after power-on to
    # appear in the neighbour table, and on 2026-08-28 the first sweep found
    # nothing while the second found it immediately.
    for attempt in 1 2 3; do
        for i in $(seq 1 254); do (ping -c1 -W1 "${SUBNET}.${i}" >/dev/null 2>&1) & done
        wait 2>/dev/null
        PI_IP=$(ip neigh | grep -iE "$PI_OUI_REGEX" | awk '{print $1}' | head -n1)
        [ -n "$PI_IP" ] && break
        echo "  sweep $attempt found nothing, retrying"
        sleep 5
    done
    # Match on the MAC OUI, never on the address. A phone answering ping at the
    # Pi's old address looks alive and then refuses port 22, which reads exactly
    # like an sshd fault and is not one.
    [ -z "$PI_IP" ] && { fail "Pi not found by MAC OUI after 3 sweeps -- power it on and retry"; exit 1; }
    pass "Pi found at $PI_IP by MAC OUI"
fi

SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes ${PI_USER}@${PI_IP}"
$SSH true 2>/dev/null || { fail "ssh to ${PI_USER}@${PI_IP} failed"; exit 1; }

echo "== 2. Services =="
for svc in "$SERVICE" "$HEALTH_SERVICE"; do
    read -r ACTIVE ENABLED < <($SSH "systemctl is-active $svc; systemctl is-enabled $svc" 2>/dev/null | tr '\n' ' ')
    [ "$ACTIVE" = "active" ]   && pass "$svc active"              || fail "$svc is '$ACTIVE', expected active"
    [ "$ENABLED" = "enabled" ] && pass "$svc enabled (autostarts)" || fail "$svc is '$ENABLED', expected enabled"
done

echo "== 3. Deployed model + threshold =="
UNIT=$($SSH "systemctl cat $SERVICE 2>/dev/null")
echo "$UNIT" | grep -q "$EXPECT_MODEL" \
    && pass "model is $EXPECT_MODEL" \
    || fail "unit does not reference $EXPECT_MODEL (INT8 costs 12.8 pp mAP50 and 39% of small-object AP)"
echo "$UNIT" | grep -q -- "--conf-thres $EXPECT_CONF" \
    && pass "conf-thres is $EXPECT_CONF" \
    || fail "conf-thres is not $EXPECT_CONF (0.25 recovers 0.3009 of people against 0.3865 at 0.10)"
echo "$UNIT" | grep -q -- "--tile" \
    && pass "tiling enabled" \
    || fail "--tile missing: untiled cross-domain AP50 is 0.0449, effectively blind"
echo "$UNIT" | grep -q -- "--calibrate-seconds 0" \
    && pass "startup calibration disabled (first-segment FPS header is correct)" \
    || warn "calibration enabled: segments 1-2 of each boot get a wrong mp4 FPS header"

echo "== 4. Wifi power save =="
# NOT `iw`: it is not installed on this Pi and the check silently false-FAILed.
ACTIVE_CON=$($SSH "nmcli -t -f NAME con show --active 2>/dev/null | head -1")
PS=$($SSH "nmcli -g 802-11-wireless.powersave con show '$ACTIVE_CON' 2>/dev/null")
case "$PS" in
    2|disable) pass "power save disabled on '$ACTIVE_CON'" ;;
    *)
        if $SSH "grep -q 'wifi.powersave *= *2' /etc/NetworkManager/conf.d/*.conf 2>/dev/null"; then
            pass "power save disabled globally via NetworkManager conf.d"
        else
            fail "power save is '$PS' on '$ACTIVE_CON' -- the Pi will vanish from the network under load while still recording"
        fi
        ;;
esac

echo "== 5. A field network is saved =="
# The blocker found on 2026-08-28: only the home SSID was saved, so the Pi would
# have joined nothing at the field. This cannot be fixed on site.
SAVED=$($SSH "nmcli -t -f NAME,TYPE con show 2>/dev/null | awk -F: '\$2==\"802-11-wireless\"{print \$1}'")
FIELD=$(echo "$SAVED" | grep -v "^${HOME_SSID}$" | grep -v '^$')
if [ -n "$FIELD" ]; then
    pass "non-home wifi saved: $(echo "$FIELD" | tr '\n' ' ')"
else
    fail "ONLY '$HOME_SSID' is saved. Add the field hotspot NOW -- you cannot add wifi in the field without wifi"
fi

echo "== 6. Disk =="
DISKLINE=$($SSH "df -h / | tail -1")
AVAIL_G=$($SSH "df -BG --output=avail / | tail -1 | tr -dc '0-9'")
echo "  $DISKLINE"
# Gigabytes, not percent: percent-free says nothing about how many minutes of
# recording remain. At ~44 MB/h, 1 GB is about 22 h, which is plenty.
[ "${AVAIL_G:-0}" -ge 1 ] && pass "${AVAIL_G} GB free" || fail "only ${AVAIL_G} GB free -- archive old segments"

echo "== 7. Camera =="
$SSH "test -e /dev/video0" && pass "/dev/video0 present" || fail "/dev/video0 missing -- reseat the USB camera"

echo "== 8. Clock =="
# Query separately: `timedatectl show` emits properties in its own canonical
# order, not the order they were asked for, so a positional read swaps them.
TZ=$($SSH "timedatectl show -p Timezone --value" 2>/dev/null)
NTPSYNC=$($SSH "timedatectl show -p NTPSynchronized --value" 2>/dev/null)
echo "  timezone=$TZ ntp_synced=$NTPSYNC"
[ "$TZ" = "Asia/Dhaka" ] && pass "timezone correct" || fail "timezone is '$TZ', expected Asia/Dhaka"
# This board has no RTC. In the field with no NTP the clock restores from
# fake-hwclock, so push the laptop's time across before flying or the segment
# filenames collide and one flight overwrites another.
[ "$NTPSYNC" = "yes" ] && pass "clock NTP-synced" \
    || warn "clock not NTP-synced -- run: ssh $PI_USER@$PI_HOST \"sudo date -s '\$(date '+%Y-%m-%d %H:%M:%S')' && sudo fake-hwclock save\""

echo "== 9. Detector actually produces output =="
# Look at the whole log, not a 50-line tail. Non-empty detections run about 3.5%
# of lines on typical footage, so a short tail usually contains none and the
# check silently degrades to a meaningless WARN.
LOG=/home/pi/sar_recordings/detections.jsonl
# Do NOT write `grep -c ... || echo 0`: grep with no matches prints "0" AND exits
# 1, so the fallback appends a second zero and the [ test sees "0\n0" and dies
# with "integer expression expected". Strip to digits instead.
NLINES=$($SSH "wc -l < $LOG 2>/dev/null"); NLINES=${NLINES//[!0-9]/}
NDET=$($SSH "grep -ac '\"detections\": \[{' $LOG 2>/dev/null"); NDET=${NDET//[!0-9]/}
if [ "${NLINES:-0}" -eq 0 ]; then
    fail "$LOG is empty or missing -- the payload has never logged an inference"
elif [ "${NDET:-0}" -gt 0 ]; then
    CONFS=$($SSH "grep -ao '\"conf\": [0-9.]*' $LOG 2>/dev/null | sort -t: -k2 -g | sed -n '1p;\$p' | tr '\n' ' '")
    pass "$NDET non-empty detections in $NLINES lines; conf range: $CONFS"
else
    # An empty fresh log means only that nothing is in front of the camera, and
    # that stays true however long it runs -- so do NOT gate this on line count.
    # The question that actually matters is whether the model+decode path has
    # EVER produced a detection. Archived evidence answers that; the size of the
    # current log does not. An earlier version gated on 2000 lines and duly
    # false-FAILed overnight, which is how a check trains you to ignore it.
    # grep -c prints one count PER FILE, so pipe the matches through wc -l to get
    # a single integer instead of a multi-line string the [ test chokes on.
    ARCH=$($SSH "grep -ah '\"detections\": \[{' /home/pi/sar_recordings/_prefield_desk/*.jsonl 2>/dev/null | wc -l")
    ARCH=${ARCH//[!0-9]/}
    if [ "${ARCH:-0}" -gt 0 ]; then
        warn "0 detections in $NLINES fresh lines, but $ARCH in the archive, so the model path works."
        warn "  Nothing is in front of the camera. Confirm with the 30 m person test before flying."
    else
        fail "$NLINES lines, 0 detections here and 0 archived -- suspect the TFLite scale=0 dequant bug. Do NOT fly"
    fi
fi

echo "== 10. Power and thermal =="
THROT=$($SSH "vcgencmd get_throttled" 2>/dev/null)
TEMP=$($SSH "vcgencmd measure_temp" 2>/dev/null)
echo "  $TEMP $THROT"
[ "$THROT" = "throttled=0x0" ] && pass "power clean, no throttling" \
    || fail "$THROT -- any nonzero value means the 5V rail sagged or the SoC got hot"

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "ALL CHECKS PASSED -- Pi ready for the field."
    echo "Still yours to do on site: set the clock, verify throttled=0x0 on BATTERY power"
    echo "with props spinning, and stop the service gracefully before pulling power."
else
    echo "ONE OR MORE CHECKS FAILED -- fix before leaving."
fi
exit $FAILED
