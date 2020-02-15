#!/bin/bash
echo <password> | sudo -S sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
