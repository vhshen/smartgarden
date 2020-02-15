#!/bin/bash
echo <password> | sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'
