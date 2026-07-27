"""
MicroPython port of the Robotic Feeder controller for Raspberry Pi Pico.
"""
import machine
import time
import sys
import select

# ------------------- PINS -------------------
STEP_PIN = 3
DIR_PIN = 2
ENABLE_PIN = 4
FRONT_STOP_PIN = 7
BACK_STOP_PIN = 8

ELECTRODE_PIN_BACK = 28  # ADC2
ELECTRODE_PIN_MIDDLE = 27 # ADC1
ELECTRODE_PIN_FRONT = 26 # ADC0

# ------------------- INIT PINS -------------------
front_stop = machine.Pin(FRONT_STOP_PIN, machine.Pin.IN, machine.Pin.PULL_UP)
back_stop = machine.Pin(BACK_STOP_PIN, machine.Pin.IN, machine.Pin.PULL_UP)

# EXPLICITLY disable internal digital pull-ups before assigning to ADC
p26 = machine.Pin(ELECTRODE_PIN_BACK, machine.Pin.IN, None)
p27 = machine.Pin(ELECTRODE_PIN_MIDDLE, machine.Pin.IN, None)
p28 = machine.Pin(ELECTRODE_PIN_FRONT, machine.Pin.IN, None)

adc_back = machine.ADC(p26)
adc_middle = machine.ADC(p27)
adc_front = machine.ADC(p28)

# ------------------- STEPPER CLASS -------------------
class SimpleStepper:
    def __init__(self, step, dir_p, enable):
        self.step_pin = machine.Pin(step, machine.Pin.OUT)
        self.dir_pin = machine.Pin(dir_p, machine.Pin.OUT)
        self.enable_pin = machine.Pin(enable, machine.Pin.OUT)
        
        self.enable_pin.value(1) # Disabled by default
        
        self.current_pos = 0
        self.target_pos = 0
        self.speed = 1000.0
        self.max_speed = 1000.0
        self.accel = 500.0
        
        self.dir_cw = True
        self.step_interval_us = 1000
        self.last_step_time = time.ticks_us()
        self.is_enabled = False

    def enable_outputs(self):
        if not self.is_enabled:
            self.enable_pin.value(0)
            self.is_enabled = True

    def disable_outputs(self):
        if self.is_enabled:
            self.enable_pin.value(1)
            self.is_enabled = False

    def set_max_speed(self, val):
        self.max_speed = float(val)

    def set_acceleration(self, val):
        self.accel = float(val) # Left simple as it maintains non-blocking code density
        
    def set_speed(self, speed):
        self.speed = speed
        if speed < 0:
            self.dir_cw = False
            self.dir_pin.value(0)
            speed = -speed
        else:
            self.dir_cw = True
            self.dir_pin.value(1)
            
        if speed > 0:
            self.step_interval_us = int(1000000 / speed)
        else:
            self.step_interval_us = 0

    def move_to(self, pos):
        self.target_pos = pos

    def stop(self):
        self.target_pos = self.current_pos

    def current_position(self):
        return self.current_pos

    def set_current_position(self, pos):
        self.current_pos = pos
        self.target_pos = pos

    def run_speed(self):
        if self.step_interval_us == 0: return False
        now = time.ticks_us()
        if time.ticks_diff(now, self.last_step_time) >= self.step_interval_us:
            self.step_pin.value(1)
            time.sleep_us(2)
            self.step_pin.value(0)
            self.current_pos += (1 if self.dir_cw else -1)
            self.last_step_time = now
            return True
        return False
        
    def run(self):
        # Operates instantly at max speed towards the target for simplicity in MicroPython
        if self.current_pos == self.target_pos:
            return False
            
        # Determine direction
        if self.target_pos > self.current_pos and not self.dir_cw:
            self.dir_cw = True
            self.dir_pin.value(1)
            self.step_interval_us = int(1000000 / self.max_speed) if self.max_speed > 0 else 0
        elif self.target_pos < self.current_pos and self.dir_cw:
            self.dir_cw = False
            self.dir_pin.value(0)
            self.step_interval_us = int(1000000 / self.max_speed) if self.max_speed > 0 else 0
            
        return self.run_speed()
        
    def move(self, rel_steps):
        # Blocking move for fail-switches
        self.target_pos = self.current_pos + rel_steps
        if rel_steps < 0:
            self.dir_cw = False
            self.dir_pin.value(0)
        else:
            self.dir_cw = True
            self.dir_pin.value(1)
            
        interval = int(1000000 / self.max_speed)
        steps_left = abs(rel_steps)
        
        while steps_left > 0:
            self.step_pin.value(1)
            time.sleep_us(2)
            self.step_pin.value(0)
            time.sleep_us(interval)
            steps_left -= 1
            self.current_pos += (1 if self.dir_cw else -1)

stepper = SimpleStepper(STEP_PIN, DIR_PIN, ENABLE_PIN)

# ------------------- STATE -------------------
STATE_IDLE = 0
STATE_WAITING = 1
STATE_EJECTING = 2
STATE_HOLDING = 3
STATE_RETRACTING = 4

state = STATE_IDLE
hold_start_time = 0
hold_duration = 10000

last_auto_feed_time = 0
auto_feed_interval = 10000
auto_feed_enabled = False

# ------------------- THRESHOLDS (0-4095) -------------------
THRESHOLD_BACK   = 2000
THRESHOLD_MIDDLE = 1000
THRESHOLD_FRONT  = 800

HYSTERISIS_BACK = 120
HYSTERISIS_MIDDLE = 120
HYSTERISIS_FRONT = 120

B_TRIGGER = THRESHOLD_BACK + HYSTERISIS_BACK
M_TRIGGER = THRESHOLD_MIDDLE + HYSTERISIS_MIDDLE
F_TRIGGER = THRESHOLD_FRONT + HYSTERISIS_FRONT

b_on = False
m_on = False
f_on = False

# ------------------- DYNAMIC VARS -------------------
number_of_steps = 0
margin = 100
eject_start_pos = 0
retract_start_pos = 0
last_wait_pos = 0
was_soft_stop = False

last_sensor_pattern = 255
last_print_time = 0

poller = select.poll()
poller.register(sys.stdin, select.POLLIN)
serial_buffer = ""

# ------------------- FUNCTIONS -------------------

def read_electrode(adc_pin):
    # The Pi Pico runs 3 pins through 1 internal ADC multiplexer. 
    # Reading pins sequentially with high impedance causes charge to "ghost" to the next pin.
    adc_pin.read_u16()  # Dummy read to switch multiplexer channel
    time.sleep_us(100)  # Wait 100 microseconds for internal capacitor to fully drain to ground
    return adc_pin.read_u16() >> 4

def get_sensor_pattern():
    return (4 if b_on else 0) | (2 if m_on else 0) | (1 if f_on else 0)

def sensor_pattern_name(pattern):
    patterns = {
        0: "EMPTY", 4: "WAITING_TARGET", 6: "MIDDLE_REACHED",
        7: "FULL", 2: "SUSPICIOUS_MIDDLE_ONLY", 1: "SUSPICIOUS_FRONT_ONLY",
        3: "SUSPICIOUS_MIDDLE_FRONT", 5: "SUSPICIOUS_BACK_FRONT_GAP"
    }
    return patterns.get(pattern, "UNKNOWN")

def update_sensor_vars():
    global b_on, m_on, f_on, last_sensor_pattern
    
    rawB = read_electrode(adc_back)
    rawM = read_electrode(adc_middle)
    rawF = read_electrode(adc_front)
    
    if not b_on and rawB > B_TRIGGER: b_on = True
    elif b_on and rawB < THRESHOLD_BACK: b_on = False
        
    if not m_on and rawM > M_TRIGGER: m_on = True
    elif m_on and rawM < THRESHOLD_MIDDLE: m_on = False
        
    if not f_on and rawF > F_TRIGGER: f_on = True
    elif f_on and rawF < THRESHOLD_FRONT: f_on = False
        
    pattern = get_sensor_pattern()
    if pattern != last_sensor_pattern:
        last_sensor_pattern = pattern

def hard_stop_and_reverse(is_back_press):
    global state
    stepper.stop()
    stepper.disable_outputs()
    time.sleep_ms(200)
    stepper.enable_outputs()
    
    bounce = 1000
    if is_back_press:  # BACK pressed
        stepper.move(bounce)
    else:              # FRONT pressed
        stepper.move(-bounce)
        
    stepper.stop()
    stepper.disable_outputs()
    state = STATE_IDLE
    print("FAILSWITCH: Recovered to IDLE")

def check_limit_switches():
    if front_stop.value() == 0:
        print("FAILSWITCH: FRONT pressed -> HARD STOP + reverse")
        hard_stop_and_reverse(False)
    elif back_stop.value() == 0:
        print("FAILSWITCH: BACK pressed -> HARD STOP + reverse")
        hard_stop_and_reverse(True)

def process_command(cmd):
    global state, eject_start_pos, retract_start_pos, hold_duration, margin
    global last_auto_feed_time, last_wait_pos, number_of_steps
    global THRESHOLD_BACK, THRESHOLD_MIDDLE, THRESHOLD_FRONT
    global HYSTERISIS_BACK, HYSTERISIS_MIDDLE, HYSTERISIS_FRONT
    global B_TRIGGER, M_TRIGGER, F_TRIGGER
    
    if not cmd: return
    type_char = cmd[0].upper()
    val = int(cmd[1:]) if len(cmd) > 1 else 0

    if type_char == 'S':
        print("CMD: STOP")
        stepper.stop()
        stepper.disable_outputs()
        state = STATE_IDLE
        return

    if type_char in ['F', 'P']:
        if state in [STATE_IDLE, STATE_WAITING]:
            print("CMD: Feed Sequence Start")
            eject_start_pos = stepper.current_position()
            state = STATE_EJECTING
            last_auto_feed_time = time.ticks_ms()
        else:
            print("CMD REJECTED: System busy.")
            
    elif type_char == 'W':
        print("CMD: Waiting Mode Active")
        state = STATE_WAITING
        last_wait_pos = stepper.current_position()
        
    elif type_char == 'L':
        print("CMD: Force Retract")
        retract_start_pos = stepper.current_position()
        state = STATE_RETRACTING

    elif type_char == 'V':
        if 0 < val <= 5000:
            stepper.set_max_speed(val)
            print("Speed set:", val)
    elif type_char == 'A':
        if 0 < val <= 5000:
            stepper.set_acceleration(val)
            print("Accel set:", val)
            
    elif type_char == 'T':
        if 0 <= val <= 4095:
            THRESHOLD_BACK = val
            B_TRIGGER = THRESHOLD_BACK + HYSTERISIS_BACK
            print("Thresh Back:", val)
    elif type_char == 'U':
        if 0 <= val <= 4095:
            THRESHOLD_MIDDLE = val
            M_TRIGGER = THRESHOLD_MIDDLE + HYSTERISIS_MIDDLE
            print("Thresh Mid:", val)
    elif type_char == 'I':
        if 0 <= val <= 4095:
            THRESHOLD_FRONT = val
            F_TRIGGER = THRESHOLD_FRONT + HYSTERISIS_FRONT
            print("Thresh Front:", val)
            
    elif type_char == 'G':
        if 0 <= val <= 4095:
            HYSTERISIS_BACK = val
            B_TRIGGER = THRESHOLD_BACK + HYSTERISIS_BACK
            print("Hysterisis Back:", val)
    elif type_char == 'H':
        if 0 <= val <= 4095:
            HYSTERISIS_MIDDLE = val
            M_TRIGGER = THRESHOLD_MIDDLE + HYSTERISIS_MIDDLE
            print("Hysterisis Mid:", val)
    elif type_char == 'J':
        if 0 <= val <= 4095:
            HYSTERISIS_FRONT = val
            F_TRIGGER = THRESHOLD_FRONT + HYSTERISIS_FRONT
            print("Hysterisis Front:", val)
            
    elif type_char == 'M':
        if 1000 <= val <= 60000:
            hold_duration = val
            print("Hold duration:", val)
    elif type_char == 'X':
        print("Soft Stop Margin set to", val)
        margin = val
    elif type_char == 'K':
        number_of_steps = 0
        print("CMD: Reset Step Calibration")

def read_serial_commands():
    if poller.poll(0):
        cmd = sys.stdin.readline().strip()
        process_command(cmd)

def check_auto_feed():
    global state, eject_start_pos, last_auto_feed_time
    if not auto_feed_enabled: return
    now = time.ticks_ms()
    if time.ticks_diff(now, last_auto_feed_time) >= auto_feed_interval:
        if state in [STATE_IDLE, STATE_WAITING]:
            print("AUTO FEED: Triggered")
            eject_start_pos = stepper.current_position()
            state = STATE_EJECTING
            last_auto_feed_time = now
        else:
            last_auto_feed_time = now

def print_status():
    global last_print_time
    now = time.ticks_ms()
    if time.ticks_diff(now, last_print_time) > 500:
        names = ["IDLE", "WAITING", "EJECTING", "HOLDING", "RETRACTING"]
        pat = sensor_pattern_name(get_sensor_pattern())
        print(f"SENSORS: B={'ON ' if b_on else 'OFF '} {read_electrode(adc_back)} M={'ON ' if m_on else 'OFF '} {read_electrode(adc_middle)} F={'ON ' if f_on else 'OFF '} {read_electrode(adc_front)} | Pattern: {pat} | State: {names[state]}")
        last_print_time = now

def run_state_machine():
    global state, number_of_steps, last_wait_pos, hold_start_time, was_soft_stop, retract_start_pos
    
    print_status()
    
    if state == STATE_IDLE:
        stepper.disable_outputs()
        
    elif state == STATE_WAITING:
        if number_of_steps > 0:
            cur_pos = stepper.current_position()
            if cur_pos != last_wait_pos:
                number_of_steps -= (cur_pos - last_wait_pos)
                last_wait_pos = cur_pos
                
        if b_on and not m_on and not f_on:
            stepper.stop()
            stepper.disable_outputs()
        elif not b_on and not m_on and not f_on:
            stepper.enable_outputs()
            stepper.set_speed(stepper.max_speed)
            stepper.run_speed()
        else:
            stepper.enable_outputs()
            stepper.set_speed(-stepper.max_speed)
            stepper.run_speed()
            
    elif state == STATE_EJECTING:
        stepper.enable_outputs()
        cur_travel = stepper.current_position() - eject_start_pos
        dist_limit = (number_of_steps > 0 and cur_travel >= (number_of_steps + margin))
        sensors_triggered = b_on and m_on and f_on
        
        if sensors_triggered or dist_limit:
            was_soft_stop = dist_limit and not sensors_triggered
            if was_soft_stop:
                print("STATUS: Soft Stop (Distance). Starting HOLD.")
            else:
                print("STATUS: Full (Sensors). Starting HOLD.")
                
            state = STATE_HOLDING
            hold_start_time = time.ticks_ms()
            stepper.stop()
        else:
            stepper.move_to(2000000000)
            stepper.run()
            
    elif state == STATE_HOLDING:
        if time.ticks_diff(time.ticks_ms(), hold_start_time) >= hold_duration:
            print("STATUS: Hold time over. Retracting.")
            retract_start_pos = stepper.current_position()
            state = STATE_RETRACTING
            return
            
        stepper.enable_outputs()
        if not was_soft_stop and b_on and m_on and not f_on:
            stepper.set_speed(stepper.max_speed / 2)
            stepper.run_speed()
        else:
            stepper.stop()
            
    elif state == STATE_RETRACTING:
        stepper.enable_outputs()
        if b_on and not m_on and not f_on:
            number_of_steps = abs(retract_start_pos - stepper.current_position())
            if was_soft_stop and number_of_steps >= margin:
                number_of_steps -= margin
            print("number_of_steps set:", number_of_steps)
            print("STATUS: Retract Complete. Entering WAITING.")
            state = STATE_WAITING
            last_wait_pos = stepper.current_position()
            stepper.stop()
        else:
            stepper.move_to(-2000000000)
            stepper.run()

# ------------------- MAIN LOOP -------------------
print("Feeder Ready")
while True:
    check_limit_switches()
    read_serial_commands()
    update_sensor_vars()
    check_auto_feed()
    run_state_machine()
    # Tiny sleep to let the RP2040 process native background tasks natively
    time.sleep_us(50)
