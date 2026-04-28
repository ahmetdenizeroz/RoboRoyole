/* * BEE FEEDER CONTROLLER - FULL DEBUG VERSION
 * -------------------------------------------------
 * All original Serial.print lines restored for hardware monitoring.
 */

#include <AccelStepper.h>

// ------------------- PINS -------------------
const int STEP_PIN   = 2;
const int DIR_PIN    = 3;
const int ENABLE_PIN = 4;
const int FRONT_STOP_PIN = 7;
const int BACK_STOP_PIN  = 8;

const int ELECTRODE_PIN_BACK   = A1;
const int ELECTRODE_PIN_MIDDLE = A2;
const int ELECTRODE_PIN_FRONT  = A3;

// ------------------- PERIODIC FEEDING -------------------
unsigned long lastAutoFeedTime = 0;
unsigned long autoFeedInterval = 10000UL; //10 sec
bool autoFeedEnabled = false;

// ------------------- STEPPER CONFIG -------------------
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ------------------- STATE MACHINE -------------------
enum State { IDLE, WAITING, EJECTING, HOLDING, RETRACTING };
State state = IDLE;

// --- TIMING ---
unsigned long holdStartTime = 0;
unsigned long holdDuration = 10000UL; 

// ------------------- SENSOR THRESHOLDS -------------------
int THRESHOLD_BACK   = 500;
int THRESHOLD_MIDDLE = 250;
int THRESHOLD_FRONT  = 200;

int HYSTERISIS_BACK = 30;
int HYSTERISIS_MIDDLE = 30;
int HYSTERISIS_FRONT = 30;

int B_TRIGGER = THRESHOLD_BACK +  HYSTERISIS_BACK;
int M_TRIGGER = THRESHOLD_MIDDLE + HYSTERISIS_MIDDLE; 
int F_TRIGGER = THRESHOLD_FRONT + HYSTERISIS_FRONT; 

bool bOn = false; bool mOn = false; bool fOn = false; 

String serialBuffer = "";
const int MAX_SERIAL_BUFFER = 20;
bool isMotorEnabled = false;

// --- DYNAMIC ENVELOPE VARIABLES ---
long numberOfSteps = 0;
long margin = 100; 
long ejectStartPos = 0;
long retractStartPos = 0;
long lastWaitPos = 0;
bool wasSoftStop = false; // Flag to disable refill if sensor failed

void setup() {
  Serial.begin(115200);
  serialBuffer.reserve(MAX_SERIAL_BUFFER);
  stepper.setMaxSpeed(1000);
  stepper.setAcceleration(500);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, HIGH);      
  pinMode(FRONT_STOP_PIN, INPUT_PULLUP);
  pinMode(BACK_STOP_PIN,   INPUT_PULLUP);
  pinMode(ELECTRODE_PIN_BACK,    INPUT);
  pinMode(ELECTRODE_PIN_MIDDLE, INPUT);
  pinMode(ELECTRODE_PIN_FRONT,   INPUT);
  Serial.println("Feeder Ready");
}

void loop() {
  checkLimitSwitches(); 
  readSerialCommands(); 
  updateSensorVars();   
  checkAutoFeedTimer(); 
  runStateMachine();    
}

void runStateMachine() {
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 500) {  //change back to 500
    printStatus();
    lastPrint = millis();
  }

  switch (state) {
    case IDLE:
      disableMotor();
      break;

    case WAITING:
      // Adjust numberOfSteps based on manual sensor-seeking movement
      if (numberOfSteps > 0) {
        long currentWaitPos = stepper.currentPosition();
        if (currentWaitPos != lastWaitPos) {
          numberOfSteps -= (currentWaitPos - lastWaitPos);
          lastWaitPos = currentWaitPos;
        }
      }
      
      if (bOn && !mOn && !fOn) {
        stepper.stop();
        stepper.setCurrentPosition(stepper.currentPosition());
        disableMotor(); 
      }
      else if (!bOn && !mOn && !fOn) {
        //Serial.println("Back Lost, Ejecting"); // RESTORED
        enableMotor();
        stepper.setSpeed(stepper.maxSpeed());
        stepper.runSpeed();
      }
      else {
        //Serial.println("Middle/Front ON, Retracting"); // RESTORED
        enableMotor();
        stepper.setSpeed(-stepper.maxSpeed());
        stepper.runSpeed();
      }
      break;

    case EJECTING:
      enableMotor();
      {
        long currentTravel = stepper.currentPosition() - ejectStartPos;

        bool distLimitReached = (numberOfSteps > 0 && currentTravel >= (numberOfSteps + margin));

        // Stop immediately when all three sensors are ON
        bool sensorsTriggered = (bOn && mOn && fOn);

        if (sensorsTriggered || distLimitReached) {
          wasSoftStop = distLimitReached && !sensorsTriggered;

          if (wasSoftStop) {
            Serial.println("STATUS: Soft Stop (Distance). Starting HOLD.");
          } else {
            Serial.println("STATUS: Full (Sensors). Starting HOLD.");
          }

          state = HOLDING;
          holdStartTime = millis();

          stepper.stop();
          stepper.setCurrentPosition(stepper.currentPosition());
        } 
        else {
          stepper.moveTo(2000000000L); 
          stepper.run();
        }
      }
      break;

    case HOLDING:
      if (millis() - holdStartTime >= holdDuration) {
        Serial.println("STATUS: Hold time over. Retracting.");
        retractStartPos = stepper.currentPosition();
        state = RETRACTING;
        return;
      }
      enableMotor(); 
       //Only refill if we DID NOT stop via distance limit (sensor must be trusted)
      if (!wasSoftStop && bOn && mOn && !fOn) {
         //Serial.println("Front Lost Refill"); // RESTORED
         stepper.setSpeed(stepper.maxSpeed() / 2);
         stepper.runSpeed();
      }
      else {
         stepper.stop();
         stepper.setCurrentPosition(stepper.currentPosition());
      }
      break;

    case RETRACTING:
      enableMotor();
      //if (bOn && !mOn && !fOn) {
      if (!bOn && !fOn) {  
        // Record new calibration distance
        numberOfSteps = abs(retractStartPos - stepper.currentPosition());
        if (wasSoftStop && numberOfSteps >= margin) {
          numberOfSteps -= margin;
        }
        Serial.print("number_of_steps set: "); Serial.println(numberOfSteps);
        
        Serial.println("STATUS: Retract Complete. Entering WAITING.");
        state = WAITING;
        lastWaitPos = stepper.currentPosition();
        stepper.stop();
        stepper.setCurrentPosition(stepper.currentPosition());
      } 
      else {
        stepper.moveTo(-2000000000L);
        stepper.run();
      }
      break;
  }
}

void checkAutoFeedTimer() {
  if (!autoFeedEnabled) return;
  unsigned long now = millis();
  if (now - lastAutoFeedTime >= autoFeedInterval) {
    if (state == IDLE || state == WAITING) {
      Serial.println("AUTO FEED: Triggered");
      ejectStartPos = stepper.currentPosition();
      state = EJECTING;
      lastAutoFeedTime = now;
    } else {
      lastAutoFeedTime = now;
    }
  }
}

void updateSensorVars() {
  int rawB = analogRead(ELECTRODE_PIN_BACK);
  int rawM = analogRead(ELECTRODE_PIN_MIDDLE);
  int rawF = analogRead(ELECTRODE_PIN_FRONT);

  // --- BACK SENSOR LOGIC ---
  if (!bOn && rawB > B_TRIGGER) {
    bOn = true;
  } else if (bOn && rawB < THRESHOLD_BACK) {
    bOn = false;
  }

  // --- MIDDLE SENSOR LOGIC ---
  if (!mOn && rawM > M_TRIGGER) {
    mOn = true;
  } else if (mOn && rawM < THRESHOLD_MIDDLE) {
    mOn = false; 
  }

  // --- FRONT SENSOR LOGIC ---
  if (!fOn && rawF > F_TRIGGER) {
    fOn = true;
  } else if (fOn && rawF < THRESHOLD_FRONT) {
    fOn = false;
  }
}

void checkLimitSwitches() {
  if (digitalRead(FRONT_STOP_PIN) == LOW) {
    Serial.println("FAILSWITCH: FRONT pressed → HARD STOP + reverse");
    hardStopAndReverse(false); 
  } 
  else if (digitalRead(BACK_STOP_PIN) == LOW) {
    Serial.println("FAILSWITCH: BACK pressed → HARD STOP + reverse");
    hardStopAndReverse(true);
  }
}

void hardStopAndReverse(bool moveForwardToClear) {
  stepper.stop();
  stepper.setCurrentPosition(stepper.currentPosition());
  disableMotor();
  delay(200);
  enableMotor();
  long bounceSteps = 1000; 
  if (moveForwardToClear) stepper.move(bounceSteps);
  else stepper.move(-bounceSteps);
  while (stepper.distanceToGo() != 0) { stepper.run(); }
  stepper.stop();
  stepper.setCurrentPosition(stepper.currentPosition());
  disableMotor();
  state = IDLE; 
  Serial.println("FAILSWITCH: Recovered to IDLE");
}

void enableMotor() {
  if (!isMotorEnabled) {
    digitalWrite(ENABLE_PIN, LOW);
    stepper.enableOutputs();
    isMotorEnabled = true;
  }
}

void disableMotor() {
  if (isMotorEnabled) {
    stepper.disableOutputs();
    digitalWrite(ENABLE_PIN, HIGH);
    isMotorEnabled = false;
  }
}

void printStatus() {
  // RESTORED RAW ANALOG VALUES
  Serial.print("SENSORS: B="); Serial.print(bOn ? "ON " : "OFF "); Serial.print(analogRead(ELECTRODE_PIN_BACK));
  Serial.print(" M="); Serial.print(mOn ? "ON " : "OFF "); Serial.print(analogRead(ELECTRODE_PIN_MIDDLE));
  Serial.print(" F="); Serial.print(fOn ? "ON " : "OFF "); Serial.print(analogRead(ELECTRODE_PIN_FRONT));
  Serial.print(" | State: ");
  const char* stateNames[] = {"IDLE", "WAITING", "EJECTING", "HOLDING", "RETRACTING"};
  Serial.println(stateNames[state]);
}

void readSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processCommand(serialBuffer);
      serialBuffer = "";
    } else if (c != '\r' && serialBuffer.length() < MAX_SERIAL_BUFFER) {
      serialBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  if (cmd.length() == 0) return;
  char type = cmd.charAt(0);
  long val  = cmd.substring(1).toInt();

  if (type == 'S') {
    Serial.println("CMD: STOP");
    stepper.stop();
    stepper.setCurrentPosition(stepper.currentPosition());
    disableMotor();
    state = IDLE;
    return;
  }

  switch (type) {
    case 'F': 
    case 'P': 
    
      if (state == IDLE || state == WAITING) {
        Serial.println("CMD: Feed Sequence Start");
        ejectStartPos = stepper.currentPosition();
        state = EJECTING;
        lastAutoFeedTime = millis();
      } else {
          // This covers EJECTING, RETRACTING, and HOLDING
          Serial.println("CMD REJECTED: System busy.");
      }
      break;
    case 'W': 
      Serial.println("CMD: Waiting Mode Active"); // RESTORED
      state = WAITING; 
      lastWaitPos = stepper.currentPosition();
      break;
    case 'L': 
      Serial.println("CMD: Force Retract"); // RESTORED
      retractStartPos = stepper.currentPosition();
      state = RETRACTING; 
      break;

    // RESTORED ALL INDIVIDUAL CMD PRINTS
    case 'V': 
      if (val > 0 && val <= 5000) {
        stepper.setMaxSpeed(val);
        Serial.print("Speed set: "); Serial.println(val);
      }
      break;
    case 'A': 
      if (val > 0 && val <= 5000) {
        stepper.setAcceleration(val);
        Serial.print("Accel set: "); Serial.println(val);
      }
      break;
    case 'T': 
      if (val >= 0 && val <= 1023) {
        THRESHOLD_BACK = val;
        B_TRIGGER = THRESHOLD_BACK +  HYSTERISIS_BACK;
        Serial.print("Thresh Back: "); Serial.println(val);
      }
      break;
    case 'U': 
      if (val >= 0 && val <= 1023) {
        THRESHOLD_MIDDLE = val;
        M_TRIGGER = THRESHOLD_MIDDLE +  HYSTERISIS_MIDDLE; 
        Serial.print("Thresh Mid: "); Serial.println(val);
      }
      break;
    case 'I': 
      if (val >= 0 && val <= 1023) {
        THRESHOLD_FRONT = val; 
        F_TRIGGER = THRESHOLD_FRONT +  HYSTERISIS_FRONT;
        Serial.print("Thresh Front: "); Serial.println(val);
      }
      break;

    case 'G': 
      if (val >= 0 && val <= 1023) {
        HYSTERISIS_BACK = val;
        B_TRIGGER = THRESHOLD_BACK +  HYSTERISIS_BACK;
        Serial.print("Hysterisis Back: "); Serial.println(val);
      }
      break;
    case 'H': 
      if (val >= 0 && val <= 1023) {
        HYSTERISIS_MIDDLE = val;
        M_TRIGGER = THRESHOLD_MIDDLE +  HYSTERISIS_MIDDLE; 
        Serial.print("Hysterisis Mid: "); Serial.println(val);
      }
      break;
    case 'J': 
      if (val >= 0 && val <= 1023) {
        HYSTERISIS_FRONT = val;
        F_TRIGGER = THRESHOLD_FRONT +  HYSTERISIS_FRONT;
        Serial.print("Hysterisis Front: "); Serial.println(val);
      }
      break;

    case 'M': 
      if (val >= 1000 && val <= 60000) {
        holdDuration = val;
        Serial.print("Hold duration: "); Serial.println(val);
      }
      break;
    case 'X':
      Serial.println("Soft Stop Margin set to");
      Serial.println(val); // RESTORED
      margin = val;
      break;
    case 'K': 
      numberOfSteps = 0;
      Serial.println("CMD: Reset Step Calibration");
      break;
  }
}